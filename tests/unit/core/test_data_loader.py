"""ChatRecord invariants and language sampling weights (torch-free)."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from slmforge.data.loader import ChatRecord, language_sampling_weights


def chat(*turns: tuple[str, str], **kw: Any) -> dict[str, Any]:
    return {"id": "r1", "messages": [{"role": r, "content": c} for r, c in turns], **kw}


def test_valid_record_and_defaults() -> None:
    rec = ChatRecord.model_validate(chat(("system", "s"), ("user", "q"), ("assistant", "a")))
    assert rec.language is None
    assert rec.provenance == {}
    assert rec.sampling_key("lang_script") == "unknown"


@pytest.mark.parametrize(
    ("turns", "match"),
    [
        ((("user", "q"),), "last turn"),
        ((("user", "q"), ("assistant", "a"), ("user", "q2")), "last turn"),
        ((("user", "q"), ("system", "s"), ("assistant", "a")), "system turn"),
        ((("tool", "x"), ("assistant", "a")), "unknown role"),
    ],
)
def test_message_invariants(turns: tuple[tuple[str, str], ...], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        ChatRecord.model_validate(chat(*turns))


def test_message_keys_and_empty_messages() -> None:
    with pytest.raises(ValidationError, match="role and content"):
        ChatRecord.model_validate({"id": "r", "messages": [{"role": "assistant", "content": "a", "x": "y"}]})
    with pytest.raises(ValidationError):
        ChatRecord.model_validate({"id": "r", "messages": []})


def test_unknown_fields_and_bad_provenance() -> None:
    with pytest.raises(ValidationError):
        ChatRecord.model_validate(chat(("user", "q"), ("assistant", "a"), meta={}))
    with pytest.raises(ValidationError, match=r"provenance\.kind"):
        ChatRecord.model_validate(chat(("user", "q"), ("assistant", "a"), provenance={"kind": "scraped"}))
    rec = ChatRecord.model_validate(
        chat(("user", "q"), ("assistant", "a"), provenance={"kind": "distilled", "model": "teacher"})
    )
    assert rec.provenance["kind"] == "distilled"


def test_fr_en_cannot_be_arabic_script() -> None:
    with pytest.raises(ValidationError, match="script=arab"):
        ChatRecord.model_validate(chat(("user", "q"), ("assistant", "a"), language="fr", script="arab"))


@pytest.mark.parametrize(
    ("language", "script", "lang_script", "lang"),
    [
        ("darija", "arab", "darija_arab", "darija"),
        ("darija", "latn", "darija_latn", "darija"),
        ("darija", None, "darija", "darija"),
        ("fr", None, "fr", "fr"),
        ("en", "latn", "en", "en"),
    ],
)
def test_sampling_key(language: str, script: str | None, lang_script: str, lang: str) -> None:
    rec = ChatRecord.model_validate(chat(("user", "q"), ("assistant", "a"), language=language, script=script))
    assert rec.sampling_key("lang_script") == lang_script
    assert rec.sampling_key("language") == lang


def test_sample_chat_file_is_valid(repo_root: Path) -> None:
    lines = (repo_root / "data/train/sample_chat.jsonl").read_text(encoding="utf-8").splitlines()
    records = [ChatRecord.model_validate(json.loads(line)) for line in lines if line.strip()]
    assert len(records) == 6
    assert len({r.id for r in records}) == 6
    assert {r.sampling_key("lang_script") for r in records} == {"en", "fr", "darija_arab", "darija_latn"}
    assert all(r.provenance.get("kind") == "human" for r in records)


# --------------------------------------------------------------------------- sampling weights


def test_alpha_one_keeps_natural_proportions() -> None:
    w = language_sampling_weights({"fr": 600, "en": 300, "darija_latn": 100}, 1.0)
    assert w == pytest.approx({"darija_latn": 0.1, "en": 0.3, "fr": 0.6})
    assert list(w) == ["darija_latn", "en", "fr"]  # sorted keys


def test_alpha_below_one_upsamples_small_buckets() -> None:
    counts = {"fr": 900, "darija_arab": 100}
    w = language_sampling_weights(counts, 0.5)
    assert w["darija_arab"] == pytest.approx(10 / 40)  # sqrt(100) / (sqrt(900) + sqrt(100))
    assert w["darija_arab"] > counts["darija_arab"] / sum(counts.values())


def test_zero_count_gets_zero_weight() -> None:
    assert language_sampling_weights({"fr": 4, "en": 0}, 0.3) == {"en": 0.0, "fr": 1.0}


@pytest.mark.parametrize(
    ("counts", "alpha"),
    [({}, 1.0), ({"fr": 0}, 1.0), ({"fr": -1, "en": 3}, 1.0), ({"fr": 1}, 0.0), ({"fr": 1}, math.nan)],
)
def test_invalid_inputs(counts: dict[str, int], alpha: float) -> None:
    with pytest.raises(ValueError):
        language_sampling_weights(counts, alpha)


@given(
    st.dictionaries(
        st.sampled_from(["en", "fr", "darija_arab", "darija_latn", "darija_mixed"]),
        st.integers(0, 10**6),
        min_size=1,
    ),
    st.floats(0.05, 1.0),
)
def test_weights_are_a_distribution_and_monotone(counts: dict[str, int], alpha: float) -> None:
    if not any(counts.values()):
        return
    w = language_sampling_weights(counts, alpha)
    assert math.isclose(math.fsum(w.values()), 1.0, rel_tol=1e-9)
    items = sorted(counts, key=counts.__getitem__)
    for a, b in itertools.pairwise(items):
        assert w[a] <= w[b] + 1e-12  # more data never means a lower probability
