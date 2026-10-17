"""Packaged resources load and validate: lexicons, units; loaders reject malformed files."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from slmforge.text.lexicons import LexiconError, LexiconSet, skeleton_literal_len
from slmforge.text.numbers import UnitTable, UnitTableError, within
from slmforge.types import LEX_BUCKETS


def test_packaged_lexicons_load() -> None:
    lex = LexiconSet.load()
    assert lex.version == "v1" and len(lex.sha256) == 64
    assert {e.bucket for e in lex.entries} == set(LEX_BUCKETS)
    assert {"abstain", "premise", "hedge", "filler", "contrast"} == {e.kind for e in lex.entries}
    assert all(e.hedge_kind for e in lex.by_kind("hedge"))
    weak = {e.id for e in lex.entries if e.strength == "weak"}
    assert "en.abstain.consult" in weak and "dar_a.abstain.allah_a3lam" in weak


def test_lexicon_hash_is_stable() -> None:
    assert LexiconSet.load().sha256 == LexiconSet.load().sha256


def test_polarity() -> None:
    lex = LexiconSet.load()
    assert lex.polarity("oui") == "yes"
    assert lex.polarity("la") == "no"
    assert lex.polarity("la", first_token=False) is None  # French article
    assert lex.polarity("بنك") is None


def _write(root: Path, body: str) -> Path:
    d = root / "abstain"
    d.mkdir(parents=True)
    (d / "en.yaml").write_text(body, encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("body", "msg"),
    [
        (
            "version: 1\nkind: abstain\nbucket: en\nentries:\n- {id: x, pattern: '(unclosed'}\n",
            "invalid regex",
        ),
        ("version: 1\nkind: abstain\nbucket: en\nentries:\n- {id: x, pattern: 'sûr'}\n", "L1"),
        ("version: 1\nkind: abstain\nbucket: en\nentries:\n- {id: x, skeleton: '\\bmkn\\b'}\n", "literal"),
        (
            "version: 1\nkind: abstain\nbucket: en\nentries:\n- {id: x, pattern: a}\n- {id: x, pattern: b}\n",
            "duplicate",
        ),
        ("version: 1\nkind: abstain\nbucket: xx\nentries: []\n", "bucket"),
        ("version: 1\nkind: abstain\nbucket: en\nentries:\n- {id: x, pattern: a, colour: red}\n", "unknown"),
    ],
)
def test_lexicon_validation(tmp_path: Path, body: str, msg: str) -> None:
    with pytest.raises(LexiconError, match=msg):
        LexiconSet.load(root=_write(tmp_path, body))


def test_skeleton_literal_len() -> None:
    assert skeleton_literal_len(r"\bm ?(?:kn ?)?Erft?C\b") == 8
    assert skeleton_literal_len(r"\bmkn\b") == 3


def test_units_load() -> None:
    t = UnitTable.load()
    assert t.by_surface["f lmya"] == "%"
    assert t.by_surface["درهم"] == "MAD"
    assert t.conversions[("years", "months")] == 12
    assert t.darija_money_scale is not None and t.darija_money_scale.factor == 10000


def test_units_reject_non_l1(tmp_path: Path) -> None:
    p = tmp_path / "units.yaml"
    p.write_text("version: 1\nunits:\n  '%': {kind: percent, surfaces: ['فالمية']}\n", encoding="utf-8")
    with pytest.raises(UnitTableError, match="L1"):
        UnitTable.load(p)


@pytest.mark.parametrize(
    ("a", "g", "abs_tol", "rel_tol", "ok"),
    [
        ("70", "70", "0", "0", True),
        ("70.0", "70", "0", "0", True),
        ("70.5", "70", "0.5", "0", True),
        ("70.51", "70", "0.5", "0", False),
        ("1010", "1000", "0", "0.01", True),
        ("1010.01", "1000", "0", "0.01", False),
        ("0.1", "0.3", "0.2", "0", True),  # exact Decimal arithmetic, no float error
    ],
)
def test_within(a: str, g: str, abs_tol: str, rel_tol: str, ok: bool) -> None:
    assert within(Decimal(a), Decimal(g), Decimal(abs_tol), Decimal(rel_tol)) is ok


def test_within_rejects_floats_and_negative_tolerance() -> None:
    with pytest.raises(TypeError):
        within(0.1, Decimal("0.1"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        within(Decimal(1), Decimal(1), Decimal(-1))
