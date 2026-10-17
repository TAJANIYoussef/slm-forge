"""protocol: hash, drift detection, lock and load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from slmforge.protocol import Protocol, ProtocolDrift, load_locked, lock_protocol

H = "0" * 64


def _proto(**changes: Any) -> Protocol:
    fields: dict[str, Any] = {
        "suite": "factual-reliability",
        "scorer_version": "1.0.0",
        "normalizer_version": "1.0.0",
        "matcher_version": "1.0.0",
        "outcome_table_version": "1.0.0",
        "lexicon_version": "v1",
        "lexicon_sha256": H,
        "prompt_template_id": "default_v1",
        "prompt_template_sha256": H,
        "prompt_mode": "abstain_ok",
        "generation_backend": "hf",
        "decoding_sha256": H,
        "scoring_sha256": H,
        "tau": 0.5,
        "tau_sweep": (0.0, 0.3, 0.5, 0.7, 0.9),
        "unanswerable_no_cue_policy": "residual",
        "utility_weights_sha256": H,
        "primary_metrics": ("factual_accuracy", "confabulation_rate", "abstention_rate"),
        "alpha": 0.05,
        "bootstrap_resamples": 2000,
        "seed": 1234,
        "dataset_sha256": H,
        "split": "test",
    }
    fields.update(changes)
    return Protocol(**fields)


def test_sha256_is_stable_and_short_is_a_prefix() -> None:
    p = _proto()
    assert p.sha256() == _proto().sha256()
    assert len(p.sha256()) == 64
    assert p.short() == p.sha256()[:8]


def test_sha256_ignores_construction_order() -> None:
    a = _proto()
    b = Protocol(**dict(reversed(list(a.model_dump().items()))))
    assert a.sha256() == b.sha256()


@pytest.mark.parametrize(
    "change",
    [
        {"tau": 0.6},
        {"lexicon_sha256": "1" * 64},
        {"subsample_ids_sha256": H},
        {"judge_id": "j@rev"},
        {"tau_sweep": (0.5,)},
        {"prompt_mode": "neutral"},
    ],
)
def test_any_field_change_changes_the_hash(change: dict[str, Any]) -> None:
    assert _proto(**change).sha256() != _proto().sha256()


def test_protocol_is_frozen_and_strict() -> None:
    p = _proto()
    with pytest.raises(ValidationError):
        p.tau = 0.1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        _proto(unknown_field=1)
    with pytest.raises(ValidationError):
        _proto(generation_backend="openai")


def test_diff_lists_differing_fields() -> None:
    a, b = _proto(), _proto(tau=0.7, split="dev")
    assert a.diff(b) == {"split": ("test", "dev"), "tau": (0.5, 0.7)}
    assert a.diff(a) == {}


def test_lock_and_load_roundtrip(tmp_path: Path) -> None:
    p = _proto()
    lock = tmp_path / "locks" / "fr-v1.lock.json"
    lock_protocol(p, lock)
    doc = json.loads(lock.read_text(encoding="utf-8"))
    assert doc["sha256"] == p.sha256() and doc["short"] == p.short()
    assert load_locked(lock) == p
    assert load_locked(lock, live=_proto()) == p


def test_load_locked_raises_drift_listing_fields(tmp_path: Path) -> None:
    lock = tmp_path / "p.lock.json"
    lock_protocol(_proto(), lock)
    live = _proto(tau=0.7, lexicon_sha256="f" * 64)
    with pytest.raises(ProtocolDrift) as ei:
        load_locked(lock, live=live)
    assert set(ei.value.fields) == {"tau", "lexicon_sha256"}
    assert ei.value.fields["tau"] == (0.5, 0.7)
    assert ei.value.lock_path == lock
    msg = str(ei.value)
    assert "tau" in msg and "lexicon_sha256" in msg and "2 field(s)" in msg


def test_relocking_same_protocol_is_a_no_op(tmp_path: Path) -> None:
    lock = tmp_path / "p.lock.json"
    lock_protocol(_proto(), lock)
    before = lock.read_text(encoding="utf-8")
    lock_protocol(_proto(), lock)
    assert lock.read_text(encoding="utf-8") == before


def test_lock_refuses_to_replace_a_different_protocol(tmp_path: Path) -> None:
    lock = tmp_path / "p.lock.json"
    lock_protocol(_proto(), lock)
    with pytest.raises(ProtocolDrift) as ei:
        lock_protocol(_proto(seed=1), lock)
    assert set(ei.value.fields) == {"seed"}
    lock_protocol(_proto(seed=1), lock, overwrite=True)
    assert load_locked(lock).seed == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_hand_edited_lock_is_rejected(tmp_path: Path) -> None:
    lock = tmp_path / "p.lock.json"
    lock_protocol(_proto(), lock)
    doc = json.loads(lock.read_text(encoding="utf-8"))
    doc["protocol"]["tau"] = 0.9
    lock.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        load_locked(lock)


def test_not_a_lock_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tau": 0.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a protocol lock"):
        load_locked(bad)


def test_stats_settings_change_the_protocol() -> None:
    assert _proto(min_bucket_n=5).sha256() != _proto().sha256()
    assert _proto(generation_backend="external").sha256() != _proto().sha256()
