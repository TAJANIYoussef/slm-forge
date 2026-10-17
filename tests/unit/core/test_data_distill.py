"""Distillation gates, PII redaction, the manifest and the distilled-data eval guard."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from slmforge.config.schema import DistillConfig
from slmforge.data.distill import (
    DISTILL_MANIFEST_NAME,
    DataEgressNotAllowed,
    DistillationDisabled,
    DistilledDataRejected,
    DistillGateError,
    DistillManifest,
    ToSNotAcknowledged,
    assert_not_distilled,
    check_gates,
    redact_pii,
    run_distillation,
)

BASE: dict[str, Any] = {
    "provider": "anthropic",
    "model": "teacher-model-id",
    "input": "in.jsonl",
    "output": "out.jsonl",
    "prompt_template": "prompt.md",
}
OPEN: dict[str, Any] = {
    **BASE,
    "enabled": True,
    "tos_acknowledged": True,
    "terms_reviewed_on": date(2026, 9, 1),
    "acknowledged_by": "maintainer",
    "allow_send_data": True,
}
TODAY = date(2026, 10, 1)


def unvalidated(**changes: Any) -> DistillConfig:
    """A config that skipped DistillConfig's own validator (model_copy does not validate)."""
    return DistillConfig.model_validate(OPEN).model_copy(update=changes)


# --------------------------------------------------------------------------- gates


def test_all_gates_open() -> None:
    check_gates(DistillConfig.model_validate(OPEN), today=TODAY)


def test_disabled_is_the_shipped_default() -> None:
    with pytest.raises(DistillationDisabled, match="enabled: false"):
        check_gates(DistillConfig.model_validate(BASE), today=TODAY)


@pytest.mark.parametrize(
    "changes",
    [
        {"tos_acknowledged": False},
        {"terms_reviewed_on": None},
        {"acknowledged_by": None},
        {"acknowledged_by": " "},
    ],
)
def test_tos_gate(changes: dict[str, Any]) -> None:
    with pytest.raises(ToSNotAcknowledged):
        check_gates(unvalidated(**changes), today=TODAY)


def test_review_date_in_the_future() -> None:
    with pytest.raises(ToSNotAcknowledged, match="future"):
        check_gates(unvalidated(terms_reviewed_on=date(2026, 12, 1)), today=TODAY)


def test_egress_gate() -> None:
    with pytest.raises(DataEgressNotAllowed, match=r"allow_send_data.*09-08"):
        check_gates(unvalidated(allow_send_data=False), today=TODAY)


def test_gate_errors_share_a_base() -> None:
    for exc in (DistillationDisabled, ToSNotAcknowledged, DataEgressNotAllowed):
        assert issubclass(exc, DistillGateError)


def test_run_distillation_checks_gates_before_anything() -> None:
    with pytest.raises(DistillationDisabled):
        run_distillation(DistillConfig.model_validate(BASE), client=None, eval_guard=None)  # type: ignore[arg-type]


def test_shipped_distill_config_is_closed(repo_root: Path) -> None:
    import yaml

    cfg = DistillConfig.model_validate(
        yaml.safe_load((repo_root / "configs/distill.yaml").read_text("utf-8"))
    )
    assert (cfg.enabled, cfg.tos_acknowledged, cfg.allow_send_data, cfg.redact_pii) == (
        False,
        False,
        False,
        True,
    )
    with pytest.raises(DistillationDisabled):
        check_gates(cfg)


# --------------------------------------------------------------------------- PII


@pytest.mark.parametrize(
    ("text", "expected", "kinds"),
    [
        ("mail: user.name+tag@atlas-fictive.ma.", "mail: [EMAIL].", ["email"]),
        ("CIN BE123456 ou A12345", "CIN [CIN] ou [CIN]", ["cin", "cin"]),
        ("tel 0612345678 wla +212612345678", "tel [PHONE] wla [PHONE]", ["phone", "phone"]),
        ("appelez le 06 12 34 56 78", "appelez le [PHONE]", ["phone"]),
        ("+212 6 12 34 56 78 / 00212522123456", "[PHONE] / [PHONE]", ["phone", "phone"]),
        ("fixe: 05-22-12-34-56", "fixe: [PHONE]", ["phone"]),
        ("RIB 011780000012345678901234 merci", "RIB [RIB] merci", ["rib"]),
        ("RIB 011 780 0000123456789012 34", "RIB [RIB]", ["rib"]),
        ("IBAN MA64011519000001205000534921", "IBAN [IBAN]", ["iban"]),
        ("IBAN MA64 0115 1900 0001 2050 0053 4921 ok", "IBAN [IBAN] ok", ["iban"]),
        ("رقم التيليفون ديالي ٠٦١٢٣٤٥٦٧٨", "رقم التيليفون ديالي [PHONE]", ["phone"]),
        ("contact.AB123456@x.ma", "[EMAIL]", ["email"]),
    ],
)
def test_redact_pii(text: str, expected: str, kinds: list[str]) -> None:
    out, found = redact_pii(text)
    assert out == expected
    assert found == kinds


@pytest.mark.parametrize(
    "text",
    [
        "Le taux est de 70 % et le dépôt de 1 000 MAD.",
        "loi 31-08, article 14, 2019 2020 2021 2022 2023 2024",
        "0812345678 n'est pas un mobile marocain",
        "ch7al howa l'LTV? 3la 7sab l9anoun, 70 f lmya",
        "Article 12 de la loi 77-19, en vigueur le 15/03/2024",
        "ABC123456 et AB1234",
    ],
)
def test_redact_pii_leaves_ordinary_text_alone(text: str) -> None:
    assert redact_pii(text) == (text, [])


def test_redact_pii_is_idempotent() -> None:
    once, kinds = redact_pii("BE123456, 0612345678, a@b.ma, MA64011519000001205000534921")
    assert len(kinds) == 4
    assert redact_pii(once) == (once, [])


# --------------------------------------------------------------------------- manifest


def manifest(**changes: Any) -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "model": "teacher-model-id",
        "started_at": "2026-10-01T00:00:00Z",
        "terms_reviewed_on": "2026-09-01",
        "acknowledged_by": "maintainer",
        "temperature": 0.7,
        "n_requested": 10,
        "n_kept": 7,
        "n_rejected_langid": 2,
        "n_rejected_overlap": 1,
        "n_redactions": 3,
        "prompt_template_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "output_sha256": "c" * 64,
        **changes,
    }


def test_manifest_roundtrip_and_strictness() -> None:
    m = DistillManifest.model_validate(manifest())
    assert DistillManifest.model_validate_json(m.model_dump_json()) == m
    assert "MIT" in m.usage_note
    with pytest.raises(ValidationError):
        DistillManifest.model_validate(manifest(extra_field=1))
    with pytest.raises(ValidationError):
        DistillManifest.model_validate(manifest(n_kept=-1))
    with pytest.raises(ValidationError):
        DistillManifest.model_validate(manifest(acknowledged_by=""))


# --------------------------------------------------------------------------- eval guard


def write_jsonl(path: Path, rows: list[Any]) -> Path:
    path.write_text("// comment\n\n" + "".join(json.dumps(r) + "\n" for r in rows) + "not json\n", "utf-8")
    return path


def test_human_files_pass(tmp_path: Path, repo_root: Path) -> None:
    assert_not_distilled(
        write_jsonl(tmp_path / "gt.jsonl", [{"id": "a", "provenance": {"kind": "human"}}, [1]])
    )
    assert_not_distilled(repo_root / "data/eval/ground_truth.sample.jsonl")
    assert_not_distilled(repo_root / "data/train/sample_chat.jsonl")


@pytest.mark.parametrize(
    "row",
    [
        {"id": "b", "provenance": {"kind": "distilled"}},
        {"id": "b", "meta": {"provenance": {"kind": "distilled"}}},
    ],
)
def test_distilled_records_are_rejected(tmp_path: Path, row: dict[str, Any]) -> None:
    path = write_jsonl(tmp_path / "gt.jsonl", [{"id": "a"}, row])
    with pytest.raises(DistilledDataRejected, match="line 4"):
        assert_not_distilled(path)


def test_file_listed_in_a_distill_manifest_is_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "out.jsonl", [{"id": "a"}])  # provenance stripped by hand
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / DISTILL_MANIFEST_NAME).write_text(json.dumps(manifest(output_sha256=sha)), "utf-8")
    with pytest.raises(DistilledDataRejected, match="listed as the output"):
        assert_not_distilled(path)
    (tmp_path / DISTILL_MANIFEST_NAME).write_text(json.dumps(manifest()), "utf-8")  # another file
    assert_not_distilled(path)
    (tmp_path / DISTILL_MANIFEST_NAME).write_text("[]", "utf-8")  # malformed manifest
    assert_not_distilled(path)
