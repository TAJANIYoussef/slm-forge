"""Ground-truth schema: per-item rules, legacy upgrade, cross-item rules, hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slmforge.eval.dataset import GroundTruthError, GroundTruthItem, load_ground_truth, parse_lines

BASE = {
    "id": "fin-fr-0001",
    "prompt": "Quel est le ratio LTV maximal du Prêt Atlas ?",
    "language": "fr",
    "domain": "finance",
    "expected_answer": "70%",
    "answer_type": "percent",
    "numeric": {"value": 70, "unit": "%"},
}
UNANSWERABLE = {
    "id": "leg-en-u-0001",
    "prompt": "What does Article 912 of the Fictional Family Code say?",
    "language": "en",
    "domain": "legal",
    "expected_answer": None,
    "unanswerable_reason": "nonexistent_entity",
}


def item(**kw: object) -> GroundTruthItem:
    return GroundTruthItem.model_validate({**BASE, **kw})


def test_resolves_derived_fields() -> None:
    i = item()
    assert i.fact_id == i.id
    assert i.script == "latn"
    assert i.answerable is True
    assert i.lang_script == "fr"
    assert len(i.content_hash) == 16


def test_null_expected_answer_means_unanswerable() -> None:
    i = GroundTruthItem.model_validate(UNANSWERABLE)
    assert i.answerable is False
    assert i.allow_abstention is True


def test_expected_answer_key_is_required() -> None:
    items, issues = parse_lines([json.dumps({k: v for k, v in BASE.items() if k != "expected_answer"})])
    assert not items
    assert [i.code for i in issues] == ["E03"]


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"expected_answer": "  "}, "E03"),
        ({"answerable": False}, "E04"),
        ({"answer_type": "percent", "numeric": None}, "E08"),
        ({"answer_type": "date"}, "E09"),
        ({"aliases": [""]}, "E14"),
        ({"language": "en", "script": "arab"}, "E07"),
        ({"language": "darija"}, "E07"),
        ({"unanswerable_reason": "false_premise"}, "E06"),
    ],
)
def test_item_rules(override: dict[str, object], code: str) -> None:
    with pytest.raises(ValidationError, match=code):
        item(**override)


@pytest.mark.parametrize(
    "override",
    [{"unanswerable_reason": None}, {"aliases": ["x"]}, {"allow_abstention": False}],
)
def test_unanswerable_rules(override: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="E05"):
        GroundTruthItem.model_validate({**UNANSWERABLE, **override})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        item(expected_anwser="70%")


def test_legacy_readme_record_upgrades() -> None:
    legacy = {"prompt": "What is the max LTV?", "expected_answer": "70%", "language": "fr", "domain": "finance"}
    items, issues = parse_lines([json.dumps(legacy)])
    assert items[0].id.startswith("legacy-")
    assert "legacy" in items[0].tags
    assert [i.code for i in issues] == ["W01"]


def test_load_reports_cross_item_rules(tmp_path: Path) -> None:
    rows = [
        BASE,
        {**BASE, "prompt": "duplicate id"},
        {**BASE, "id": "fin-en-0001", "fact_id": "fin-fr-0001", "language": "en", "domain": "legal"},
        {**BASE, "id": "fin-en-0002", "fact_id": "fin-fr-0001", "language": "en", "split": "dev"},
    ]
    p = tmp_path / "gt.jsonl"
    p.write_text("// comment\n\n" + "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(GroundTruthError) as exc:
        load_ground_truth(p)
    codes = {i.code for i in exc.value.issues}
    assert {"E02", "E12", "E13"} <= codes


def test_dataset_hash_is_order_independent(tmp_path: Path) -> None:
    rows = [BASE, UNANSWERABLE]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    b.write_text("\n".join(json.dumps(r) for r in reversed(rows)), encoding="utf-8")
    assert load_ground_truth(a).sha256 == load_ground_truth(b).sha256


def test_filter_by_split(tmp_path: Path) -> None:
    rows = [BASE, {**UNANSWERABLE, "split": "dev"}]
    p = tmp_path / "gt.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    gt = load_ground_truth(p)
    assert [i.id for i in gt.filter(splits=("test",)).items] == ["fin-fr-0001"]
    assert len(gt.filter(splits=("all",))) == 2
