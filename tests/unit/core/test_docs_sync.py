"""The reference docs stay in sync with the code contracts they describe."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import get_args

import pytest

from slmforge.eval.dataset import parse_lines, validate_set
from slmforge.eval.factual_reliability import DEFAULT_WEIGHTS, METRICS, UTILITY_METRIC
from slmforge.eval.outcomes import Outcome
from slmforge.eval.runner import FlagCode
from slmforge.types import UnanswerableReason

DOCS = Path(__file__).resolve().parents[3] / "docs"
DECISION_RULES = ("R0", "R1", "R2", "R3", "R3n", "R3u", "R4", "R4n", "R4u", "R5")


def _doc(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _json_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.findall(r"```json\n(.*?)\n```", text, re.S)]


@pytest.mark.parametrize(
    "name", ["metrics.md", "dataset.md", "architecture.md", "distillation-terms.md", "annotation.md"]
)
def test_doc_exists(name: str) -> None:
    assert _doc(name).startswith("# ")


def test_dataset_examples_validate_and_cover_every_bucket() -> None:
    lines = _json_blocks(_doc("dataset.md"))
    items, issues = parse_lines(lines)
    issues += validate_set(items)
    assert issues == []
    assert len(items) == len(lines) >= 5
    assert {"en", "fr", "darija_arab", "darija_latn"} <= {i.lang_script for i in items}
    assert any(not i.answerable for i in items)
    assert all(json.loads(line)["schema_version"] == 1 for line in lines)


def test_dataset_doc_lists_every_rule_code_and_reason() -> None:
    text = _doc("dataset.md")
    for code in [f"E{n:02d}" for n in range(1, 15)] + [f"W{n:02d}" for n in range(1, 14)]:
        assert f"| {code} |" in text, code
    for reason in get_args(UnanswerableReason):
        assert f"`{reason}`" in text, reason


def test_metrics_doc_covers_every_metric_outcome_rule_and_flag() -> None:
    text = _doc("metrics.md")
    for name in [*METRICS, UTILITY_METRIC]:
        assert f"`{name}`" in text, name
    for o in Outcome:
        assert f"`{o.value}`" in text, o
    for rule in DECISION_RULES:
        assert f"| {rule} |" in text, rule
    for code in get_args(FlagCode):
        assert f"`{code}" in text, code
    for key in DEFAULT_WEIGHTS:
        assert f"`{key}`" in text, key


def test_annotation_doc_covers_every_unanswerable_reason() -> None:
    text = _doc("annotation.md")
    for reason in get_args(UnanswerableReason):
        assert f"`{reason}`" in text, reason


def test_architecture_doc_lists_every_exit_code() -> None:
    text = _doc("architecture.md")
    for code in range(6):
        assert f"| {code} |" in text, code
