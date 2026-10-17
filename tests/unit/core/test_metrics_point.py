"""Metric table on a hand-built toy set: formulas, denominators, withholding, bounds, tau sweep, macro."""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from slmforge.eval.factual_reliability import (
    CONF_UNAVAILABLE,
    METRICS,
    compute_bounds,
    compute_point_metrics,
    group_by,
    macro,
    metrics_table,
    outcome_counts,
    tau_sweep,
    utility,
)
from slmforge.eval.outcomes import Evidence, ItemResult, decide


def make(
    i: int,
    *,
    answerable: bool = True,
    allow: bool = True,
    bucket: str = "fr",
    atype: str = "percent",
    **ev: object,
) -> ItemResult:
    e = Evidence(answerable=answerable, allow_abstention=allow, answer_type=atype, **ev)  # type: ignore[arg-type]
    d = decide(e, tau=0.5)
    return ItemResult(
        f"i{i}",
        f"f{i}",
        bucket,
        "finance",
        atype,
        answerable,
        allow,
        None if answerable else "false_premise",
        False,
        e,
        d.outcome,
        d.best_case,
        d.worst_case,
        d.hedged,
        d.rule,
        region_method="final_line" if i % 2 else "full_text",
    )


MATCH = {"verdict": "match", "answered_signal": True, "confidence": 0.9}
WRONG_HI = {"verdict": "mismatch", "answered_signal": True, "confidence": 0.9}
WRONG_LO = {"verdict": "mismatch", "answered_signal": True, "confidence": 0.3}

# 20 items: 14 answerable (2 with allow_abstention=false), 6 unanswerable.
TOY = [
    make(0, **MATCH),
    make(1, **MATCH),
    make(2, **MATCH),
    make(3, **MATCH),  # 4 CORRECT
    make(4, **MATCH, epistemic_hedge=True),  # CORRECT_HEDGED
    make(5, **WRONG_HI),
    make(6, **WRONG_HI),  # 2 CONFABULATION
    make(7, **WRONG_LO),  # WRONG_LOW_CONF
    make(8, **WRONG_HI, epistemic_hedge=True),  # WRONG_HEDGED
    make(9, abstain_cue=True),  # ABSTAIN_ACCEPTED
    make(10, allow=False, abstain_cue=True),  # ABSTAIN_UNWARRANTED
    make(11, allow=False, verdict="unresolved", verdict_rule="no_candidate"),  # NON_ANSWER
    make(12, invalid_reason="empty"),  # INVALID
    make(
        13,
        atype="entity",
        verdict="unresolved",
        verdict_rule="skeleton_band",
        answered_signal=True,
        confidence=0.9,
    ),  # UNRESOLVED
    make(14, answerable=False, abstain_cue=True),  # ABSTAIN_CORRECT
    make(15, answerable=False, premise_rejection=True),  # ABSTAIN_CORRECT
    make(16, answerable=False, abstain_cue=True),  # ABSTAIN_CORRECT
    make(
        17, answerable=False, answered_signal=True, answered_via="candidate", confidence=0.8
    ),  # CONFABULATION
    make(
        18, answerable=False, answered_signal=True, answered_via="candidate", epistemic_hedge=True
    ),  # W_HEDGED
    make(19, answerable=False),  # NON_ANSWER
]


def test_toy_outcomes() -> None:
    c = outcome_counts(TOY)
    assert c == {
        "correct": 4,
        "correct_hedged": 1,
        "confabulation": 3,
        "wrong_low_conf": 1,
        "wrong_hedged": 2,
        "non_answer": 2,
        "abstain_correct": 3,
        "abstain_accepted": 1,
        "abstain_unwarranted": 1,
        "unresolved": 1,
        "invalid": 1,
    }
    assert sum(c.values()) == len(TOY)


EXPECTED = {
    "factual_accuracy": (5, 14),
    "confabulation_rate": (3, 20),
    "abstention_rate": (3, 6),
    "selective_accuracy": (5, 9),
    "coverage": (10, 14),  # the UNRESOLVED item is answered in its best and worst case
    "hedge_rate": (3, 12),  # CORRECT_HEDGED + 2 WRONG_HEDGED among 11 answered + the UNRESOLVED one
    "unhedged_error_rate": (4, 20),
    "error_rate": (6, 20),
    "confabulation_rate_answerable": (2, 14),
    "unanswerable_confabulation_rate": (1, 6),
    "confab_given_answered": (3, 11),
    "confident_error_share": (3, 6),
    "p_unhedged_given_error": (4, 6),
    "p_confident_given_unhedged_error": (3, 4),
    "raw_abstention_rate": (5, 20),
    "abstention_precision": (3, 5),
    "over_abstention_rate": (2, 14),
    "unwarranted_abstention_rate": (1, 2),
    "non_answer_rate": (2, 20),
    "invalid_rate": (1, 20),
    "unresolved_rate": (1, 20),
    "truncation_rate": (0, 20),
    "format_compliance_rate": (10, 20),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_formula(name: str) -> None:
    r = compute_point_metrics(TOY)[name]
    assert (r.num, r.den) == EXPECTED[name]


def test_every_metric_is_tested_or_fidelity() -> None:
    assert set(METRICS) - set(EXPECTED) == {"language_fidelity_rate", "script_fidelity_rate"}


def test_zero_denominator_is_none() -> None:
    only_answerable = [r for r in TOY if r.answerable]
    m = compute_point_metrics(only_answerable)
    assert m["abstention_rate"].den == 0 and m["abstention_rate"].value is None
    assert compute_point_metrics([])["factual_accuracy"].value is None


def test_confabulation_decomposition() -> None:
    m = compute_point_metrics(TOY)
    lhs = m["confabulation_rate"].value
    rhs = (
        m["error_rate"].value
        * m["p_unhedged_given_error"].value
        * m["p_confident_given_unhedged_error"].value
    )
    assert lhs == pytest.approx(rhs)


def test_missing_confidence_withholds_gated_metrics() -> None:
    rs = [*TOY, make(99, verdict="mismatch", answered_signal=True, confidence=None)]
    m = compute_point_metrics(rs)
    assert m["confabulation_rate"].value is None and m["confabulation_rate"].withheld == CONF_UNAVAILABLE
    assert m["unhedged_error_rate"].value == pytest.approx(5 / 21)  # tau-free stays available
    assert m["factual_accuracy"].value is not None


def test_utility() -> None:
    # resolved items: 19; weights from ScoringConfig defaults
    expected = (4 + 1 - 3 - 0.75 - 1.0 + 0 - 0.5 - 0.25 - 0.5 + 3 + 0) / 19
    assert utility(TOY) == pytest.approx(expected)


def test_bounds_contain_point_and_use_best_worst() -> None:
    point_metrics = compute_point_metrics(TOY)
    for name in (*METRICS, "reliability_utility"):
        point = point_metrics[name].value
        lo, hi = compute_bounds(TOY, name)
        if point is None:
            continue
        assert lo is not None and hi is not None, name
        assert lo <= point <= hi, (name, lo, point, hi)
    # the UNRESOLVED answerable item is CORRECT at best and CONFABULATION at worst
    assert compute_bounds(TOY, "factual_accuracy") == (5 / 14, 6 / 14)
    assert compute_bounds(TOY, "confabulation_rate") == (3 / 20, 4 / 20)


def _unresolved(i: int, **ev: object) -> ItemResult:
    return make(i, atype="entity", verdict="unresolved", verdict_rule="no_candidate", **ev)


def test_bounds_regressions_from_review() -> None:
    confab = [make(i, atype="entity", **WRONG_HI) for i in range(3)]
    rs = [*confab, _unresolved(3, epistemic_hedge=True, answered_signal=True)]
    # coverage: both cases of the UNRESOLVED item are answered, so the point counts it as answered
    assert compute_point_metrics(rs)["coverage"].value == 1.0
    assert compute_bounds(rs, "coverage") == (1.0, 1.0)
    hedge = compute_point_metrics(rs)["hedge_rate"].value
    lo, hi = compute_bounds(rs, "hedge_rate")
    assert lo <= hedge <= hi
    u_lo, u_hi = compute_bounds(rs, "reliability_utility")
    assert u_lo <= utility(rs) <= u_hi
    # only an UNRESOLVED item can enter the denominator: a defined bound, not None
    lone = [_unresolved(0, answered_signal=True, confidence=0.9)]
    lo, hi = compute_bounds(lone, "confab_given_answered")
    assert (lo, hi) == (0.0, 1.0)


def test_tau_sweep_monotone_and_tau0_identity() -> None:
    sweep = tau_sweep(TOY, [0.0, 0.3, 0.5, 0.7, 0.95])
    vals = list(sweep.values())
    assert all(a >= b for a, b in itertools.pairwise(vals))
    assert sweep["0"] == compute_point_metrics(TOY)["unhedged_error_rate"].value
    assert list(sweep) == ["0", "0.3", "0.5", "0.7", "0.95"]


def test_macro_excludes_small_buckets() -> None:
    big = [
        dataclasses.replace(r, item_id=f"{b}{r.item_id}", lang_script=b)
        for b in ("fr", "en")
        for r in TOY * 2
    ]
    value, excluded = macro(big, "factual_accuracy", min_n=20)
    assert value == pytest.approx(5 / 14)
    assert excluded == ["darija_arab", "darija_latn"]
    assert macro(TOY, "factual_accuracy", min_n=30) == (None, ["darija_arab", "darija_latn", "fr", "en"])


def test_group_by_labels() -> None:
    groups = group_by(TOY, "answerable")
    assert set(groups) == {"true", "false"} and len(groups["false"]) == 6
    with pytest.raises(ValueError):
        group_by(TOY, "evidence")


def test_metrics_table_is_declarative() -> None:
    t = metrics_table()
    assert t["factual_accuracy"]["denominator_filter"] == "answerable"
    assert t["confabulation_rate"]["conf_gated"] is True
    assert t["factual_accuracy"]["tier"] == "primary"
