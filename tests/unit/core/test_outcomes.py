"""Decision table: hand-written rows for every precedence edge, totality, and monotonicity properties."""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from slmforge.eval.outcomes import (
    ABSTAINED,
    ANSWERED,
    Evidence,
    ItemResult,
    Outcome,
    decide,
    reclassify,
)

ANS = {"answerable": True, "allow_abstention": True, "answer_type": "percent"}
UNANS = {"answerable": False, "allow_abstention": True, "answer_type": "percent"}


def ev(base: dict[str, object], **kw: object) -> Evidence:
    return Evidence(**{**base, **kw})  # type: ignore[arg-type]


ROWS = [
    # (evidence, expected outcome, note)
    (ev(ANS, invalid_reason="empty"), Outcome.INVALID, "R0 beats everything"),
    (ev(ANS, invalid_reason="degenerate", verdict="match", answered_signal=True), Outcome.INVALID, "R0"),
    (ev(ANS, verdict="match", answered_signal=True, confidence=0.9), Outcome.CORRECT, "R3 match"),
    (
        ev(ANS, verdict="match", answered_signal=True, epistemic_hedge=True),
        Outcome.CORRECT_HEDGED,
        "hedged match",
    ),
    (
        ev(ANS, verdict="mismatch", answered_signal=True, confidence=0.9),
        Outcome.CONFABULATION,
        "confident wrong",
    ),
    (
        ev(ANS, verdict="mismatch", answered_signal=True, confidence=0.2),
        Outcome.WRONG_LOW_CONF,
        "unsure wrong",
    ),
    (
        ev(ANS, verdict="mismatch", answered_signal=True, confidence=None),
        Outcome.CONFABULATION,
        "no logprobs",
    ),
    (
        ev(ANS, verdict="mismatch", answered_signal=True, epistemic_hedge=True, confidence=0.9),
        Outcome.WRONG_HEDGED,
        "",
    ),
    (
        ev(ANS, verdict="mismatch", answered_signal=True, forces_hedge=True, confidence=0.9),
        Outcome.WRONG_HEDGED,
        "several values in final answer",
    ),
    (
        ev(
            ANS,
            verdict="mismatch",
            answered_signal=True,
            abstain_cue=True,
            abstain_cue_strong=True,
            confidence=0.9,
        ),
        Outcome.WRONG_HEDGED,
        "'je ne suis pas sûr, mais 80 %'",
    ),
    (
        ev(ANS, verdict="mismatch", answered_signal=True, abstain_cue=True, confidence=0.9),
        Outcome.CONFABULATION,
        "weak courtesy cue never hedges",
    ),
    (ev(ANS, abstain_cue=True), Outcome.ABSTAIN_ACCEPTED, "R2 allow_abstention=true"),
    (
        ev({**ANS, "allow_abstention": False}, abstain_cue=True),
        Outcome.ABSTAIN_UNWARRANTED,
        "R2 required knowledge",
    ),
    (ev(ANS, verdict="unresolved", verdict_rule="no_candidate"), Outcome.NON_ANSWER, "typed, no value given"),
    (
        ev({**ANS, "answer_type": "entity"}, verdict="unresolved", verdict_rule="no_candidate"),
        Outcome.UNRESOLVED,
        "",
    ),
    (
        ev(ANS, verdict="unresolved", verdict_rule="skeleton_band", finish_reason="length"),
        Outcome.INVALID,
        "R5",
    ),
    (
        ev(ANS, verdict="match", answered_signal=True, finish_reason="length"),
        Outcome.CORRECT,
        "truncated but decided",
    ),
    (
        ev(UNANS, premise_rejection=True, answered_signal=True),
        Outcome.ABSTAIN_CORRECT,
        "R1 premise rejection",
    ),
    (ev(UNANS, abstain_cue=True), Outcome.ABSTAIN_CORRECT, "R2"),
    (ev(UNANS, answered_signal=True, answered_via="candidate", confidence=0.9), Outcome.CONFABULATION, "R4"),
    (
        ev(UNANS, answered_signal=True, answered_via="candidate", epistemic_hedge=True),
        Outcome.WRONG_HEDGED,
        "R4",
    ),
    (
        ev(UNANS, abstain_cue=True, abstain_cue_strong=True, answered_signal=True, confidence=0.9),
        Outcome.WRONG_HEDGED,
        "abstains then guesses anyway",
    ),
    (ev(UNANS), Outcome.NON_ANSWER, "R4n evasive"),
    (ev(UNANS, finish_reason="length"), Outcome.INVALID, "R5"),
]


@pytest.mark.parametrize(
    ("evidence", "expected", "note"), ROWS, ids=[f"{i}:{r[2]}" for i, r in enumerate(ROWS)]
)
def test_decision_rows(evidence: Evidence, expected: Outcome, note: str) -> None:
    assert decide(evidence, tau=0.5).outcome is expected


def test_residual_policy_unresolved() -> None:
    e = ev(UNANS, answered_signal=True, answered_via="residual", confidence=0.9)
    assert decide(e, tau=0.5, policy="residual").outcome is Outcome.CONFABULATION
    d = decide(e, tau=0.5, policy="unresolved")
    assert (d.outcome, d.best_case, d.worst_case) == (
        Outcome.UNRESOLVED,
        Outcome.ABSTAIN_CORRECT,
        Outcome.CONFABULATION,
    )


def _grid() -> list[Evidence]:
    out = []
    for (
        answerable,
        allow,
        atype,
        invalid,
        finish,
        cue,
        strong,
        prem,
        hedge,
        answered,
        verdict,
        forces,
        conf,
    ) in itertools.product(
        [True, False],
        [True, False],
        ["percent", "entity"],
        [None, "empty"],
        ["eos", "length"],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        [None, "match", "mismatch", "unresolved"],
        [False, True],
        [None, 0.2, 0.8],
    ):
        if not answerable and (not allow or verdict is not None or forces):
            continue  # invalid combinations: the schema / matcher never produce them
        if strong and not cue:
            continue
        out.append(
            Evidence(
                answerable,
                allow,
                atype,
                finish,
                invalid,
                cue,
                strong,
                prem,
                hedge,
                answered,
                "candidate" if answered else None,
                verdict,
                "no_candidate",
                forces,
                conf,
            )
        )
    return out


GRID = _grid()


def test_total_and_bounds_consistent() -> None:
    for e in GRID:
        d = decide(e, tau=0.5)
        assert isinstance(d.outcome, Outcome)
        if d.outcome is not Outcome.UNRESOLVED:
            assert d.best_case is d.outcome and d.worst_case is d.outcome
        if not e.answerable:
            assert d.outcome not in (
                Outcome.CORRECT,
                Outcome.CORRECT_HEDGED,
                Outcome.ABSTAIN_ACCEPTED,
                Outcome.ABSTAIN_UNWARRANTED,
            )
        if d.outcome in ABSTAINED:
            assert not e.answered_signal or e.premise_rejection


def test_hedging_never_creates_confabulation() -> None:
    for e in GRID:
        if decide(e, tau=0.5).outcome is not Outcome.CONFABULATION:
            assert (
                decide(dataclasses.replace(e, epistemic_hedge=True), tau=0.5).outcome
                is not Outcome.CONFABULATION
            )


def test_confabulation_monotone_in_confidence_and_tau() -> None:
    for e in GRID:
        if e.confidence is None:
            continue
        lo = decide(dataclasses.replace(e, confidence=0.2), tau=0.5).outcome is Outcome.CONFABULATION
        hi = decide(dataclasses.replace(e, confidence=0.8), tau=0.5).outcome is Outcome.CONFABULATION
        assert hi or not lo
        assert (
            decide(e, tau=0.0).outcome is Outcome.CONFABULATION
            or decide(e, tau=0.9).outcome is not Outcome.CONFABULATION
        )


def test_answered_outcomes_require_commitment() -> None:
    for e in GRID:
        if decide(e, tau=0.5).outcome in ANSWERED and e.invalid_reason is None:
            assert e.answered_signal or e.verdict in ("match", "mismatch")


def _result(e: Evidence) -> ItemResult:
    d = decide(e, tau=0.5)
    return ItemResult(
        "i1",
        "f1",
        "fr",
        "finance",
        e.answer_type,
        e.answerable,
        e.allow_abstention,
        None,
        False,
        e,
        d.outcome,
        d.best_case,
        d.worst_case,
        d.hedged,
        d.rule,
    )


def test_reclassify_matches_decide_and_roundtrips() -> None:
    e = ev(ANS, verdict="mismatch", answered_signal=True, confidence=0.6)
    r = _result(e)
    assert r.outcome is Outcome.CONFABULATION
    assert reclassify(r, tau=0.7).outcome is Outcome.WRONG_LOW_CONF
    assert reclassify(r, tau=0.5, verdict="match", verdict_source="human").outcome is Outcome.CORRECT
    assert ItemResult.from_json(r.to_json()) == r


def test_nan_confidence_is_unavailable() -> None:
    e = ev(ANS, verdict="mismatch", answered_signal=True, confidence=float("nan"))
    assert e.confidence is None
    assert decide(e, tau=0.0).outcome is Outcome.CONFABULATION
    with pytest.raises(ValueError, match="probability"):
        ev(ANS, confidence=1.5)


def test_reclassify_keeps_scoring_policy() -> None:
    e = ev(UNANS, answered_signal=True, answered_via="residual", confidence=0.9, no_cue_policy="unresolved")
    r = _result(e)
    assert r.outcome is Outcome.UNRESOLVED
    assert reclassify(r, tau=0.5).outcome is Outcome.UNRESOLVED
    assert reclassify(r, tau=0.5, policy="residual").outcome is Outcome.CONFABULATION


def test_override_verdict_needs_source() -> None:
    r = _result(ev(ANS, verdict="mismatch", answered_signal=True, confidence=0.6))
    with pytest.raises(ValueError, match="verdict_source"):
        reclassify(r, tau=0.5, verdict="match")


def test_human_asserted_verdict_on_unanswerable_item() -> None:
    e = ev(UNANS, answered_signal=True, answered_via="residual", confidence=0.9, no_cue_policy="unresolved")
    r = reclassify(_result(e), tau=0.5, verdict="mismatch", verdict_source="human")
    assert (r.outcome, r.verdict_source) == (Outcome.CONFABULATION, "human")
    # a human verdict beats a lexical premise-rejection cue
    e2 = ev(UNANS, premise_rejection=True, answered_signal=True, confidence=0.9)
    assert reclassify(_result(e2), tau=0.5, verdict="mismatch", verdict_source="human").outcome is (
        Outcome.CONFABULATION
    )
