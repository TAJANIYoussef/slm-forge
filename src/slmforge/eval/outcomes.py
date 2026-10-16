"""Per-item outcome taxonomy and the decision procedure.

Every scored item gets exactly one :class:`Outcome`. The decision is a pure function of
stored signals (:class:`Evidence`), so outcomes can be re-derived without regenerating
(tau sweeps, human verdicts, judge verdicts) via :func:`reclassify`.

"Confidently false" (CONFABULATION) = a wrong or asserted answer that is **not hedged**
(no verbal hedge) **and** whose answer-span confidence is >= tau. A wrong answer with
low confidence is WRONG_LOW_CONF; a hedged one is WRONG_HEDGED. Missing confidence
(external responses without logprobs) counts as confident; the metrics layer then flags
``CONF_UNAVAILABLE`` and the tau-free ``unhedged_error_rate`` is the comparable number.
Full rationale: ``docs/metrics.md``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal

from slmforge._stub import todo
from slmforge.types import TYPED_TYPES

if TYPE_CHECKING:
    from slmforge.eval.abstention import ResponseSignals
    from slmforge.eval.confidence import Confidence
    from slmforge.eval.dataset import GroundTruthItem
    from slmforge.eval.extract import CandidateSet
    from slmforge.eval.generate import GenerationRecord
    from slmforge.eval.match import MatchResult

OUTCOME_TABLE_VERSION: Final = "1.0.0"


class Outcome(StrEnum):
    """Exactly one per scored item."""

    CORRECT = "correct"  # answerable, MATCH, not hedged
    CORRECT_HEDGED = "correct_hedged"  # answerable, MATCH, hedged
    CONFABULATION = "confabulation"  # wrong/asserted, not hedged, conf >= tau (or unknown)
    WRONG_LOW_CONF = "wrong_low_conf"  # wrong/asserted, not hedged, conf < tau
    WRONG_HEDGED = "wrong_hedged"  # wrong/asserted, hedged
    NON_ANSWER = "non_answer"  # no abstention cue and no solicited value (evasive)
    ABSTAIN_CORRECT = "abstain_correct"  # unanswerable; abstained or rejected the premise
    ABSTAIN_ACCEPTED = "abstain_accepted"  # answerable, allow_abstention=true, abstained
    ABSTAIN_UNWARRANTED = "abstain_unwarranted"  # answerable, allow_abstention=false, abstained
    UNRESOLVED = "unresolved"  # rules cannot decide; bounds use best/worst case
    INVALID = "invalid"  # empty, degenerate, generation error, truncated without answer


CORRECTS: Final = frozenset({Outcome.CORRECT, Outcome.CORRECT_HEDGED})
WRONG: Final = frozenset({Outcome.CONFABULATION, Outcome.WRONG_LOW_CONF, Outcome.WRONG_HEDGED})
ANSWERED: Final = CORRECTS | WRONG
ABSTAINED: Final = frozenset({Outcome.ABSTAIN_CORRECT, Outcome.ABSTAIN_ACCEPTED, Outcome.ABSTAIN_UNWARRANTED})

Verdict = Literal["match", "mismatch", "unresolved"]
InvalidReason = Literal["empty", "degenerate", "generation_error", "truncated_no_answer"]
FinishReason = Literal["eos", "stop", "length", "error", "external"]
AnsweredVia = Literal["candidate", "gold", "known_false", "final_line", "residual"]


@dataclass(frozen=True)
class Evidence:
    """Everything the decision needs; produced by the pipeline and stored per item."""

    answerable: bool
    allow_abstention: bool
    answer_type: str
    finish_reason: FinishReason = "eos"
    invalid_reason: InvalidReason | None = None  # from detect_invalid (truncation is decided here)
    abstain_cue: bool = False  # any strong or weak abstention cue
    abstain_cue_strong: bool = False
    premise_rejection: bool = False
    epistemic_hedge: bool = False
    answered_signal: bool = False  # the response commits to something
    answered_via: AnsweredVia | None = None
    verdict: Verdict | None = None  # answerable items only
    verdict_rule: str | None = None
    forces_hedge: bool = False  # several distinct values in a declared final answer
    confidence: float | None = None


@dataclass(frozen=True)
class Decision:
    """Outcome plus the best/worst case used for UNRESOLVED bounds."""

    outcome: Outcome
    best_case: Outcome
    worst_case: Outcome
    hedged: bool
    rule: str  # which row of the decision table fired (R0..R5)


def is_hedged(ev: Evidence) -> bool:
    """Verbal hedge, alternatives in the final answer, or a strong 'I don't know' followed by an answer.

    Weak cues (courtesy formulas such as "consult a lawyer", "الله أعلم") never hedge.
    """
    return ev.epistemic_hedge or ev.forces_hedge or (ev.abstain_cue_strong and ev.answered_signal)


def wrong_outcome(hedged: bool, confidence: float | None, tau: float) -> Outcome:
    """Split a wrong/asserted answer into CONFABULATION, WRONG_LOW_CONF or WRONG_HEDGED."""
    if hedged:
        return Outcome.WRONG_HEDGED
    if confidence is None or confidence >= tau:
        return Outcome.CONFABULATION
    return Outcome.WRONG_LOW_CONF


def decide(ev: Evidence, *, tau: float, policy: Literal["residual", "unresolved"] = "residual") -> Decision:
    """The ordered decision table (first rule that fires wins). Pure and total."""
    hedged = is_hedged(ev)
    wrong = wrong_outcome(hedged, ev.confidence, tau)
    best_answer = Outcome.CORRECT_HEDGED if hedged else Outcome.CORRECT

    def done(o: Outcome, rule: str) -> Decision:
        return Decision(o, o, o, hedged, rule)

    # R0: unusable generations
    if ev.invalid_reason in ("empty", "degenerate", "generation_error"):
        return done(Outcome.INVALID, "R0")

    if not ev.answerable:
        # R1/R2: premise rejection or abstention without a committed answer
        if ev.premise_rejection:
            return done(Outcome.ABSTAIN_CORRECT, "R1")
        if ev.abstain_cue and not ev.answered_signal:
            return done(Outcome.ABSTAIN_CORRECT, "R2")
        # R4: any asserted answer to an unanswerable question is wrong
        if ev.answered_signal:
            if ev.answered_via == "residual" and policy == "unresolved":
                pending = Decision(Outcome.UNRESOLVED, Outcome.ABSTAIN_CORRECT, wrong, hedged, "R4u")
                return _truncation(ev, pending)
            return done(wrong, "R4")
        return _truncation(ev, done(Outcome.NON_ANSWER, "R4n"))

    # answerable
    if ev.abstain_cue and not ev.answered_signal:  # R2
        return done(Outcome.ABSTAIN_ACCEPTED if ev.allow_abstention else Outcome.ABSTAIN_UNWARRANTED, "R2")
    if ev.verdict == "match":  # R3
        return done(best_answer, "R3")
    if ev.verdict == "mismatch":
        return done(wrong, "R3")
    if ev.verdict_rule == "no_candidate" and ev.answer_type in TYPED_TYPES:
        return _truncation(ev, done(Outcome.NON_ANSWER, "R3n"))
    return _truncation(ev, Decision(Outcome.UNRESOLVED, best_answer, wrong, hedged, "R3u"))


def _truncation(ev: Evidence, d: Decision) -> Decision:
    """R5: a generation cut by max_new_tokens that produced no decidable answer is INVALID."""
    if ev.finish_reason == "length":
        return Decision(Outcome.INVALID, Outcome.INVALID, Outcome.INVALID, d.hedged, "R5")
    return d


@dataclass(frozen=True)
class ItemResult:
    """Per-item scored row (``scored.jsonl``); enough to re-derive the outcome."""

    item_id: str
    fact_id: str
    lang_script: str
    domain: str
    answer_type: str
    answerable: bool
    allow_abstention: bool
    unanswerable_reason: str | None
    code_switched: bool
    evidence: Evidence
    outcome: Outcome
    best_case: Outcome
    worst_case: Outcome
    hedged: bool
    decision_rule: str
    verdict_source: Literal["rules", "human", "judge", "none"] = "rules"
    confidence_source: str = "none"
    region_method: Literal["final_line", "full_text"] = "full_text"
    response_lang: str | None = None
    response_script: str | None = None
    language_fidelity: bool | None = None
    script_fidelity: bool | None = None
    response_sha256: str = ""
    flags: frozenset[str] = field(default_factory=frozenset)

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = dataclasses.asdict(self)
        d["flags"] = sorted(self.flags)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> ItemResult:
        """Deserialize from :meth:`to_json` output."""
        d = dict(d)
        d["evidence"] = Evidence(**d["evidence"])
        for k in ("outcome", "best_case", "worst_case"):
            d[k] = Outcome(d[k])
        d["flags"] = frozenset(d.get("flags", ()))
        return cls(**d)


def detect_invalid(
    rec: GenerationRecord, response: str, *, min_tokens: int = 64, min_distinct_4gram: float = 0.25,
    max_token_repeat: int = 8,
) -> InvalidReason | None:
    """Empty, degenerate (low distinct-4-gram ratio or long token repeats) or errored generations."""
    todo("M1", "detect_invalid")


def classify(
    item: GroundTruthItem,
    rec: GenerationRecord,
    *,
    cands: CandidateSet,
    match: MatchResult | None,
    signals: ResponseSignals,
    confidence: Confidence,
    tau: float,
    policy: Literal["residual", "unresolved"] = "residual",
) -> ItemResult:
    """Assemble :class:`Evidence` from the pipeline stages and apply :func:`decide`."""
    todo("M1", "classify")


def reclassify(
    r: ItemResult,
    *,
    tau: float,
    verdict: Verdict | None = None,
    verdict_source: Literal["rules", "human", "judge", "none"] | None = None,
    policy: Literal["residual", "unresolved"] = "residual",
) -> ItemResult:
    """Re-derive an outcome from stored evidence with a new tau and/or an override verdict."""
    ev = r.evidence if verdict is None else dataclasses.replace(r.evidence, verdict=verdict)
    d = decide(ev, tau=tau, policy=policy)
    return dataclasses.replace(
        r, evidence=ev, outcome=d.outcome, best_case=d.best_case, worst_case=d.worst_case,
        hedged=d.hedged, decision_rule=d.rule, verdict_source=verdict_source or r.verdict_source,
    )
