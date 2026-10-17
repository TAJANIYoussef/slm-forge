"""Answer matching: one :data:`~slmforge.eval.outcomes.Verdict` per answerable item (plan §2.5).

Stages run in order on the answer region and the first decisive one wins:

* **S1 exact**: L1(region) equals L1 of the expected answer or an alias.
* **S2 typed** (numeric, percent, currency, duration, date, citation, boolean, choice):
  compare compatible, non-negated candidates with the gold value. Only candidates with the
  gold's dimension count; several distinct values give MISMATCH ``multi`` with
  ``forces_hedge`` in a declared final line, UNRESOLVED ``multi_full_text`` otherwise.
* **S3 alias containment**: whole-token gold/known_false spans.
* **S4 skeleton** (Darija entity/free_text): skeleton ratio >= 0.85 is a MATCH, the
  [0.70, 0.85) band is UNRESOLVED and goes to the review queue.
* **S5 no evidence**: short entity answers are MISMATCH ``entity_short``; otherwise
  UNRESOLVED ``no_candidate`` (which ``decide`` turns into NON_ANSWER for typed items).
* **False denial**: premise rejection on an answerable item with no gold found is MISMATCH.

Tolerance: ``|a - g| <= max(abs_tol, rel_tol * |g|)`` in ``Decimal``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from slmforge._stub import todo
from slmforge.eval.outcomes import Verdict

if TYPE_CHECKING:
    from slmforge.config.schema import MatchSection
    from slmforge.eval.abstention import ResponseSignals
    from slmforge.eval.dataset import GroundTruthItem
    from slmforge.eval.extract import Candidate, CandidateSet

MATCHER_VERSION: Final = "1.0.0"


@dataclass(frozen=True)
class MatchConfig:
    """Matcher switches and thresholds (``scoring.matching`` in the config, plus fixed constants)."""

    accept_locale_ambiguity: bool = True  # "1,500" in fr/en: accept either reading, flag it
    darija_money_scale: bool = True  # colloquial "melyoun" = 10,000 DH reading
    colloquial_units: bool = True  # ryal = 0.05 DH, centime = 0.01 DH
    containment_min_chars: int = 3  # alias targets shorter than this must equal the whole region
    contrast_window_tokens: int = 3  # "pas 80%": a value this close after a contrast cue is rejected
    entity_short_max_tokens: int = 6
    skeleton_enabled: bool = True
    skeleton_threshold: float = 0.85
    skeleton_review_band: float = 0.70
    skeleton_min_len: int = 4

    @classmethod
    def from_section(cls, section: MatchSection) -> MatchConfig:
        """Build from the config section; fields the section lacks keep their defaults."""
        return cls(**section.model_dump())


@dataclass(frozen=True)
class MatchResult:
    """Verdict plus the rule that produced it."""

    verdict: Verdict
    rule: str  # exact, typed, typed_mismatch, known_false, multi, multi_full_text, alias, skeleton, ...
    primary: Candidate | None = None
    matched_target: str | None = None
    gold_found: bool = False
    known_false_found: bool = False
    forces_hedge: bool = False  # several distinct values in a declared final answer
    flags: frozenset[str] = field(default_factory=frozenset)  # locale_ambiguous, darija_money_scale, ...


def match(
    item: GroundTruthItem, cands: CandidateSet, *, signals: ResponseSignals, cfg: MatchConfig
) -> MatchResult:
    """Apply stages S1-S5 and the false-denial rule; answerable items only."""
    todo("M1", "match")
