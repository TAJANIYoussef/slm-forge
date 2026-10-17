"""Response signals: abstention cues, premise rejection and hedges (plan §2.3).

Every lexicon bucket (en, fr, msa, darija_arab, darija_latn) is applied to every response,
whatever the item's language, so code-switched abstentions ("désolé, ma3reftch") and MSA
replies to Darija prompts are caught. Surface patterns run on L1 text; Darija cues are
written once as skeleton regexes that cover both scripts.

Definitions used by :func:`slmforge.eval.outcomes.decide`:

* ``abstain_cue_strong``: at least one strong abstain hit; ``abstain_cue``: strong or weak.
* ``premise_rejection``: a premise hit (after ``unless_followed_by``) or an item-level
  ``premise_rejection_cues`` hit; ignored for answerable ``boolean`` items ("la, makaynch"
  is a "no" there).
* ``epistemic_hedge``: at least one epistemic hedge hit. Approximation words ("environ",
  "t9riban") are recorded but never hedge; neither do weak cues or domain disclaimers.
* A not-know cue followed by a complementizer ("I don't know why people say 60%") is
  suppressed, so it cannot turn a confabulation into WRONG_HEDGED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.eval.dataset import GroundTruthItem
    from slmforge.text.lexicons import LexiconMatch, LexiconSet


@dataclass(frozen=True)
class SignalConfig:
    """Thresholds for signal detection."""

    min_content_tokens: int = 4  # residual content tokens that count as a committed prose answer


@dataclass(frozen=True)
class ResponseSignals:
    """Abstention, premise and hedge signals of one response (raw-response spans in ``hits``)."""

    abstain_cue: bool = False
    abstain_cue_strong: bool = False
    premise_rejection: bool = False
    epistemic_hedge: bool = False
    approximation: bool = False
    refusal_kind: Literal["epistemic", "policy"] | None = None  # policy refusals are reported separately
    categories: frozenset[str] = field(default_factory=frozenset)
    hits: tuple[LexiconMatch, ...] = ()


class SignalDetector:
    """Applies the union of all lexicon buckets on L1 and skeleton views."""

    def __init__(self, lexicons: LexiconSet, cfg: SignalConfig | None = None) -> None:
        self.lexicons = lexicons
        self.cfg = cfg or SignalConfig()

    def detect(self, item: GroundTruthItem, response: str) -> ResponseSignals:
        """Detect cues; the premise lexicon is ignored for answerable boolean items."""
        todo("M1", "SignalDetector.detect")
