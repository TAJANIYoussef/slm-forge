"""Answer-span confidence from stored teacher-forced token logprobs (plan §2.6, critique 5).

Token logprobs come from a teacher-forced forward pass over prompt + response (see
``eval.generate``), stored per token together with each token's character span in the raw
response. Rescoring can therefore recompute the span confidence when the matcher moves the
answer span, without regenerating.

Answer span S, in order of preference:

1. tokens overlapping the matched span, the primary candidate or the alias span (``answer_span``);
2. the first content-token run of the answer region (``region``);
3. the first K = 32 non-special tokens (``first_k``).

``conf = exp(mean_{t in S} logprob_t)`` and ``min_token_prob = exp(min_{t in S} logprob_t)``.
Without logprobs (external responses) ``conf`` is ``None`` and ``source`` is ``"none"``;
the decision then treats the answer as confident and the metrics layer withholds the
confidence-gated rates (``CONF_UNAVAILABLE``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.eval.abstention import ResponseSignals, SignalDetector
    from slmforge.eval.dataset import GroundTruthItem
    from slmforge.eval.extract import CandidateSet
    from slmforge.eval.generate import GenerationRecord
    from slmforge.eval.match import MatchResult
    from slmforge.text.lexicons import LexiconSet
    from slmforge.types import Span

ConfidenceSource = Literal["answer_span", "region", "first_k", "none"]


@dataclass(frozen=True)
class Confidence:
    """Answer-span confidence."""

    value: float | None
    min_token_prob: float | None
    source: ConfidenceSource
    token_idx: tuple[int, ...] = ()  # indices into GenerationRecord.token_logprobs


def tokens_overlapping(rec: GenerationRecord, span: Span) -> list[int]:
    """Generated-token indices whose character span overlaps the raw-response ``span``."""
    todo("M1", "tokens_overlapping")


def compute_confidence(
    rec: GenerationRecord, cands: CandidateSet, m: MatchResult | None, *, first_k: int = 32
) -> Confidence:
    """``exp(mean logprob)`` over the answer span, with the region / first-k fallbacks."""
    todo("M1", "compute_confidence")


def canonical_answer(item: GroundTruthItem, cands: CandidateSet, signals: ResponseSignals) -> str:
    """Canonical form for consistency: ``ABSTAIN``, ``num:<v>|<unit>``, ``alias:<i>``, ``text:<64>``."""
    todo("M1", "canonical_answer")


def consistency_confidence(
    item: GroundTruthItem,
    greedy: str,
    samples: Sequence[str],
    *,
    detector: SignalDetector,
    lexicons: LexiconSet,
) -> float:
    """Share of sampled responses whose canonical answer equals the greedy one (optional, off by default)."""
    todo("M9", "consistency_confidence")
