"""Optional ``task`` suite: fluency/overlap metrics against references, and eval loss.

These feed the TRADEOFF flag ("trading reliability for fluency"): a checkpoint whose
eval loss fell (or whose chrF++ rose) while a primary reliability metric worsened.

* chrF++ (sacrebleu, the ``hf`` extra, imported lazily) is the primary fluency metric for
  Darija, where word-level overlap is unreliable.
* ROUGE-L uses a Unicode whitespace tokenizer, because ``rouge_score`` drops Arabic.
* EM and token F1 compare L1-normalized text; token F1 can compare Darija skeletons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.eval.generate import GenerationRecord


def exact_match(pred: str, ref: str) -> float:
    """Exact match of the L1-normalized strings (1.0 or 0.0)."""
    todo("M8", "exact_match")


def token_f1(pred: str, ref: str, *, skeleton_for_darija: bool = True) -> float:
    """Bag-of-tokens F1 on L1 tokens (skeletons for Darija when enabled)."""
    todo("M8", "token_f1")


def chrf_pp(preds: Sequence[str], refs: Sequence[str]) -> float:
    """Corpus chrF++ via sacrebleu (lazy import)."""
    todo("M8", "chrf_pp")


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-measure with a Unicode whitespace tokenizer."""
    todo("M8", "rouge_l")


def eval_loss_by_step(run_dir: Path) -> dict[int, float]:
    """``eval_loss`` per global step from ``train_log.jsonl`` or ``trainer_state.json``."""
    todo("M3", "eval_loss_by_step")


class TaskMetricsSuite:
    """Optional ``task`` suite on a ``{prompt, reference, language, script}`` JSONL."""

    name: ClassVar[str] = "task"
    METRICS: ClassVar[tuple[str, ...]] = ("chrf_pp", "rouge_l", "token_f1", "exact_match")

    def score(
        self, records: Mapping[str, GenerationRecord], refs: Mapping[str, str], metrics: Sequence[str]
    ) -> dict[str, float | None]:
        """Score generations against references by item id; ``None`` for a metric with no pairs."""
        todo("M8", "TaskMetricsSuite.score")
