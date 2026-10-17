"""Overlap between training data and the eval set (our data only, not pretraining).

Two uses:

* **Training-time guard** (:func:`guard_training_records`, ``data.contamination_guard``):
  fail (default), drop or warn on training records that reproduce an eval prompt. The M3
  version checks exact matches of L1-normalized prompts plus 8-gram overlap; M8 adds
  Darija skeleton shingles so a cross-script paraphrase is caught too.
* **Eval-time report** (:func:`contamination_report`, M8): per item, n-gram overlap,
  Jaccard near-duplicates and ``fact_seen`` (an anchor term and the gold value co-occur
  in some training text). Flagged items are excluded from headline numbers and reported
  as a separate stratum.

Shingles are word n-grams over L1 tokens (``text.normalize``) for fr/en and over
consonant skeletons (``text.arabizi.skeleton``) for Darija, so Arabic-script and Arabizi
copies of one prompt collide.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.data.loader import ChatRecord
    from slmforge.eval.dataset import GroundTruthItem, GroundTruthSet

Shingle = tuple[str, ...]


@dataclass(frozen=True)
class ItemContamination:
    """Overlap between one eval item and training data."""

    item_id: str
    prompt_ngram_overlap: float  # share of the prompt's n-grams present in training text
    near_dup_train_ids: tuple[str, ...]  # training records with Jaccard >= threshold
    fact_seen: bool | None  # None when the item has no anchor_terms
    contaminated: bool


class ContaminationError(Exception):
    """Raised by the guard with ``policy="error"``; lists the overlapping eval and training ids."""

    def __init__(self, pairs: Sequence[tuple[str, str]]) -> None:
        self.pairs = list(pairs)  # (eval item id, training record id)
        shown = ", ".join(f"{i}<->{r}" for i, r in self.pairs[:10])
        super().__init__(
            f"{len(self.pairs)} training record(s) overlap eval prompts: {shown}; "
            "remove them or set data.contamination_guard.policy to drop/warn"
        )


def shingles(text: str, *, n: int = 8, skeleton: bool = False) -> set[Shingle]:
    """Word n-grams of L1 tokens (or of skeletons with ``skeleton=True``); short texts give one gram."""
    if skeleton:
        todo("M8", "shingles (skeleton)")
    todo("M3", "shingles")


class TrainIndex:
    """8-gram shingles (skeleton tokens for Darija, L1 otherwise); MinHash LSH with the [contam] extra."""

    def __init__(self, *, n: int = 8) -> None:
        self.n = n
        self._postings: dict[Shingle, set[str]] = {}
        self._texts: dict[str, str] = {}

    @classmethod
    def build(cls, records: Iterable[ChatRecord], *, n: int = 8) -> TrainIndex:
        """Index every turn of every record (user and assistant text)."""
        todo("M8", "TrainIndex.build")

    def __len__(self) -> int:
        return len(self._texts)

    def prompt_overlap(self, text: str) -> float:
        """Share of the text's n-grams present in training."""
        todo("M8", "TrainIndex.prompt_overlap")

    def near_duplicates(self, text: str, *, threshold: float = 0.5) -> list[tuple[str, float]]:
        """Jaccard near-dups as ``(record id, similarity)``, most similar first."""
        todo("M8", "TrainIndex.near_duplicates")

    def fact_seen(self, item: GroundTruthItem, *, window: int = 300) -> bool | None:
        """An anchor term and the gold value co-occur in some training text; None without anchor_terms."""
        todo("M8", "TrainIndex.fact_seen")


def contamination_report(
    gt: GroundTruthSet, index: TrainIndex, *, ngram_threshold: float = 0.5, jaccard_threshold: float = 0.5
) -> dict[str, ItemContamination]:
    """Per-item report keyed by item id."""
    todo("M8", "contamination_report")


def guard_training_records(
    records: Sequence[ChatRecord],
    gt: GroundTruthSet,
    *,
    policy: Literal["error", "drop", "warn"],
    ngram_n: int = 8,
    ngram_threshold: float = 0.5,
) -> list[ChatRecord]:
    """Fail, drop or warn on train records overlapping eval prompts (exact L1 match or n-gram share)."""
    todo("M3", "guard_training_records")
