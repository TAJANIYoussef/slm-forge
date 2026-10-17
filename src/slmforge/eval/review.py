"""Human review loop for items the rules cannot decide (M7).

``slm-forge review export`` writes UNRESOLVED (and borderline, e.g. skeleton-band) items
from one or more eval dirs to a JSONL queue, deduplicated by (item, L1-normalized
response). Annotators label them (``docs/annotation.md``); ``review import`` stores the
labels in a SQLite :class:`VerdictCache` shared across checkpoints and runs, and the next
``rescore`` applies them (``verdict_source="human"``), narrowing the UNRESOLVED bounds.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from slmforge._stub import todo
from slmforge.eval.outcomes import Outcome

ReviewLabel = Literal[
    "match", "mismatch", "abstain", "premise_rejection", "asserted", "non_answer", "unclear"
]


@dataclass(frozen=True)
class CachedVerdict:
    """A human or judge label for (item, normalized response)."""

    label: ReviewLabel
    source: Literal["human", "judge"]
    source_id: str  # annotator id or judge id
    created_at: str


class VerdictCache:
    """SQLite store of verdicts keyed by :meth:`key`, shared across checkpoints and runs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @staticmethod
    def key(item_id: str, response: str) -> str:
        """``sha256(item_id + '\\0' + L1(response))``, so whitespace/diacritic variants share a verdict."""
        todo("M7", "VerdictCache.key")

    def get(self, key: str) -> CachedVerdict | None:
        """Look up one verdict."""
        todo("M7", "VerdictCache.get")

    def put(self, key: str, v: CachedVerdict) -> None:
        """Insert or replace one verdict."""
        todo("M7", "VerdictCache.put")


def export_review_queue(
    eval_dirs: Sequence[Path],
    out: Path,
    *,
    outcomes: Collection[Outcome] = (Outcome.UNRESOLVED,),
    include_rules: Collection[str] = ("skeleton_band",),
    sample_per_stratum: int | None = None,
    seed: int = 0,
) -> int:
    """Write deduplicated items to label (with prompt, gold, response, rule trace); returns the count."""
    todo("M7", "export_review_queue")


def import_reviews(path: Path, cache: VerdictCache, *, annotator: str) -> int:
    """Load labelled review JSONL into the cache (``source="human"``); returns the number imported."""
    todo("M7", "import_reviews")
