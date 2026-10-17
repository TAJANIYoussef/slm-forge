"""Partial dates (year, month or day granularity) in FR/EN/MSA/Moroccan/Arabizi text.

Numeric dates are read day-first (``15/03/2024``); month-first is used only with an
``en`` hint and when day-first is impossible. ISO dates and month names (French, English,
MSA, Moroccan ``غشت``/``شتنبر`` and Arabizi ``ghocht``/``chtanbr``) come from
``resources/months.yaml``. Predictions are compared at the gold's granularity; a
prediction coarser than the gold is ``coarser`` (UNRESOLVED), never a match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from slmforge._stub import todo
from slmforge.types import Language, Span

if TYPE_CHECKING:
    from slmforge.eval.dataset import DateSpec


@dataclass(frozen=True)
class DateValue:
    """A partial date with granularity."""

    year: int
    month: int | None
    day: int | None
    span: Span
    raw: str

    @property
    def granularity(self) -> Literal["year", "month", "day"]:
        """Finest component present."""
        return "year" if self.month is None else "month" if self.day is None else "day"


def extract_dates(text: str, *, lang_hint: Language | None) -> list[DateValue]:
    """Day-first numeric dates, ISO, FR/EN/MSA/Moroccan/Arabizi month names.

    M1 covers numeric/ISO dates and French/English month names; M4 adds MSA, Moroccan and
    Arabizi month names.
    """
    todo("M1", "extract_dates")


def compare_date(pred: DateValue, gold: DateSpec) -> Literal["equal", "different", "coarser"]:
    """Compare at the gold granularity."""
    todo("M1", "compare_date")
