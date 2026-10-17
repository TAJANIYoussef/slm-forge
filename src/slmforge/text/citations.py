"""Legal article and instrument references ("article 12 de la loi 31-08", الفصل 12, l-fasl 12).

Citations are extracted (and masked) before dates and quantities, so "loi 31.08" is an
instrument id, never the decimal 31.08. Instrument ids are normalized to ``NN-YY``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from slmforge._stub import todo
from slmforge.types import Span

# Spec patterns (on L1 text; implemented in M4).
ARTICLE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?:article|art\.?)\s*(\d+(?:[-.]\d+)?)(?:\s*(?:bis|ter))?"),
    re.compile(r"(?:الفصل|الماده|فصل|ماده)\s*(\d+)"),
    re.compile(r"(?:l[- ]?)?(?:fasl|fsl|mada|madda)\s*(\d+)"),
)
INSTRUMENT_PATTERN: Final = re.compile(
    r"(?:loi|law|dahir|قانون|9anon|qanoun)\s*(?:n[°o]\s*)?(\d{1,3})[.\-/](\d{2,4})"
)


@dataclass(frozen=True)
class CitationRef:
    """An article reference, optionally with an instrument id like '31-08'."""

    article: str | None
    instrument: str | None
    span: Span
    raw: str


def extract_citations(text: str) -> list[CitationRef]:
    """FR/EN/Arabic/Arabizi article and law references."""
    todo("M4", "extract_citations")
