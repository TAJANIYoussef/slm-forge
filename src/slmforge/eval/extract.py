"""Answer region and typed candidates (plan §2.4).

Prompt contract: each locale's template asks for a short answer that ends with a localized
final line (en ``Final answer:``, fr ``Réponse finale :``, darija_arab ``الجواب:``,
darija_latn ``Jawab:``). :func:`find_answer_region` scores the last such line when present
(``method="final_line"``, which is also ``format_compliance``) and the whole response
otherwise.

Extraction order (earlier spans are masked for later steps): citations, dates, Arabizi
letter-digits, quantities, boolean/choice, alias spans. Values that also appear in the
prompt are dropped unless within tolerance of the gold value (prompt-echo exclusion), and
only candidates with the gold's dimension are compatible.

All spans are half-open ``[start, end)`` offsets into the string the function was given.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Final, Literal

from slmforge._stub import todo
from slmforge.eval.outcomes import AnsweredVia
from slmforge.types import Span

if TYPE_CHECKING:
    from slmforge.eval.abstention import ResponseSignals
    from slmforge.eval.dataset import GroundTruthItem
    from slmforge.eval.match import MatchConfig
    from slmforge.text.lexicons import LexiconMatch, LexiconSet
    from slmforge.text.numbers import UnitTable

CandidateKind = Literal["number", "percent", "money", "duration", "date", "citation", "boolean", "choice"]
RegionMethod = Literal["final_line", "full_text"]

# Leaked chat-turn markers; the response is cut at the earliest one.
TURN_MARKERS: Final[tuple[str, ...]] = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end|>",
    "<|user|>",
    "<|endoftext|>",
    "\nUser:",
    "\nQuestion:",
    # Leaked Alpaca-style turns only: a model may well put its answer under "### Final answer".
    "\n### Instruction",
    "\n### Input",
    "\n### Human",
    "\n### User",
    "\n### Question",
)

# Final-answer markers, matched on the folded line (casefold, accents/harakat/hamza dropped,
# ة→ه, ى→ي). ``[\s*_]*`` lets markdown emphasis sit before the colon ("**Final answer**:").
ANSWER_MARKERS: Final[tuple[str, ...]] = (
    r"final answer[\s*_]*:",
    r"reponse(?: finale)?[\s*_]*:",
    r"الجواب[\s*_]*:",
    r"الاجابه[\s*_]*:",
    r"\b(?:l[- ]?)?jawab[\s*_]*:",
)
_MARKER_RE: Final = re.compile("|".join(f"(?:{p})" for p in ANSWER_MARKERS))
_EDGE_CHARS: Final = " \t\r\n *_`"  # stripped around a region (whitespace, markdown emphasis)
_CHAR_MAP: Final = {"ة": "ه", "ى": "ي", "ٱ": "ا", "ـ": ""}


@dataclass(frozen=True)
class AnswerRegion:
    """The span scored as the answer."""

    text: str
    span: Span
    method: RegionMethod


@dataclass(frozen=True)
class Candidate:
    """A typed answer candidate."""

    kind: CandidateKind
    value: Decimal | str
    unit: str | None
    span: Span
    raw: str
    interpretations: tuple[Decimal | str, ...] = ()  # every plausible reading (locale ambiguity, money scale)
    flags: frozenset[str] = field(default_factory=frozenset)
    negated: bool = False  # preceded by a contrast cue ("pas 80%"): a rejected value


@dataclass(frozen=True)
class CandidateSet:
    """Everything extracted from a response for one item."""

    region: AnswerRegion
    candidates: tuple[Candidate, ...] = ()
    compatible: tuple[Candidate, ...] = ()  # candidates with the gold's dimension
    primary: Candidate | None = None  # first compatible candidate in the region
    gold_spans: tuple[Span, ...] = ()
    known_false_spans: tuple[Span, ...] = ()
    residual_content_tokens: int = 0
    answered_signal: bool = False
    answered_via: AnsweredVia | None = None


def clean_span(text: str) -> Span:
    """Span of :func:`clean_response` within ``text`` (cut at the first turn marker, then stripped)."""
    end = min((i for i in (text.find(m) for m in TURN_MARKERS) if i >= 0), default=len(text))
    start = 0
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end)


def clean_response(text: str) -> str:
    """Cut at the first leaked turn marker and strip."""
    start, end = clean_span(text)
    return text[start:end]


def find_answer_region(response: str) -> AnswerRegion:
    """Last localized final-answer line, else the full text.

    The region is the rest of the marker line, or the next non-empty line when the rest is
    empty. A marker with nothing after it at all falls back to the full text.
    """
    lines = _line_spans(response)
    for li in range(len(lines) - 1, -1, -1):
        ls, le = lines[li]
        folded, idx = _fold(response[ls:le])
        hits = list(_MARKER_RE.finditer(folded))
        if not hits:
            continue
        cut = hits[-1].end()
        rest_start = ls + (idx[cut] if cut < len(idx) else le - ls)
        span = _trim(response, rest_start, le)
        if span[0] == span[1]:
            span = next((s for s in (_trim(response, a, b) for a, b in lines[li + 1 :]) if s[0] < s[1]), span)
        if span[0] < span[1]:
            return AnswerRegion(response[span[0] : span[1]], span, "final_line")
        break
    span = _trim(response, 0, len(response))
    return AnswerRegion(response[span[0] : span[1]], span, "full_text")


def extract_candidates(
    item: GroundTruthItem,
    response: str,
    *,
    signals: ResponseSignals,
    lexicons: LexiconSet,
    units: UnitTable,
    cfg: MatchConfig,
) -> CandidateSet:
    """Citations, dates, quantities, boolean/choice, alias spans; prompt-echo exclusion; answered_signal."""
    todo("M1", "extract_candidates")


def residual_content_tokens(
    response: str, prompt: str, hits: Sequence[LexiconMatch], stopwords: frozenset[str]
) -> int:
    """Content tokens left after removing lexicon hits, prompt echo, stopwords and punctuation."""
    todo("M1", "residual_content_tokens")


# --------------------------------------------------------------------------- helpers


def _line_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    start = 0
    for m in re.finditer(r"\r\n|\n|\r", text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def _trim(text: str, start: int, end: int) -> Span:
    while start < end and text[start] in _EDGE_CHARS:
        start += 1
    while end > start and text[end - 1] in _EDGE_CHARS:
        end -= 1
    return (start, end)


def _fold(s: str) -> tuple[str, list[int]]:
    """Marker-matching fold with an offset map: ``idx[i]`` is the source offset of folded char ``i``.

    Per code point: NFKC, drop combining marks (Latin accents, harakat, hamza seats), map
    ة→ه / ى→ي / ٱ→ا, drop tatweel and format characters, casefold. Deliberately small and
    local so region finding does not depend on the full L1 normalizer.
    """
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(s):
        for c in unicodedata.normalize("NFD", unicodedata.normalize("NFKC", ch)):
            if unicodedata.category(c) in ("Mn", "Cf"):
                continue
            for f in _CHAR_MAP.get(c, c).casefold():
                out.append(f)
                idx.append(i)
    return "".join(out), idx
