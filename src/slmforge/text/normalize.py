"""Text normalization levels, tokens and script profiles (torch-free).

Three levels, each with one job:

* **L0** :func:`normalize_light` -- training text, prompts and display. NFC, invisible
  bidi/format characters and tatweel removed, whitespace collapsed. Never strips
  diacritics or changes letters.
* **L1** :func:`normalize_for_match` -- matching. L0, then NFKC, digit folding, mark
  stripping (harakat, hamza marks, Latin accents), an Arabic letter map, a punctuation
  map, casefold and letter-run collapsing. Returns an offset map back to the raw text.
* **L2** ``text.arabizi.skeleton`` -- a consonant skeleton shared by both Darija scripts.

Every surface form under ``resources/`` (lexicon literals, number words, month names,
unit names) is written in L1 form; :func:`l1_char_problem` says why a character can
never appear in L1 output, and the resource loaders use it to reject cues that could
never match. Full spec: ``docs/metrics.md`` (normalization).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Literal

from slmforge._stub import todo
from slmforge.types import Script, Span

NORMALIZER_VERSION: Final = "1.0.0"

# L0 deletions: ZWSP..RLM, LRE..RLO, LRI..PDI, ALM, BOM/ZWNBSP, tatweel.
_L0_DELETE: Final = frozenset(
    chr(c)
    for c in (*range(0x200B, 0x2010), *range(0x202A, 0x202F), *range(0x2066, 0x206A), 0x061C, 0xFEFF, 0x0640)
)
_L0_DELETE_TABLE: Final = dict.fromkeys(map(ord, _L0_DELETE))
_SPACE_RUN_RE: Final = re.compile(r"[\s    　]+")
# str.splitlines() boundaries; with keep_newlines=True a whitespace run containing one becomes "\n".
_LINE_BREAKS: Final = frozenset("\n\r\x0b\x0c\x1c\x1d\x1e\x85  ")

# Arabic-Indic (U+0660..0669) and Extended Arabic-Indic / Persian (U+06F0..06F9) digits.
_DIGIT_TABLE: Final = {
    **{0x0660 + i: ord("0") + i for i in range(10)},
    **{0x06F0 + i: ord("0") + i for i in range(10)},
}

# Characters the L1 letter and punctuation maps (steps 5-6) rewrite or delete.
_L1_MAPPED: Final = frozenset("ٱىةکیۀڭگڤڨءœæ،؛؟٪٫٬«»“”„‘’ʼ`´–—")

# All-caps Latin acronyms (LTV, BAM, MAD), bounded by non-letters; digits do not bound-break.
_ACRONYM_RE: Final = re.compile(r"(?<![^\W\d_])[A-Z]{2,}(?![^\W\d_])")
_COUNTED_LETTER_CATEGORIES: Final = frozenset(
    {"Lu", "Ll", "Lt", "Lo"}
)  # Lm (tatweel, modifier letters) excluded
SCRIPT_THRESHOLD: Final = 0.7


@dataclass(frozen=True)
class Normalized:
    """L1 text plus a map from each normalized char to its raw-text offset."""

    text: str
    idx_map: tuple[int, ...]

    def to_raw_span(self, span: Span) -> Span:
        """Map a normalized span back to raw offsets."""
        todo("M1", "Normalized.to_raw_span")


@dataclass(frozen=True)
class Token:
    """A word token with raw offsets and script class."""

    text: str
    start: int
    end: int
    script: Literal["arab", "latn", "digit", "punct", "other"]


@dataclass(frozen=True)
class ScriptProfile:
    """Letter shares by script (acronyms and digits ignored)."""

    arab_frac: float
    latn_frac: float
    dominant: Script | None  # None when the text has no letters


def normalize_light(s: str, *, keep_newlines: bool = False) -> str:
    """L0: NFC, strip bidi/format chars and tatweel, collapse spaces.

    Steps: NFC; delete U+200B-U+200F, U+202A-U+202E, U+2066-U+2069, U+061C, U+FEFF and
    tatweel U+0640; collapse runs of whitespace (incl. NBSP, NNBSP, thin, figure and
    ideographic spaces) to one space; strip. A second NFC runs after the deletions so the
    function is idempotent even when a deleted character separated a base and its mark.

    ``keep_newlines=True`` (an addition to the spec, for line-oriented callers such as the
    final-answer finder): a whitespace run that contains a line break becomes one ``"\\n"``.
    """
    s = unicodedata.normalize("NFC", s).translate(_L0_DELETE_TABLE)
    s = unicodedata.normalize("NFC", s)
    if keep_newlines:
        return _SPACE_RUN_RE.sub(lambda m: "\n" if _LINE_BREAKS.intersection(m.group()) else " ", s).strip()
    return _SPACE_RUN_RE.sub(" ", s).strip()


def fold_digits(s: str) -> str:
    """Arabic-Indic and Extended Arabic-Indic digits to ASCII."""
    return s.translate(_DIGIT_TABLE)


def l1_char_problem(ch: str) -> str | None:
    """Why ``ch`` can never appear in L1 output, or None if it can (used to validate resources)."""
    if len(ch) != 1:
        raise ValueError(f"expected one character, got {ch!r}")
    if ch in _L0_DELETE:
        return "deleted by L0 (bidi/format character or tatweel)"
    if ch.isspace() and ch != " ":
        return "whitespace other than U+0020 is collapsed to a space"
    if ord(ch) in _DIGIT_TABLE:
        return "Arabic-Indic digits are folded to ASCII"
    if ch in _L1_MAPPED:
        return "rewritten by the L1 letter/punctuation map"
    if unicodedata.normalize("NFKC", ch) != ch:
        return "changed by NFKC (presentation form, full-width or compatibility character)"
    if any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", ch)):
        return "carries a combining mark (harakat, hamza mark or accent), which L1 drops"
    if ch.casefold() != ch:
        return "not casefolded"
    return None


def normalize_for_match(s: str) -> Normalized:
    """L1 matching normalization (idempotent), with offset map.

    One pass over code points with per-code-point NFKC (a documented deviation from
    whole-string NFKC): L0; NFKC; fold digits; NFD + drop Mn + NFC; letter map
    (ٱ→ا ى→ي ة→ه ک→ك ی→ي ۀ→ه ڭ/گ→ك ڤ→ف ڨ→ق, standalone ء→"", œ→oe, æ→ae);
    punctuation map (،→, ؛→; ؟→? ٪→% ٫→. ٬→, quotes→" or ', dashes→-); casefold;
    runs of >= 3 identical letters → 1 (``([^\\W\\d_])\\1{2,}``); collapse whitespace.
    """
    todo("M1", "normalize_for_match")


def norm(s: str) -> str:
    """Shorthand for normalize_for_match(s).text."""
    todo("M1", "norm")


def tokenize(s: str) -> list[Token]:
    """Unicode word tokenization with script tags."""
    todo("M1", "tokenize")


def strip_arabic_clitic(tok: str) -> str:
    """Conservative al-/wal-/bal-/fal-/kal-/lil- stripping.

    Applied symmetrically at L1 only, and only when at least 3 letters remain (so بالغ,
    والي and فالح are untouched): وال/فال/بال/كال drop the first letter; لل with length > 4
    becomes ال; ال with length >= 4 is dropped. Bare one-letter proclitics are never
    stripped (بنك must not become نك).
    """
    todo("M4", "strip_arabic_clitic")


def split_sentences(s: str) -> list[Span]:
    """Split on [.!?؟\\n]+."""
    todo("M1", "split_sentences")


def _script_counts(s: str) -> tuple[int, int, int]:
    arab = latn = total = 0
    for ch in s:
        if unicodedata.category(ch) not in _COUNTED_LETTER_CATEGORIES:
            continue
        total += 1
        name = unicodedata.name(ch, "")
        if name.startswith("ARABIC"):
            arab += 1
        elif name.startswith("LATIN"):
            latn += 1
    return arab, latn, total


def script_profile(s: str, *, ignore_acronyms: bool = True) -> ScriptProfile:
    """Classify arab/latn/mixed at 0.7.

    Counts letters only (digits, punctuation, marks and tatweel are ignored). With
    ``ignore_acronyms`` all-caps Latin runs of >= 2 letters (LTV, BAM, MAD) are skipped,
    unless that would leave no letters at all (all-caps text), in which case they count.
    Shares are over all counted letters, so other scripts dilute both.
    """
    arab, latn, total = _script_counts(_ACRONYM_RE.sub(" ", s) if ignore_acronyms else s)
    if total == 0 and ignore_acronyms:
        arab, latn, total = _script_counts(s)
    if total == 0:
        return ScriptProfile(0.0, 0.0, None)
    arab_frac, latn_frac = arab / total, latn / total
    dominant: Script = (
        "arab" if arab_frac >= SCRIPT_THRESHOLD else "latn" if latn_frac >= SCRIPT_THRESHOLD else "mixed"
    )
    return ScriptProfile(arab_frac, latn_frac, dominant)
