"""Shared literals and constants used across the package (no heavy imports)."""

from __future__ import annotations

from typing import Final, Literal

Language = Literal["darija", "fr", "en"]
Script = Literal["arab", "latn", "mixed"]
LangScript = Literal["darija_arab", "darija_latn", "darija_mixed", "fr", "en"]
CORE_BUCKETS: Final[tuple[LangScript, ...]] = ("darija_arab", "darija_latn", "fr", "en")

# Lexicon buckets: every bucket is applied to every response (code-switching).
LexBucket = Literal["en", "fr", "msa", "darija_arab", "darija_latn"]
LEX_BUCKETS: Final[tuple[LexBucket, ...]] = ("en", "fr", "msa", "darija_arab", "darija_latn")

Domain = str  # validated against EvalConfig.allowed_domains; defaults: finance, legal, general
DEFAULT_DOMAINS: Final[tuple[str, ...]] = ("finance", "legal", "general")

AnswerType = Literal[
    "numeric",
    "percent",
    "currency",
    "duration",
    "date",
    "boolean",
    "choice",
    "citation",
    "entity",
    "free_text",
]
NUMERIC_TYPES: Final[frozenset[str]] = frozenset({"numeric", "percent", "currency", "duration"})
TYPED_TYPES: Final[frozenset[str]] = NUMERIC_TYPES | {"date", "boolean", "choice", "citation"}

UnanswerableReason = Literal[
    "false_premise", "nonexistent_entity", "future_or_unknowable", "not_in_any_source", "underspecified"
]
Split = Literal["dev", "test"]

Span = tuple[int, int]
"""Half-open ``[start, end)`` character offsets into the raw response."""


def lang_script_of(language: Language, script: Script | None) -> LangScript:
    """Bucket key: ``darija_<script>`` for Darija, otherwise the language code."""
    if language != "darija":
        return language
    if script is None:
        raise ValueError("Darija items need a resolved script (arab, latn or mixed)")
    return f"darija_{script}"  # type: ignore[return-value]
