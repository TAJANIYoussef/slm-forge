"""Tokenizer coverage per language/script bucket (tokenizer only, no model weights).

``slm-forge diagnose-tokenizer`` measures how each base model's tokenizer handles the
eval text: fertility (tokens per word), characters per token, byte-fallback rate,
encode/decode round-trip exactness, whether Arabizi digit-letters (``3``, ``7``, ``9``
in ``3la``, ``s7i7``, ``l9ism``) are split from their word, and the token premium of each
bucket over English. Target tokenizers split Darija into 2-4x more tokens than English;
the premium sets per-bucket ``max_new_tokens`` (:func:`recommend_max_new_tokens`) and is
why answer confidence is span-level.

The tokenizer is loaded from ``tokenizer.json`` with the ``tokenizers`` library
(imported lazily; no transformers, no torch), offline by default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.eval.dataset import GroundTruthSet


@dataclass(frozen=True)
class TokenizerStats:
    """Coverage numbers for one tokenizer and bucket."""

    bucket: str
    n_texts: int
    n_words: int
    n_tokens: int
    fertility: float  # tokens per whitespace word
    chars_per_token: float
    byte_fallback_rate: float  # share of tokens that are byte-fallback pieces
    roundtrip_exact_rate: float  # share of texts with decode(encode(t)) == t
    arabizi_digit_isolation_rate: float | None  # darija_latn only: digit-letters split off their word
    premium_vs_en: float | None  # n_tokens per char relative to the en bucket; None without en


def diagnose_tokenizer(
    model: str,
    texts_by_bucket: Mapping[str, Sequence[str]],
    *,
    revision: str | None = None,
    offline: bool = True,
) -> dict[str, TokenizerStats]:
    """Tokenizer only, no weights."""
    todo("M8", "diagnose_tokenizer")


def texts_from_ground_truth(gt: GroundTruthSet) -> dict[str, list[str]]:
    """Prompts and answers per bucket."""
    todo("M8", "texts_from_ground_truth")


def recommend_max_new_tokens(
    stats: Mapping[str, TokenizerStats], *, base_en: int = 96, margin: float = 1.2
) -> dict[str, int]:
    """ceil(base_en * premium * margin) per bucket."""
    todo("M8", "recommend_max_new_tokens")
