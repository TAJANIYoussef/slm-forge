"""Language and script identification for responses (Darija vs MSA vs French vs English).

Used for language/script fidelity metrics and LANGUAGE_DRIFT flags, and to decide when
Darija skeleton matching applies. The default backend is rule-based and deterministic:
it scores Darija markers (ديال, واش, دابا, the كن- verb prefix, -ش negation; dyal, wach,
daba, kan-, -ch), MSA function words, and French/English stopwords. An optional GlotLID
fastText backend (``ary_Arab``/``ary_Latn`` -> darija) can be configured with a local model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from slmforge._stub import todo
from slmforge.types import Script

if TYPE_CHECKING:
    from slmforge.text.lexicons import LexiconSet

LangLabel = Literal["darija", "msa", "fr", "en", "unk"]
MIN_CONTENT_TOKENS = 3  # fewer content tokens -> "unk"
CODE_SWITCH_MIN_SHARE = 0.3  # a second language with >= this share marks the text code-switched


@dataclass(frozen=True)
class LangProfile:
    """Primary language, script and shares for a text."""

    primary: LangLabel
    script: Script | None
    lang_shares: Mapping[str, float]
    code_switched: bool
    n_content_tokens: int


class LangIdBackend(Protocol):
    """Anything that identifies language and script."""

    def identify(self, text: str) -> LangProfile:
        """Identify language and script."""
        ...


class RuleLangId:
    """Deterministic marker/stopword scorer; fewer than 3 content tokens gives ``unk``."""

    def __init__(self, lexicons: LexiconSet) -> None:
        self.lexicons = lexicons

    def identify(self, text: str) -> LangProfile:
        """Rule-based identification."""
        todo("M4", "RuleLangId.identify")


class GlotLidBackend:
    """Optional backend from a local GlotLID fastText model file (never downloaded at eval time)."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)

    def identify(self, text: str) -> LangProfile:
        """fastText identification mapped onto :data:`LangLabel`."""
        todo("M8", "GlotLidBackend.identify")


def get_langid(
    backend: Literal["rule", "glotlid"], lexicons: LexiconSet, *, model_path: Path | None = None
) -> LangIdBackend:
    """Factory for the configured backend."""
    if backend == "rule":
        return RuleLangId(lexicons)
    if model_path is None:
        raise ValueError("the glotlid backend needs a local model_path")
    return GlotLidBackend(model_path)
