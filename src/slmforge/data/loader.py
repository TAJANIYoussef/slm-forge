"""Training data: records, formats, language balancing and assistant-masked tokenization.

Input formats (JSONL, one object per line; blank lines and ``//`` lines are skipped):

* ``chat``: ``{"messages": [{"role", "content"}, ...], "id"?, "language"?, "script"?,
  "domain"?, "provenance"?}`` -- the :class:`ChatRecord` fields (see
  ``data/train/sample_chat.jsonl``);
* ``prompt_response``: ``{"prompt", "response", "system"?, ...}``;
* ``alpaca``: ``{"instruction", "input"?, "output", ...}``.

Text is normalized with ``text.normalize.normalize_light`` (L0) only: matching
normalization never touches training text.

Labels (M3) are built from **assistant-turn boundaries**, never from the pad id: base
Qwen2.5 has ``pad == eos`` and Mistral has no pad token, so masking by pad id would stop
EOS from ever being trained, and the model would stop terminating (truncation and INVALID
would then rise over checkpoints and read as a reliability trend). Every assistant turn's
end-of-turn/EOS token keeps a real label, which a test asserts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from slmforge._stub import todo
from slmforge.types import Language, Script

if TYPE_CHECKING:
    from slmforge.config.schema import DataSpec
    from slmforge.eval.prompts import PromptTemplateSet

TrainFormat = Literal["chat", "prompt_response", "alpaca"]
ProvenanceKind = Literal["human", "distilled", "abstention_mix"]
PROVENANCE_KINDS: Final[frozenset[str]] = frozenset({"human", "distilled", "abstention_mix"})
ROLES: Final[frozenset[str]] = frozenset({"system", "user", "assistant"})
IGNORE_INDEX: Final = -100


class ChatRecord(BaseModel):
    """One training conversation with language tags and provenance.

    Invariants: every message is ``{"role", "content"}`` with a known role and string
    content; a system turn may only come first; the last turn is the assistant's.
    ``provenance["kind"]``, when present, is ``human``, ``distilled`` or ``abstention_mix``
    (distilled records also carry the teacher id, date and request parameters).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    messages: list[dict[str, str]] = Field(min_length=1)
    language: Language | None = None
    script: Script | None = None
    domain: str | None = None
    provenance: dict[str, Any] = {}

    @field_validator("messages")
    @classmethod
    def _check_messages(cls, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        for n, m in enumerate(messages):
            if set(m) != {"role", "content"}:
                raise ValueError(f"message {n}: keys must be exactly role and content, got {sorted(m)}")
            if m["role"] not in ROLES:
                raise ValueError(f"message {n}: unknown role {m['role']!r} (system, user or assistant)")
            if not isinstance(m["content"], str):
                raise ValueError(f"message {n}: content must be a string")
            if m["role"] == "system" and n != 0:
                raise ValueError(f"message {n}: a system turn may only come first")
        if messages[-1]["role"] != "assistant":
            raise ValueError("the last turn must be the assistant's (it is what gets trained)")
        return messages

    @model_validator(mode="after")
    def _check_tags(self) -> ChatRecord:
        if self.language is not None and self.language != "darija" and self.script == "arab":
            raise ValueError("fr/en records cannot have script=arab")
        kind = self.provenance.get("kind")
        if kind is not None and kind not in PROVENANCE_KINDS:
            raise ValueError(f"provenance.kind must be one of {sorted(PROVENANCE_KINDS)}, got {kind!r}")
        return self

    def sampling_key(self, key: Literal["language", "lang_script"]) -> str:
        """Bucket used for language resampling: ``unknown``, the language, or ``darija_<script>``."""
        if self.language is None:
            return "unknown"
        if key == "language" or self.language != "darija":
            return self.language
        return f"darija_{self.script}" if self.script is not None else "darija"


def language_sampling_weights(counts: Mapping[str, int], alpha: float) -> dict[str, float]:
    """``p_l ∝ n_l^alpha``, normalized to sum to 1 (sorted by key).

    ``alpha = 1`` keeps the natural proportions; ``alpha < 1`` flattens them, upsampling
    small buckets such as Darija. Buckets with ``n_l = 0`` get probability 0.
    """
    if not alpha > 0 or not math.isfinite(alpha):
        raise ValueError(f"alpha must be a positive finite number, got {alpha!r}")
    if any(n < 0 for n in counts.values()):
        raise ValueError(f"counts must be non-negative, got {dict(counts)}")
    mass = {k: float(n) ** alpha if n > 0 else 0.0 for k, n in sorted(counts.items())}
    total = math.fsum(mass.values())
    if total <= 0:
        raise ValueError("counts must contain at least one positive count")
    return {k: v / total for k, v in mass.items()}


def detect_format(obj: Mapping[str, Any]) -> TrainFormat:
    """Infer the format from the keys: messages, prompt/response or instruction/output."""
    todo("M3", "detect_format")


def to_chat_record(obj: Mapping[str, Any], fmt: TrainFormat, *, source: Path, line: int) -> ChatRecord:
    """Convert one parsed line to a :class:`ChatRecord` (L0 text; id derived from content if absent)."""
    todo("M3", "to_chat_record")


def load_train_records(
    specs: Sequence[DataSpec], *, seed: int, max_samples: int | None = None
) -> list[ChatRecord]:
    """Load chat / prompt_response / alpaca JSONL; normalize_light only; last turn must be assistant."""
    todo("M3", "load_train_records")


def resample_by_language(
    records: Sequence[ChatRecord], *, alpha: float, key: Literal["language", "lang_script"], seed: int
) -> list[ChatRecord]:
    """Temperature resampling with :func:`language_sampling_weights` (same total size, seeded)."""
    todo("M3", "resample_by_language")


def load_abstention_mix(
    path: Path, templates: PromptTemplateSet, *, ratio: float, seed: int
) -> list[ChatRecord]:
    """Unanswerable training prompts answered with the locale's canonical abstention."""
    todo("M3", "load_abstention_mix")


def choose_pad_token_id(tokenizer: Any) -> int:
    """A pad id distinct from EOS without resizing the vocabulary (existing pad/unk/reserved token)."""
    todo("M3", "choose_pad_token_id")


def tokenize_with_assistant_mask(rec: ChatRecord, tokenizer: Any, *, max_len: int) -> dict[str, list[int]]:
    """labels=-100 outside assistant turns; asserts chat-template prefix stability.

    Each assistant turn is located by rendering the conversation prefix with and without
    it (``apply_chat_template``); a template that is not prefix-stable raises. The turn's
    closing end-of-turn/EOS token is labelled. Over-long records are truncated from the
    left of the prompt, never through the final assistant turn.
    """
    todo("M3", "tokenize_with_assistant_mask")


def assert_final_eos_labeled(features: Mapping[str, Sequence[int]], eos_token_ids: Sequence[int]) -> None:
    """Raise unless the last labelled token of an example is an EOS/end-of-turn id."""
    todo("M3", "assert_final_eos_labeled")


class CompletionCollator:
    """Right-pads input_ids/attention_mask and pads labels with -100."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: Sequence[Mapping[str, list[int]]]) -> dict[str, Any]:
        """Pad a batch to its longest example and return torch tensors (torch imported lazily)."""
        todo("M3", "CompletionCollator.__call__")


def build_sft_dataset(records: Sequence[ChatRecord], tokenizer: Any, *, max_seq_len: int) -> Any:
    """Tokenized dataset (no ``datasets`` map cache, or one keyed on tokenizer + template hash)."""
    todo("M3", "build_sft_dataset")


def dataset_fingerprint(records: Sequence[ChatRecord]) -> str:
    """Order-independent hash of records."""
    todo("M3", "dataset_fingerprint")
