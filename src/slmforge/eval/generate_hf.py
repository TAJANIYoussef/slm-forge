"""Hugging Face generation backend (torch, transformers and peft are imported lazily, inside functions).

* Batches are left-padded, length-sorted and grouped by lang_script, so per-bucket
  ``max_new_tokens`` budgets never pad English up to Darija length.
* Decoding uses a **fresh** ``GenerationConfig`` (:func:`build_generation_config`): greedy,
  ``repetition_penalty=1.0``, ``no_repeat_ngram_size=0``, temperature/top_p/top_k unset,
  explicit eos (+ known end-of-turn tokens) and pad. The model's shipped
  ``generation_config`` (e.g. Qwen2.5's ``repetition_penalty=1.05``) never leaks in, and
  the effective config is recorded in ``metrics.json`` (critique 16).
* Token logprobs come from a separate **teacher-forced** forward pass over rendered prompt +
  response in the eval dtype (:meth:`HFGenerator.score_logprobs`), not from ``generate()``
  scores, which are post-processed by logits processors (critique 5). Each record stores
  per-token logprobs and token -> response-character spans, so rescoring can recompute
  span confidence when the matcher moves the answer span.
* ``attn_implementation`` is set explicitly (eager and sdpa differ numerically).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.eval.checkpoints import CheckpointRef
    from slmforge.eval.generate import DecodingParams, GenerationRecord, GenRequest
    from slmforge.eval.prompts import PromptBuilder, RenderedPrompt
    from slmforge.train.backends import TrainBackend
    from slmforge.types import Span

END_OF_TURN_TOKENS: Final[tuple[str, ...]] = ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>", "</s>")
"""Tokens that end an assistant turn in the supported chat templates; added to eos ids when present."""


def build_generation_config(
    tokenizer: Any, params: DecodingParams, *, bucket: str, eos_token_ids: Sequence[int] | None = None
) -> Any:
    """A fresh greedy ``transformers.GenerationConfig`` for one lang_script bucket (never the model's)."""
    todo("M2", "build_generation_config")


def effective_generation_config(generation_config: Any) -> dict[str, Any]:
    """JSON-safe dump of the generation config actually used (recorded in ``metrics.json``)."""
    todo("M2", "effective_generation_config")


def load_model_and_tokenizer(
    ckpt: CheckpointRef,
    *,
    dtype: str = "auto",
    quantization: str = "none",
    device: str = "auto",
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Base + PeftModel for adapters (after the manifest compatibility check), AutoModelForCausalLM otherwise.

    The tokenizer comes from the checkpoint when it ships one, else from the base. A model
    missing from the local cache in offline mode raises OSError (exit code 3) with a hint.
    """
    todo("M2", "load_model_and_tokenizer")


class HFGenerator:
    """Greedy, left-padded, length-sorted batches per lang_script; teacher-forced logprobs + char spans."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        builder: PromptBuilder,
        batch_size: int = 16,
        dtype: str = "auto",
        quantization: str = "none",
        device: str = "auto",
        model_fingerprint: str = "",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.builder = builder
        self.batch_size = batch_size
        self.dtype = dtype
        self.quantization = quantization
        self.device = device
        self.model_fingerprint = model_fingerprint

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: CheckpointRef,
        *,
        builder: PromptBuilder,
        dtype: str = "auto",
        quantization: str = "none",
        device: str = "auto",
        batch_size: int = 16,
        attn_implementation: str = "sdpa",
    ) -> HFGenerator:
        """Load a checkpoint for generation (:func:`load_model_and_tokenizer`)."""
        todo("M2", "HFGenerator.from_checkpoint")

    def swap_adapter(self, ckpt: CheckpointRef) -> None:
        """``load_adapter`` + ``set_adapter`` + delete the previous adapter (no base reload)."""
        todo("M3", "HFGenerator.swap_adapter")

    def use_base(self) -> None:
        """Disable adapters for the step-0 reference."""
        todo("M3", "HFGenerator.use_base")

    @property
    def fingerprint(self) -> str:
        """Model fingerprint + dtype/quantization/batch/device/attn + torch/transformers versions."""
        todo("M2", "HFGenerator.fingerprint")

    def render(self, req: GenRequest) -> RenderedPrompt:
        """Render via the checkpoint's own chat template (``PromptBuilder.render``)."""
        todo("M2", "HFGenerator.render")

    def generate(self, reqs: Sequence[GenRequest], params: DecodingParams) -> list[GenerationRecord]:
        """Greedy generation (:func:`build_generation_config`), then :meth:`score_logprobs`; input order."""
        todo("M2", "HFGenerator.generate")

    def score_logprobs(
        self, prompts: Sequence[RenderedPrompt], responses: Sequence[str]
    ) -> list[tuple[tuple[float, ...], tuple[Span, ...]]]:
        """Teacher-forced pass over prompt + response: per-token logprobs and response char spans."""
        todo("M2", "HFGenerator.score_logprobs")


class InMemoryGenerator(HFGenerator):
    """Wraps the live training model for inline eval; toggles backend.for_inference/for_training around it."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        backend: TrainBackend,
        builder: PromptBuilder,
        step: int,
        batch_size: int = 16,
    ) -> None:
        todo("M6", "InMemoryGenerator.__init__")
