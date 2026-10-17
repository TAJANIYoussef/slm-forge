"""Optional vLLM generation backend (M9); ``vllm`` is imported lazily, inside methods.

vLLM serves LoRA adapters without merging and returns ``logprobs=1``, but it is not
bitwise deterministic and its raw-vs-processed logprob behaviour varies by version, so
confidence still comes from the teacher-forced pass (``HFGenerator.score_logprobs``) and
the backend is part of the protocol. Target: >= 98% greedy parity against HF.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.eval.checkpoints import CheckpointRef
    from slmforge.eval.generate import DecodingParams, GenerationRecord, GenRequest
    from slmforge.eval.prompts import PromptBuilder, RenderedPrompt


class VLLMGenerator:
    """vLLM backend with LoRA adapters; not bitwise deterministic (the backend is in the protocol)."""

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: CheckpointRef,
        *,
        builder: PromptBuilder,
        dtype: str = "auto",
        max_lora_rank: int = 64,
        gpu_memory_utilization: float = 0.85,
    ) -> VLLMGenerator:
        """Start a vLLM engine on the base model and register the adapter."""
        todo("M9", "VLLMGenerator.from_checkpoint")

    @property
    def fingerprint(self) -> str:
        """Model fingerprint + dtype + vLLM version + adapter id."""
        todo("M9", "VLLMGenerator.fingerprint")

    def render(self, req: GenRequest) -> RenderedPrompt:
        """Render with the checkpoint's chat template (same as HF)."""
        todo("M9", "VLLMGenerator.render")

    def generate(self, reqs: Sequence[GenRequest], params: DecodingParams) -> list[GenerationRecord]:
        """Greedy generation in input order."""
        todo("M9", "VLLMGenerator.generate")
