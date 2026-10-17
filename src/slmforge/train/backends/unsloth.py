"""Unsloth backend (NVIDIA CUDA as tested): ``FastLanguageModel`` loading and LoRA wrapping.

``import unsloth`` must come before transformers, peft and trl, raises without a
supported GPU, and patches those libraries for the rest of the process. So this module
never imports it at module level, the pytest process never imports it at all, and the
only caller is ``slmforge.train.worker``: a fresh subprocess that imports unsloth first
(``train.lora.finetune`` spawns it). Exact versions are pinned in
``requirements/cuda-unsloth.lock`` and must match the CPU lock for transformers/peft/trl.

Adapters trained on Unsloth's repackaged bases ("llamafied" Phi with split q/k/v,
``-bnb-4bit`` snapshots) only load on that same base; the adapter manifest records it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.config.schema import RootConfig


class UnslothBackend:
    """FastLanguageModel.from_pretrained / get_peft_model / for_inference / for_training; CUDA only."""

    name: Literal["unsloth"] = "unsloth"

    def load(self, cfg: RootConfig) -> tuple[Any, Any]:
        """``FastLanguageModel.from_pretrained`` (4-bit if ``quant.load_in_4bit``) at the pinned revision."""
        todo("M6", "UnslothBackend.load")

    def apply_lora(self, model: Any, cfg: RootConfig) -> Any:
        """``FastLanguageModel.get_peft_model`` with resolved targets and the configured r/alpha/dropout."""
        todo("M6", "UnslothBackend.apply_lora")

    def for_inference(self, model: Any) -> None:
        """``FastLanguageModel.for_inference(model)`` (inline eval)."""
        todo("M6", "UnslothBackend.for_inference")

    def for_training(self, model: Any) -> None:
        """``FastLanguageModel.for_training(model)``."""
        todo("M6", "UnslothBackend.for_training")
