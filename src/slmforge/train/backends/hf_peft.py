"""Reference backend: ``AutoModelForCausalLM`` + peft. Runs on CPU (CI), MPS and CUDA.

torch, transformers, peft and bitsandbytes are imported **inside** functions only, so
importing this module stays torch-free. 4-bit (QLoRA) loading needs bitsandbytes and
CUDA; without them ``select_backend`` already raised ``BackendUnavailable`` (exit 3).

Loading rules (M3): ``attn_implementation`` and ``dtype`` are passed explicitly (they
change numerics); the vocabulary is never resized by default; the tokenizer gets a pad
token distinct from EOS without resizing (see ``data.loader.choose_pad_token_id``);
``model.config.use_cache`` is turned off while training with gradient checkpointing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.config.schema import QuantConfig, RootConfig


def cuda_available() -> bool:
    """``torch.cuda.is_available()``, importing torch lazily (callers check it is installed)."""
    import torch

    return bool(torch.cuda.is_available())


def bnb_quantization_config(quant: QuantConfig) -> Any:
    """``transformers.BitsAndBytesConfig`` for 4-bit QLoRA (nf4/fp4, double quant, compute dtype)."""
    todo("M6", "bnb_quantization_config")


class HFPeftBackend:
    """AutoModelForCausalLM + peft (bitsandbytes 4-bit only on CUDA, else EnvironmentError); runs on CPU."""

    name: Literal["hf_peft"] = "hf_peft"

    def load(self, cfg: RootConfig) -> tuple[Any, Any]:
        """Load model and tokenizer at ``model.revision`` with explicit dtype and attention implementation."""
        todo("M3", "HFPeftBackend.load")

    def apply_lora(self, model: Any, cfg: RootConfig) -> Any:
        """``peft.get_peft_model`` with targets from ``resolve_targets(model.config.model_type, ...)``."""
        todo("M3", "HFPeftBackend.apply_lora")

    def for_inference(self, model: Any) -> None:
        """``model.eval()``, ``use_cache=True`` and left padding for generation (inline eval)."""
        todo("M3", "HFPeftBackend.for_inference")

    def for_training(self, model: Any) -> None:
        """Restore ``model.train()``, ``use_cache=False`` and right padding."""
        todo("M3", "HFPeftBackend.for_training")
