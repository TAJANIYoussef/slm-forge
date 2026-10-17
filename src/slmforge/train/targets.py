"""LoRA target modules per architecture (torch-free).

``lora.target_modules: auto`` resolves through :data:`DEFAULT_TARGETS`, keyed by the
``model_type`` in the base model's ``config.json``. SmolLM2 is ``llama``; Qwen2.5 is
``qwen2``; Phi-4-mini is ``phi3`` and keeps **fused** ``qkv_proj`` / ``gate_up_proj``.
Unsloth's repackaged ("llamafied") Phi bases split those projections, so an adapter
trained on one cannot be loaded on the other; the adapter manifest records the targets
(see ``train.lora.AdapterManifest``).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

_LLAMA_LIKE: Final = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

DEFAULT_TARGETS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "llama": _LLAMA_LIKE,
        "mistral": _LLAMA_LIKE,
        "qwen2": _LLAMA_LIKE,
        "phi3": ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
    }
)
"""Attention and MLP projections for every architecture the example configs use."""


def resolve_targets(model_type: str, requested: list[str] | Literal["auto", "all-linear"]) -> list[str] | str:
    """Target modules for an architecture.

    ``auto`` looks ``model_type`` up (case-insensitive) in :data:`DEFAULT_TARGETS` and
    raises ``ValueError`` for an unknown architecture rather than guessing names that
    PEFT would silently skip. ``all-linear`` is passed through for PEFT to expand. An
    explicit list is returned as a de-duplicated copy in its original order.
    """
    if requested == "all-linear":
        return "all-linear"
    if requested == "auto":
        key = model_type.strip().lower()
        if key not in DEFAULT_TARGETS:
            known = ", ".join(sorted(DEFAULT_TARGETS))
            raise ValueError(
                f"no default LoRA targets for model_type {model_type!r} (known: {known}); "
                "set lora.target_modules to a list of module names or to 'all-linear'"
            )
        return list(DEFAULT_TARGETS[key])
    if isinstance(requested, str):
        raise ValueError(f"lora.target_modules must be 'auto', 'all-linear' or a list, got {requested!r}")
    names = list(dict.fromkeys(requested))
    if not names or any(not n.strip() for n in names):
        raise ValueError("lora.target_modules must be a non-empty list of non-empty module names")
    return names
