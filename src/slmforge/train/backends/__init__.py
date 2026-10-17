"""Training backends: model loading and LoRA wrapping (the Trainer itself is shared).

The backend is chosen **before** transformers is imported, because Unsloth patches
transformers, peft and trl on import. :func:`select_backend` therefore decides from
``importlib.util.find_spec`` alone and only touches torch (through
``hf_peft.cuda_available``, inside a function) when it actually needs the CUDA answer.
Neither this module nor importing ``hf_peft`` / ``unsloth`` loads torch.

Unsloth is never imported in the calling process: ``train.lora.finetune`` runs an
Unsloth run in a fresh ``python -m slmforge.train.worker`` subprocess whose first import
is ``unsloth`` (and the test suite never imports it at all).

Runtime rules (each failure is a :class:`BackendUnavailable`, an ``EnvironmentError``
that the CLI maps to exit code 3):

* ``unsloth`` needs the ``unsloth`` package and an NVIDIA CUDA GPU.
* ``hf_peft`` needs torch, transformers and peft (the ``[hf]`` extra); with
  ``quant.load_in_4bit`` it also needs bitsandbytes and CUDA. QLoRA never silently
  downgrades to 16-bit LoRA.
* ``auto`` picks ``unsloth`` iff it is importable and CUDA is available, else ``hf_peft``.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from slmforge.config.schema import RootConfig

log = logging.getLogger(__name__)

BackendName = Literal["unsloth", "hf_peft"]
Requested = Literal["auto", "unsloth", "hf_peft"]

HF_MODULES = ("torch", "transformers", "peft")


class BackendUnavailable(OSError):
    """The requested backend cannot run on this machine (an ``EnvironmentError``: CLI exit code 3)."""


@runtime_checkable
class TrainBackend(Protocol):
    """Model loading and LoRA wrapping; the Trainer itself is shared."""

    @property
    def name(self) -> BackendName:
        """``unsloth`` or ``hf_peft`` (a class attribute on implementations)."""
        ...

    def load(self, cfg: RootConfig) -> tuple[Any, Any]:
        """Load model and tokenizer (4-bit if configured)."""
        ...

    def apply_lora(self, model: Any, cfg: RootConfig) -> Any:
        """Wrap with LoRA adapters."""
        ...

    def for_inference(self, model: Any) -> None:
        """Switch to inference mode (inline eval)."""
        ...

    def for_training(self, model: Any) -> None:
        """Switch back to training mode."""
        ...


@dataclass(frozen=True)
class BackendChoice:
    """Which backend runs and why (logged and recorded in the run manifest)."""

    name: BackendName
    reason: str
    cuda: bool | None  # None when the decision did not need to probe CUDA


def module_available(name: str) -> bool:
    """Whether a top-level module is importable, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # broken install, or a module whose __spec__ is None
        return False


def cuda_available() -> bool:
    """Whether torch sees a CUDA device; ``False`` without importing anything when torch is absent."""
    if not module_available("torch"):
        return False
    from slmforge.train.backends.hf_peft import cuda_available as probe

    return probe()


def choose_backend(
    requested: Requested,
    *,
    load_in_4bit: bool = False,
    cuda: bool | None = None,
    has_module: Callable[[str], bool] | None = None,
) -> BackendChoice:
    """Decide the backend from what is installed; ``cuda=None`` probes lazily, only if needed.

    Raises :class:`BackendUnavailable` (exit code 3) instead of falling back when the
    request cannot be honoured. ``has_module`` and ``cuda`` exist for tests.
    """
    has_module = has_module or module_available
    probed: list[bool] = []

    def has_cuda() -> bool:
        if cuda is not None:
            return cuda
        if not probed:
            probed.append(has_module("torch") and cuda_available())
        return probed[0]

    def seen() -> bool | None:
        return cuda if cuda is not None else (probed[0] if probed else None)

    if requested not in ("auto", "unsloth", "hf_peft"):
        raise ValueError(f"train.backend must be auto, unsloth or hf_peft, got {requested!r}")

    if requested == "unsloth":
        if not has_module("unsloth"):
            raise BackendUnavailable(
                "train.backend=unsloth but unsloth is not installed (pip install 'slm-forge[unsloth]' on an "
                "NVIDIA CUDA machine), or use --backend hf_peft; exit code 3"
            )
        if not has_cuda():
            raise BackendUnavailable(
                "train.backend=unsloth needs an NVIDIA CUDA GPU and none is available; use --backend hf_peft "
                "(CPU/MPS, 16-bit LoRA) or run on a CUDA machine; exit code 3"
            )
        return BackendChoice("unsloth", "requested explicitly; unsloth importable and CUDA available", True)

    if requested == "auto":
        if has_module("unsloth") and has_cuda():
            return BackendChoice("unsloth", "auto: unsloth importable and CUDA available", True)
        why = "unsloth not installed" if not has_module("unsloth") else "no CUDA device"
        reason = f"auto: {why}, using hf_peft"
    else:
        reason = "requested explicitly"

    missing = [m for m in HF_MODULES if not has_module(m)]
    if missing:
        raise BackendUnavailable(
            f"the hf_peft backend needs {', '.join(missing)} (pip install 'slm-forge[hf]'); exit code 3"
        )
    if load_in_4bit:
        if not has_module("bitsandbytes"):
            raise BackendUnavailable(
                "quant.load_in_4bit needs bitsandbytes (pip install 'slm-forge[qlora]'); "
                "set quant.load_in_4bit=false for 16-bit LoRA; exit code 3"
            )
        if not has_cuda():
            raise BackendUnavailable(
                "quant.load_in_4bit (QLoRA) needs an NVIDIA CUDA GPU and none is available; "
                "set quant.load_in_4bit=false for 16-bit LoRA on CPU/MPS; exit code 3"
            )
    return BackendChoice("hf_peft", reason, seen())


def select_backend(requested: Requested, *, load_in_4bit: bool = False) -> TrainBackend:
    """Resolve and instantiate the backend; call before importing transformers.

    ``auto`` -> unsloth iff CUDA is available and unsloth importable; the reason is
    logged. Instantiating a backend imports nothing heavy: every backend method imports
    its libraries lazily.
    """
    choice = choose_backend(requested, load_in_4bit=load_in_4bit)
    log.info("training backend: %s (%s)", choice.name, choice.reason)
    if choice.name == "unsloth":
        from slmforge.train.backends.unsloth import UnslothBackend

        return UnslothBackend()
    from slmforge.train.backends.hf_peft import HFPeftBackend

    return HFPeftBackend()


__all__ = [
    "BackendChoice",
    "BackendName",
    "BackendUnavailable",
    "TrainBackend",
    "choose_backend",
    "cuda_available",
    "module_available",
    "select_backend",
]
