"""Reliability eval hooks for the HF Trainer, with transformers imported lazily.

``transformers.TrainerCallback`` cannot be a base class at module level (this module must
import without torch/transformers, and the Unsloth worker must import unsloth first), so
the callback class is built inside :func:`make_reliability_callback` and only delegates
to :class:`ReliabilityEvalHooks`, which holds the logic and is torch-free.

Modes:

* ``posthoc`` (default, M3): ``on_save`` writes ``checkpoint-<step>/.eval_pending`` for
  steps on the eval schedule and ``on_train_end`` writes ``.eval_final`` into the last
  checkpoint; ``eval-run`` later evaluates them in a separate process. Training numerics
  are untouched.
* ``inline`` (M6): ``on_save`` evaluates the checkpoint just saved through the same
  ``HFGenerator`` code path as posthoc, inside a context that calls ``eval()``, enters
  ``inference_mode``, sets ``use_cache=True`` and left padding, disables autocast, toggles
  Unsloth's ``for_inference``/``for_training`` and restores every RNG stream
  (``repro.preserved_rng_state``). ``on_train_begin`` evaluates the base (adapters
  disabled) as step 0. Callbacks get no Trainer handle, so a trainer that is needed for
  logging is passed to the factory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.eval.runner import EvalRunner
    from slmforge.train.backends import TrainBackend

PENDING_MARKER: Final = ".eval_pending"
FINAL_MARKER: Final = ".eval_final"

Mode = Literal["posthoc", "inline"]


class ReliabilityEvalHooks:
    """Torch-free logic behind the Trainer callback (unit-testable without transformers)."""

    def __init__(
        self,
        *,
        mode: Mode,
        every_n_steps: int,
        run_dir: Path,
        runner: EvalRunner | None = None,
        backend: TrainBackend | None = None,
        trainer: Any | None = None,
    ) -> None:
        self.mode = mode
        self.every_n_steps = every_n_steps
        self.run_dir = run_dir
        self.runner = runner
        self.backend = backend
        self.trainer = trainer

    def is_scheduled(self, step: int) -> bool:
        """Whether a checkpoint saved at ``step`` is on the eval schedule (``step % every_n_steps == 0``)."""
        todo("M3", "ReliabilityEvalHooks.is_scheduled")

    def train_begin(self, *, model: Any | None) -> None:
        """Inline mode: evaluate the base (adapters disabled) as the step-0 reference."""
        if self.mode == "inline":
            todo("M6", "ReliabilityEvalHooks.train_begin (inline)")

    def checkpoint_saved(
        self, step: int, checkpoint_dir: Path, *, model: Any | None, tokenizer: Any | None
    ) -> None:
        """posthoc: write ``.eval_pending``; inline: evaluate this checkpoint in place."""
        if self.mode == "inline":
            todo("M6", "ReliabilityEvalHooks.checkpoint_saved (inline)")
        todo("M3", "ReliabilityEvalHooks.checkpoint_saved (posthoc)")

    def train_end(self, final_step: int) -> None:
        """Mark the final checkpoint (``.eval_final``) for the full-set endpoint eval."""
        todo("M3", "ReliabilityEvalHooks.train_end")


def mark_pending(checkpoint_dir: Path, *, step: int, final: bool = False) -> Path:
    """Write the posthoc marker (``.eval_pending``, or ``.eval_final`` for the last checkpoint)."""
    todo("M3", "mark_pending")


def pending_checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoint dirs carrying a marker, sorted by step (what ``eval-run`` picks up)."""
    todo("M3", "pending_checkpoints")


def make_reliability_callback(
    *,
    mode: Mode,
    every_n_steps: int,
    run_dir: Path,
    runner: EvalRunner | None = None,
    backend: TrainBackend | None = None,
    trainer: Any | None = None,
) -> Any:
    """Import ``transformers`` lazily and return a ``TrainerCallback`` instance wrapping the hooks.

    The returned object's ``on_train_begin`` / ``on_save`` / ``on_train_end`` call
    :class:`ReliabilityEvalHooks`; ``on_save`` resolves ``checkpoint-<global_step>``
    under ``args.output_dir``. ``mode="inline"`` requires ``runner`` (and ``backend``).
    """
    if mode == "inline":
        todo("M6", "make_reliability_callback (inline)")
    todo("M3", "make_reliability_callback (posthoc)")
