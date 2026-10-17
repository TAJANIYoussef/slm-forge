"""Training subprocess for the Unsloth backend: ``python -m slmforge.train.worker``.

Unsloth must be the first ML import of a process and patches transformers/peft/trl
globally, so the CLI process (which may already have imported other libraries, and
which runs eval afterwards) never imports it. ``train.lora.finetune`` writes
``resolved_config.yaml`` into the run directory and spawns::

    python -m slmforge.train.worker --config runs/<name>/resolved_config.yaml --run-dir runs/<name> [--resume]

:func:`main` imports ``unsloth`` **before anything else heavy**, rebuilds the config,
runs ``train.lora.train_in_process`` with ``UnslothBackend`` and exits 0, or 3 when
Unsloth/CUDA is unavailable. The parent reads ``run_manifest.json`` and the checkpoints
afterwards; nothing is passed back over pipes. Importing this module loads nothing heavy.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from slmforge._stub import todo

WORKER_MODULE: Final = "slmforge.train.worker"


def worker_command(
    resolved_config: Path, run_dir: Path, *, resume: bool, python: str | None = None
) -> list[str]:
    """argv for the subprocess (``sys.executable -m slmforge.train.worker ...``)."""
    todo("M6", "worker_command")


def spawn_worker(resolved_config: Path, run_dir: Path, *, resume: bool) -> int:
    """Run the worker in a fresh process with the parent's environment; return its exit code."""
    todo("M6", "spawn_worker")


def main(argv: Sequence[str] | None = None) -> int:
    """Worker entry point: import unsloth first, load the resolved config, train, return the exit code."""
    todo("M6", "train.worker.main")


if __name__ == "__main__":
    raise SystemExit(main())
