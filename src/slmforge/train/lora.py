"""LoRA/QLoRA fine-tuning orchestration (torch-free at import; backends are imported lazily).

:func:`finetune` is what ``slm-forge finetune`` calls. The order matters:

1. ``select_backend`` (before any transformers import; Unsloth patches it on import);
2. ``prepare_run_dir`` + ``resolved_config.yaml`` + ``run_manifest.json`` (status ``running``);
3. for ``unsloth``: spawn ``python -m slmforge.train.worker`` and wait (Unsloth is never
   imported in this process); for ``hf_peft``: :func:`train_in_process` here;
4. ``train_in_process``: seed (``repro.set_determinism``), load + LoRA, data (contamination
   guard, language resampling, assistant-mask labels), Trainer with the reliability
   callback, ``train_log.jsonl``, an adapter manifest + tokenizer in every checkpoint;
5. free the GPU, mark the manifest ``finished``, then ``eval-run`` (posthoc) if
   ``eval.schedule.after_training`` and not ``skip_eval``.

Inline eval is off by default (``eval.schedule.mode: posthoc``): eval-run runs in a
separate step after training, never concurrently on the same GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.config.schema import RootConfig
    from slmforge.repro import Provenance
    from slmforge.train.backends import TrainBackend

RUN_MANIFEST_NAME: Final = "run_manifest.json"
RESOLVED_CONFIG_NAME: Final = "resolved_config.yaml"
TRAIN_LOG_NAME: Final = "train_log.jsonl"
ADAPTER_MANIFEST_NAME: Final = "slmforge_adapter_manifest.json"


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a fine-tuning run."""

    run_dir: Path
    checkpoints: tuple[Path, ...]
    config_sha256: str
    final_step: int
    backend: str


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunManifest(_Record):
    """``runs/<name>/run_manifest.json``: what was trained, on what, and whether it finished."""

    schema_version: Literal[1] = 1
    run_name: str
    status: Literal["running", "finished", "failed"]
    started_at: str
    finished_at: str | None = None
    config_sha256: str  # repro.config_sha256 after --set overrides
    config: dict[str, Any]  # resolved config, model_dump(mode="json")
    base_model: str
    base_revision: str | None  # resolved commit SHA of the snapshot actually loaded
    backend: Literal["unsloth", "hf_peft"]
    backend_reason: str
    provenance: dict[str, Any]  # repro.Provenance.model_dump(mode="json")
    data_fingerprints: dict[str, str]  # train file path -> loader.dataset_fingerprint
    contamination: dict[str, Any]  # guard summary: policy, n_checked, n_flagged, n_dropped
    checkpoints: list[str] = []  # checkpoint-<step> dir names, filled in as they are saved


class AdapterManifest(_Record):
    """``checkpoint-<step>/slmforge_adapter_manifest.json``: what an adapter can be loaded onto.

    Eval checks modules, vocab size and tokenizer hash against the base before loading
    and exits with code 3 on a mismatch; a 4-bit base is never merged into.
    """

    schema_version: Literal[1] = 1
    step: int
    base_model: str
    base_revision: str | None
    architecture: str  # config.json model_type (phi3 vs a llamafied repackage differ)
    target_modules: list[str]
    lora_r: int
    lora_alpha: int
    use_rslora: bool = False
    vocab_size: int
    tokenizer_sha256: str
    chat_template_sha256: str | None
    quantization: Literal["none", "4bit-nf4", "4bit-fp4"]
    backend: Literal["unsloth", "hf_peft"]
    slmforge_version: str


def prepare_run_dir(cfg: RootConfig, *, resume: bool, overwrite: bool) -> Path:
    """Create or validate ``runs/<name>/`` (refuse an existing dir unless ``resume`` or ``overwrite``)."""
    todo("M3", "prepare_run_dir")


def write_run_manifest(
    run_dir: Path,
    cfg: RootConfig,
    *,
    provenance: Provenance,
    base_revision: str | None,
    data_fingerprints: Mapping[str, str],
    contamination: Mapping[str, Any],
) -> Path:
    """Write ``run_manifest.json`` (a :class:`RunManifest`)."""
    todo("M3", "write_run_manifest")


def write_adapter_manifest(checkpoint_dir: Path, manifest: AdapterManifest) -> Path:
    """Write the adapter manifest next to the adapter weights and the saved tokenizer."""
    todo("M3", "write_adapter_manifest")


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any | None,
    cfg: RootConfig,
    run_dir: Path,
    *,
    callbacks: Sequence[Any],
) -> Any:
    """Plain ``transformers.Trainer`` for both backends, with ``data.loader.CompletionCollator``."""
    todo("M3", "build_trainer")


def train_in_process(cfg: RootConfig, run_dir: Path, backend: TrainBackend, *, resume: bool) -> TrainResult:
    """Seed, load + LoRA, data, train and save; shared by the hf_peft path and the Unsloth worker."""
    todo("M3", "train_in_process")


def finetune(
    cfg: RootConfig,
    *,
    dry_run: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    skip_eval: bool = False,
) -> TrainResult:
    """Backend select -> run dir + manifest -> train (in process or worker) -> eval-run (posthoc).

    ``dry_run`` validates the config, the backend choice and the training data
    (including the contamination guard) and prints the plan without loading a model.
    """
    todo("M3", "finetune")
