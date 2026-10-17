"""What to evaluate: checkpoint references, discovery in a run directory, selection and fingerprints.

A :class:`CheckpointRef` is one of:

* ``base``: an unmodified model loaded by id (``org/name[@revision]``) or the run's base
  model with adapters disabled (the step-0 reference of a run);
* ``adapter``: a PEFT adapter directory (``adapter_config.json``) on top of ``base_model``;
* ``full``: a full local model directory (``config.json``).

Identity is content, never a path: ``checkpoint-500`` exists in every run, so the
fingerprint (part of the generation-cache key) hashes adapter weights/config and the base
model id + revision (critique 17). Before an adapter is loaded on a base, M2 checks the
``slmforge_adapter_manifest.json`` written next to it (architecture, target modules,
vocab size, tokenizer hash, quantization) and refuses incompatible combinations
(:class:`AdapterIncompatible`, exit code 3).
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from slmforge._stub import todo
from slmforge.repro import canonical_json, sha256_text

if TYPE_CHECKING:
    from slmforge.train.lora import AdapterManifest

CheckpointKind = Literal["base", "adapter", "full"]

ADAPTER_CONFIG: Final = "adapter_config.json"
MODEL_CONFIG: Final = "config.json"
# Written by slmforge.train.lora (RUN_MANIFEST_NAME / ADAPTER_MANIFEST_NAME); eval does not import train.
RUN_MANIFEST: Final = "run_manifest.json"
ADAPTER_MANIFEST: Final = "slmforge_adapter_manifest.json"
CHECKPOINT_DIR_RE: Final = re.compile(r"^checkpoint-(\d+)$")
HUB_ID_RE: Final = re.compile(r"^(?P<org>[A-Za-z0-9][A-Za-z0-9_.-]*)/(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)$")
_REVISION_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]*$")


@dataclass(frozen=True)
class CheckpointRef:
    """Identity of something to evaluate (see the module docstring for the three kinds)."""

    name: str  # filesystem-safe: names the eval output directory
    path: Path | None  # None: load ``base_model`` by id
    step: int | None  # 0 for a run's base, N for checkpoint-N, None otherwise
    kind: CheckpointKind
    base_model: str
    base_revision: str | None = None

    @cached_property
    def fingerprint(self) -> str:
        """Content identity: kind + base id/revision, plus the checkpoint files for local dirs."""
        ident: dict[str, Any] = {
            "kind": self.kind,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
        }
        if self.path is not None and self.kind != "base":
            ident["files"] = checkpoint_fingerprint(self.path)
        return sha256_text(canonical_json(ident))


class AdapterIncompatible(OSError):
    """An adapter cannot be loaded on the requested base (modules, vocab, tokenizer); exit code 3."""


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _step_of(name: str) -> int | None:
    m = CHECKPOINT_DIR_RE.match(name)
    return int(m.group(1)) if m else None


def _hub_slug(model_id: str, revision: str | None) -> str:
    slug = model_id.replace("/", "--")
    return f"{slug}@{revision[:12].replace('/', '-')}" if revision else slug


def _resolve_dir(p: Path, *, base_model: str | None, revision: str | None) -> CheckpointRef:
    """A local adapter or full-model directory (named after its resolved directory name)."""
    name = p.resolve().name or p.as_posix()
    if (p / ADAPTER_CONFIG).is_file():
        cfg = _read_json(p / ADAPTER_CONFIG)
        base = base_model or cfg.get("base_model_name_or_path")
        if not base:
            raise ValueError(
                f"{p}: {ADAPTER_CONFIG} has no base_model_name_or_path; pass base_model (--base-model)"
            )
        rev = revision or cfg.get("revision") or None
        return CheckpointRef(
            name=name, path=p, step=_step_of(name), kind="adapter", base_model=str(base), base_revision=rev
        )
    if (p / MODEL_CONFIG).is_file():
        cfg = _read_json(p / MODEL_CONFIG)
        origin = base_model or cfg.get("_name_or_path") or p.as_posix()
        return CheckpointRef(
            name=name,
            path=p,
            step=_step_of(name),
            kind="full",
            base_model=str(origin),
            base_revision=revision,
        )
    raise ValueError(f"{p}: not a checkpoint (neither {ADAPTER_CONFIG} nor {MODEL_CONFIG})")


def resolve_checkpoint(
    ref: str | Path, *, base_model: str | None = None, revision: str | None = None
) -> CheckpointRef:
    """Local dir (``adapter_config.json`` -> adapter, ``config.json`` -> full) or hub ``org/name[@revision]``.

    For adapters the base comes from ``base_model`` or ``adapter_config.json`` and the base
    revision from ``revision`` or the adapter config. A string that looks like a path
    (``./x``, ``/x``, ``~/x``, several ``/``, or whose first component is a local directory)
    but does not exist raises FileNotFoundError instead of being read as a hub id.
    """
    if isinstance(ref, str) and not ref.strip():
        raise ValueError("empty checkpoint reference")
    p = Path(ref).expanduser()
    if p.is_dir():
        return _resolve_dir(p, base_model=base_model, revision=revision)
    if p.exists():
        raise ValueError(f"{p}: a checkpoint must be a directory or a hub id, not a file")
    if isinstance(ref, Path):
        raise FileNotFoundError(f"checkpoint directory not found: {p}")

    text = ref.strip()
    model_id, _, at_rev = text.partition("@")
    m = HUB_ID_RE.match(model_id)
    looks_local = (
        text.startswith((".", "/", "~", "\\")) or "\\" in text or (m is not None and Path(m["org"]).is_dir())
    )
    if m is None or looks_local:
        raise FileNotFoundError(
            f"checkpoint {ref!r} is neither a local directory nor a hub id 'org/name[@revision]'"
        )
    if at_rev and not _REVISION_RE.match(at_rev):
        raise ValueError(f"invalid revision in {ref!r}")
    if at_rev and revision and at_rev != revision:
        raise ValueError(f"conflicting revisions for {model_id}: {at_rev!r} in the id, {revision!r} passed")
    if base_model is not None:
        raise ValueError(
            f"base_model applies to local adapter directories only; download the adapter {model_id!r} "
            "and pass its directory"
        )
    rev = at_rev or revision or None
    return CheckpointRef(
        name=_hub_slug(model_id, rev),
        path=None,
        step=None,
        kind="base",
        base_model=model_id,
        base_revision=rev,
    )


def _manifest_base(manifest: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Base model and revision from a run manifest (top-level, ``model`` or ``config.model``)."""
    base, rev = manifest.get("base_model"), manifest.get("base_revision")
    config = manifest.get("config")
    for section in (manifest.get("model"), config.get("model") if isinstance(config, Mapping) else None):
        if isinstance(section, Mapping):
            base = base or section.get("name_or_path")
            rev = rev or section.get("revision")
    return (str(base) if base else None), (str(rev) if rev else None)


def discover_checkpoints(run_dir: Path, *, include_base: bool = True) -> list[CheckpointRef]:
    """``checkpoint-<step>/`` dirs sorted by step (base first); base and revision from ``run_manifest.json``.

    Without a manifest the base comes from the first adapter's ``adapter_config.json``.
    Directories that are not (yet) complete checkpoints are skipped.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    manifest_path = run_dir / RUN_MANIFEST
    base, base_rev = _manifest_base(_read_json(manifest_path)) if manifest_path.is_file() else (None, None)

    found: list[tuple[int, Path]] = []
    for d in run_dir.iterdir():
        step = _step_of(d.name)
        if (
            step is not None
            and d.is_dir()
            and ((d / ADAPTER_CONFIG).is_file() or (d / MODEL_CONFIG).is_file())
        ):
            found.append((step, d))
    ckpts = [_resolve_dir(d, base_model=base, revision=base_rev) for _, d in sorted(found)]

    if base is None:
        adapters = [c for c in ckpts if c.kind == "adapter"]
        if adapters:
            base, base_rev = adapters[0].base_model, adapters[0].base_revision
    if not include_base:
        return ckpts
    if base is None:
        raise ValueError(
            f"{run_dir}: cannot determine the base model (no {RUN_MANIFEST} and no adapter checkpoint)"
        )
    return [
        CheckpointRef(name="base", path=None, step=0, kind="base", base_model=base, base_revision=base_rev),
        *ckpts,
    ]


def select_checkpoints(
    ckpts: Sequence[CheckpointRef], *, every_n_steps: int | None, always: Collection[str] = ("base", "final")
) -> list[CheckpointRef]:
    """Keep ``step % every_n_steps == 0`` (step > 0) plus ``always``: "base", "final" (highest step) or names.

    ``every_n_steps=None`` keeps every checkpoint. Input order is preserved.
    """
    if every_n_steps is not None and every_n_steps <= 0:
        raise ValueError("every_n_steps must be positive")
    stepped = [c for c in ckpts if c.kind != "base" and c.step is not None]
    final = max(stepped, key=lambda c: c.step or 0) if stepped else None
    out: list[CheckpointRef] = []
    for c in ckpts:
        keep = (
            every_n_steps is None
            or (c.kind != "base" and c.step is not None and c.step > 0 and c.step % every_n_steps == 0)
            or ("base" in always and c.kind == "base")
            or ("final" in always and c is final)
            or c.name in always
        )
        if keep and c not in out:
            out.append(c)
    return out


def checkpoint_fingerprint(path: Path) -> str:
    """Adapter: config + weights sha; full model: config.json + safetensors headers (no full hash)."""
    todo("M2", "checkpoint_fingerprint")


def read_adapter_manifest(path: Path) -> AdapterManifest | None:
    """The ``train.lora.AdapterManifest`` of a checkpoint dir, or None for adapters trained elsewhere."""
    todo("M2", "read_adapter_manifest")


def check_adapter_compatibility(
    ckpt: CheckpointRef, *, base_config: Mapping[str, Any], tokenizer_sha256: str | None
) -> list[str]:
    """Problems loading ``ckpt`` on a base with this config/tokenizer (modules, vocab, architecture)."""
    todo("M2", "check_adapter_compatibility")
