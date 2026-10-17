"""Reproducibility: canonical hashing, provenance, seeding, RNG preservation and the offline guard.

Everything here is torch-free at import. Functions that touch torch (seeding, RNG state,
CUDA/GPU provenance) only do so when ``torch`` is **already** in ``sys.modules``; they never
import it, so calling them from the scoring core or the CLI cannot load the ML stack.

Hashes are hex SHA-256 over :func:`canonical_json`, so the same value always hashes the
same way regardless of dict order or path separators. ``Decimal`` keeps its exact
representation (``1.50`` and ``1.5`` differ); data models that need value equality
canonicalize first (``eval.dataset.canonical_decimal``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import Enum
from fnmatch import fnmatch
from importlib import metadata
from pathlib import Path, PurePath
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

import slmforge

NON_SEMANTIC_FIELDS: Final[frozenset[str]] = frozenset(
    {"run.name", "run.output_dir", "train.logging_steps", "eval", "report", "distill"}
)
"""Dotted config paths that do not change the trained weights; excluded from :func:`config_sha256`.

Eval identity lives in the protocol (``slmforge.protocol``), not in the training config hash.
"""

TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "tokenizers",
    "safetensors",
    "bitsandbytes",
    "unsloth",
    "unsloth_zoo",
    "vllm",
    "numpy",
    "pydantic",
    "rapidfuzz",
    "sacrebleu",
)
"""Distributions whose versions are recorded in every manifest and ``metrics.json`` (None if absent)."""

OFFLINE_ENV_VARS: Final[tuple[str, ...]] = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
CUBLAS_WORKSPACE: Final = ":4096:8"
DEFAULT_DIR_PATTERNS: Final[tuple[str, ...]] = (
    "*.safetensors",
    "adapter_config.json",
    "config.json",
    "tokenizer*.json",
    "*.jinja",
)
_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})


# --------------------------------------------------------------------------- canonical hashing


def _jsonable(obj: Any) -> Any:
    """Convert ``obj`` to plain JSON types with the canonical conversions."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"canonical_json: non-finite float {obj!r} has no canonical form")
        return obj
    if isinstance(obj, Enum):
        return _jsonable(obj.value)
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, BaseModel):
        return _jsonable(obj.model_dump(mode="json"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = _jsonable(k)
            if not isinstance(key, str):
                key = json.dumps(key, ensure_ascii=False)  # 1 -> "1", True -> "true", None -> "null"
            if key in out:
                raise ValueError(f"canonical_json: two keys map to {key!r}")
            out[key] = _jsonable(v)
        return out
    if isinstance(obj, (set, frozenset)):
        items = [_jsonable(x) for x in obj]
        try:
            return sorted(items)
        except TypeError:  # mixed or unorderable element types: order by canonical text
            return sorted(items, key=canonical_json)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).hex()
    raise TypeError(f"canonical_json: unsupported type {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Sorted keys, ``(',', ':')`` separators, ``ensure_ascii=False``; Decimal->str, Path->posix, date->ISO.

    Sets become sorted lists, tuples lists, enums their value, pydantic models and dataclasses
    their fields. Non-finite floats and unknown types raise, because they have no stable form.
    """
    return json.dumps(
        _jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_text(s: str) -> str:
    """Hex SHA-256 of UTF-8 text."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: Path, patterns: Sequence[str] = DEFAULT_DIR_PATTERNS) -> str:
    """Order-independent hash of the files under ``path`` whose name matches any pattern.

    The digest covers each matching file's relative POSIX path and content hash, sorted by
    path, so it does not depend on filesystem enumeration order but does change when a
    file is renamed, added, removed or edited. A directory with no matching file hashes to
    the digest of an empty listing.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"sha256_dir: {root} is not a directory")
    rows = sorted(
        f"{p.relative_to(root).as_posix()}\0{sha256_file(p)}"
        for p in root.rglob("*")
        if p.is_file() and any(fnmatch(p.name, pat) for pat in patterns)
    )
    return sha256_text("\n".join(rows))


def _drop_dotted(d: dict[str, Any], dotted: str) -> None:
    node: Any = d
    *parents, leaf = dotted.split(".")
    for part in parents:
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return
    if isinstance(node, dict):
        node.pop(leaf, None)


def config_sha256(
    cfg: BaseModel | Mapping[str, Any], *, exclude: Collection[str] = NON_SEMANTIC_FIELDS
) -> str:
    """``sha256(canonical_json(model_dump(mode='json') minus the excluded dotted paths)))``.

    Hash the config *after* ``--set`` overrides and env expansion, i.e. the resolved model.
    """
    data = cfg.model_dump(mode="json") if isinstance(cfg, BaseModel) else json.loads(canonical_json(cfg))
    for dotted in exclude:
        _drop_dotted(data, dotted)
    return sha256_text(canonical_json(data))


# --------------------------------------------------------------------------- provenance


def _run(cmd: Sequence[str], cwd: Path | None = None, timeout: float = 5.0) -> str | None:
    """stdout of a command, or None on any failure (missing binary, non-zero exit, timeout)."""
    try:
        out = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_info(cwd: Path | None = None) -> dict[str, str | bool | None]:
    """``{"commit": sha, "dirty": bool}`` of the repo containing ``cwd``; None values outside git.

    ``cwd`` defaults to the installed ``slmforge`` package directory, i.e. the code version.
    Dirty means tracked files differ from HEAD (untracked files are ignored).
    """
    where = Path(cwd) if cwd is not None else Path(slmforge.__file__).resolve().parent
    commit = _run(["git", "rev-parse", "HEAD"], cwd=where)
    if commit is None or not commit.strip():
        return {"commit": None, "dirty": None}
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=where)
    return {"commit": commit.strip(), "dirty": None if status is None else bool(status.strip())}


class Provenance(BaseModel):
    """Environment and model identity recorded in every manifest and ``metrics.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slmforge_version: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    python: str
    platform: str
    packages: dict[str, str | None]
    cuda: str | None = None
    cudnn: str | None = None
    gpu: str | None = None
    driver: str | None = None
    torch_num_threads: int | None = None
    deterministic_algorithms: bool = False
    cublas_workspace_config: str | None = None
    pythonhashseed: str | None = None
    hf_hub_offline: bool = False
    timestamp_utc: str
    notes: list[str] = []


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed distribution versions via ``importlib.metadata`` (None when not installed)."""
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
        except Exception:  # broken metadata must never break provenance
            out[name] = None
    return out


def _torch_details(notes: list[str]) -> dict[str, Any]:
    """CUDA/cuDNN/GPU/determinism details, only if torch is already imported."""
    torch = sys.modules.get("torch")
    if torch is None:
        notes.append("torch not loaded in this process: cuda/cudnn/gpu/determinism not collected")
        return {}
    info: dict[str, Any] = {}
    try:
        info["cuda"] = getattr(torch.version, "cuda", None)
        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["cudnn"] = None if cudnn is None else str(cudnn)
        info["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
        info["torch_num_threads"] = int(torch.get_num_threads())
        if torch.cuda.is_available():
            info["gpu"] = str(torch.cuda.get_device_name(0))
            driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
            info["driver"] = driver.strip().splitlines()[0] if driver and driver.strip() else None
    except Exception as e:  # fail-soft: provenance is best effort
        notes.append(f"torch details incomplete: {type(e).__name__}: {e}")
    return info


def collect_provenance(*, cwd: Path | None = None) -> Provenance:
    """Fail-soft collection via ``importlib.metadata``, git and torch (only if already imported)."""
    notes: list[str] = []
    git = git_info(cwd)
    commit, dirty = git["commit"], git["dirty"]
    return Provenance(
        slmforge_version=slmforge.__version__,
        git_commit=commit if isinstance(commit, str) else None,
        git_dirty=dirty if isinstance(dirty, bool) else None,
        python=platform.python_version(),
        platform=platform.platform(),
        packages=package_versions(),
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        hf_hub_offline=os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _TRUTHY,
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        **_torch_details(notes),
        notes=notes,
    )


# --------------------------------------------------------------------------- environment


@contextmanager
def offline_mode(enabled: bool = True) -> Iterator[None]:
    """Set ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE``/``HF_DATASETS_OFFLINE`` to 1; restore on exit.

    Enter it **before** importing huggingface_hub/transformers: they read these variables at
    import time. ``enabled=False`` leaves the environment untouched.
    """
    if not enabled:
        yield
        return
    saved = {k: os.environ.get(k) for k in OFFLINE_ENV_VARS}
    try:
        for k in OFFLINE_ENV_VARS:
            os.environ[k] = "1"
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def set_determinism(seed: int, *, strict: bool = False) -> list[str]:
    """Seed python and numpy always; torch/CUDA (+ deterministic algorithms) only if torch is loaded.

    Also sets ``CUBLAS_WORKSPACE_CONFIG`` (if unset) so a later CUDA init is deterministic.
    Returns notes on what was seeded or skipped; call again after importing torch.
    """
    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    notes = ["python", "numpy"]
    torch = sys.modules.get("torch")
    if torch is None:
        notes.append("torch not loaded: not seeded (call set_determinism again after importing torch)")
        return notes
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        notes.append("cuda")
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    notes.append(f"torch (deterministic algorithms, warn_only={not strict})")
    return notes


@contextmanager
def preserved_rng_state() -> Iterator[None]:
    """Restore the python/numpy (and torch/CUDA, if loaded) RNG streams on exit.

    Used around inline eval so the training loss sequence is unchanged by evaluating.
    """
    import numpy as np

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch = sys.modules.get("torch")
    torch_state = torch.get_rng_state() if torch is not None else None
    cuda_state = None
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_state = torch.cuda.get_rng_state_all()
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
