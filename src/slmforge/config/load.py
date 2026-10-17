"""YAML loading with ``--set dotted.key=value`` overrides and ``${VAR}`` expansion."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from slmforge.config.schema import EvalConfig, RootConfig

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SCI_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def load_yaml(path: Path) -> dict[str, Any]:
    """Safe-load a YAML mapping (an empty file gives ``{}``)."""
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides in order; values are parsed as YAML scalars."""
    out: dict[str, Any] = _deepcopy(data)
    for ov in overrides:
        key, sep, raw = ov.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like dotted.key=value, got {ov!r}")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            if not isinstance(child, dict):
                raise ValueError(f"override {ov!r}: {part!r} is not a mapping")
            node = child
        node[parts[-1]] = _parse_scalar(raw)
    return out


def _parse_scalar(raw: str) -> Any:
    """YAML scalar parsing, plus floats like ``1e-4`` that YAML 1.1 reads as strings."""
    value = yaml.safe_load(raw)
    if isinstance(value, str) and _SCI_RE.fullmatch(value):
        return float(value)
    return value


def expand_env(data: Any) -> Any:
    """Expand ``${VAR}`` in every string; an unset variable is an error."""
    if isinstance(data, dict):
        return {k: expand_env(v) for k, v in data.items()}
    if isinstance(data, list):
        return [expand_env(v) for v in data]
    if isinstance(data, str):

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ValueError(f"environment variable {name} is not set (used as ${{{name}}})")
            return os.environ[name]

        return _ENV_RE.sub(sub, data)
    return data


def load_root_config(path: Path, overrides: Sequence[str] = ()) -> RootConfig:
    """Load a run config: YAML -> overrides -> env expansion -> validation."""
    return RootConfig.model_validate(expand_env(apply_overrides(load_yaml(path), overrides)))


def load_eval_config(
    path: Path | None, overrides: Sequence[str] = (), *, ground_truth: Path | None = None
) -> EvalConfig:
    """Load a standalone eval config, or the ``eval:`` section of a run config.

    With ``path=None`` the defaults are used and ``ground_truth`` must be given.
    """
    data: dict[str, Any] = load_yaml(path) if path is not None else {}
    if "run" in data and "model" in data:  # a run config: use its eval section
        data = data.get("eval") or {}
    if ground_truth is not None:
        data["ground_truth"] = str(ground_truth)
    return EvalConfig.model_validate(expand_env(apply_overrides(data, overrides)))


def dump_resolved(cfg: BaseModel, path: Path) -> None:
    """Write the fully resolved config (defaults included) as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)


def _deepcopy(d: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _deepcopy(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)
        for k, v in d.items()
    }
