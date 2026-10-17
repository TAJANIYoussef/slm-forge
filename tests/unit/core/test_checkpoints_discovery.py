"""checkpoints: resolving references, discovering a run's checkpoints and selecting the schedule."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slmforge.eval.checkpoints import (
    CheckpointRef,
    discover_checkpoints,
    resolve_checkpoint,
    select_checkpoints,
)


def _adapter(d: Path, base: str | None = "org/base", revision: str | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {"peft_type": "LORA", "r": 16}
    if base is not None:
        cfg["base_model_name_or_path"] = base
    if revision is not None:
        cfg["revision"] = revision
    (d / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


def _full(d: Path, name_or_path: str | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, Any] = {"model_type": "llama"}
    if name_or_path:
        cfg["_name_or_path"] = name_or_path
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d


# --------------------------------------------------------------------------- resolve_checkpoint


def test_resolve_adapter_dir(tmp_path: Path) -> None:
    d = _adapter(tmp_path / "checkpoint-500", revision="abc123")
    ref = resolve_checkpoint(d)
    assert (ref.kind, ref.name, ref.step, ref.path) == ("adapter", "checkpoint-500", 500, d)
    assert (ref.base_model, ref.base_revision) == ("org/base", "abc123")
    assert resolve_checkpoint(str(d)) == ref


def test_resolve_adapter_overrides(tmp_path: Path) -> None:
    d = _adapter(tmp_path / "my-adapter")
    ref = resolve_checkpoint(d, base_model="org/other", revision="r2")
    assert (ref.base_model, ref.base_revision, ref.step) == ("org/other", "r2", None)


def test_resolve_adapter_without_base_needs_override(tmp_path: Path) -> None:
    d = _adapter(tmp_path / "a", base=None)
    with pytest.raises(ValueError, match="base_model"):
        resolve_checkpoint(d)
    assert resolve_checkpoint(d, base_model="org/base").base_model == "org/base"


def test_resolve_full_model_dir(tmp_path: Path) -> None:
    ref = resolve_checkpoint(_full(tmp_path / "merged", "org/origin"))
    assert (ref.kind, ref.base_model, ref.name) == ("full", "org/origin", "merged")
    bare = resolve_checkpoint(_full(tmp_path / "bare"))
    assert bare.base_model == (tmp_path / "bare").as_posix()


def test_adapter_config_wins_over_config_json(tmp_path: Path) -> None:
    d = _adapter(tmp_path / "both")
    _full(d)
    assert resolve_checkpoint(d).kind == "adapter"


def test_resolve_dir_without_configs(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="not a checkpoint"):
        resolve_checkpoint(tmp_path / "empty")
    (tmp_path / "file.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="directory"):
        resolve_checkpoint(tmp_path / "file.bin")


@pytest.mark.parametrize(
    ("ref", "revision", "name", "rev"),
    [
        ("Qwen/Qwen2.5-7B-Instruct", None, "Qwen--Qwen2.5-7B-Instruct", None),
        (
            "HuggingFaceTB/SmolLM2-1.7B@0123456789abcdef",
            None,
            "HuggingFaceTB--SmolLM2-1.7B@0123456789abcdef",
            "0123456789abcdef",
        ),
        ("org/name", "main", "org--name@main", "main"),
        ("org/name@v1", "v1", "org--name@v1", "v1"),
    ],
)
def test_resolve_hub_id(
    ref: str,
    revision: str | None,
    name: str,
    rev: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    c = resolve_checkpoint(ref, revision=revision)
    assert (c.kind, c.path, c.step, c.name, c.base_revision) == ("base", None, None, name, rev)
    assert c.base_model == ref.partition("@")[0]
    assert "/" not in c.name


def test_hub_id_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="conflicting"):
        resolve_checkpoint("org/name@a", revision="b")
    with pytest.raises(ValueError, match="adapter"):
        resolve_checkpoint("org/name", base_model="org/base")


@pytest.mark.parametrize("ref", ["./missing", "/abs/missing/dir", "~/nope/x", "a/b/c", "plainname", "a\\b"])
def test_missing_local_paths_are_not_hub_ids(
    ref: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(ref)


def test_empty_ref_and_dot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="empty"):
        resolve_checkpoint("  ")
    _adapter(tmp_path / "checkpoint-9")
    monkeypatch.chdir(tmp_path / "checkpoint-9")
    ref = resolve_checkpoint(".")
    assert (ref.name, ref.step, ref.kind) == ("checkpoint-9", 9, "adapter")


def test_missing_path_under_existing_dir_is_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("runs/checkpoint-5")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(Path("org/name"))  # a Path is always local


def test_base_fingerprint_is_content_identity() -> None:
    a = CheckpointRef(name="base", path=None, step=0, kind="base", base_model="org/m", base_revision="r1")
    b = CheckpointRef(
        name="other-name", path=None, step=None, kind="base", base_model="org/m", base_revision="r1"
    )
    c = CheckpointRef(name="base", path=None, step=0, kind="base", base_model="org/m", base_revision="r2")
    assert a.fingerprint == b.fingerprint  # the name/step are not identity
    assert a.fingerprint != c.fingerprint
    assert len(a.fingerprint) == 64


# --------------------------------------------------------------------------- discover / select


def _run(tmp_path: Path, steps: list[int], manifest: dict[str, Any] | None = None) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    for s in steps:
        _adapter(run / f"checkpoint-{s}", base="org/from-adapter", revision="adapter-rev")
    if manifest is not None:
        (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_discover_sorts_numerically_and_reads_manifest(tmp_path: Path) -> None:
    run = _run(tmp_path, [10, 2, 100], {"base_model": "org/base", "base_revision": "sha1"})
    (run / "checkpoint-7").mkdir()  # incomplete: no config yet
    (run / "checkpoint-x").mkdir()
    _adapter(run / "other")
    (run / "checkpoint-3.tmp").mkdir()
    ckpts = discover_checkpoints(run)
    assert [c.name for c in ckpts] == ["base", "checkpoint-2", "checkpoint-10", "checkpoint-100"]
    assert [c.step for c in ckpts] == [0, 2, 10, 100]
    assert all((c.base_model, c.base_revision) == ("org/base", "sha1") for c in ckpts)
    assert ckpts[0].kind == "base" and ckpts[0].path is None


@pytest.mark.parametrize(
    "manifest",
    [
        {"model": {"name_or_path": "org/base", "revision": "sha1"}},
        {"config": {"model": {"name_or_path": "org/base", "revision": "sha1"}}},
    ],
)
def test_discover_manifest_shapes(tmp_path: Path, manifest: dict[str, Any]) -> None:
    base = discover_checkpoints(_run(tmp_path, [1], manifest))[0]
    assert (base.base_model, base.base_revision) == ("org/base", "sha1")


def test_discover_without_manifest_uses_adapter_config(tmp_path: Path) -> None:
    ckpts = discover_checkpoints(_run(tmp_path, [4, 2]))
    assert (ckpts[0].base_model, ckpts[0].base_revision) == ("org/from-adapter", "adapter-rev")
    assert discover_checkpoints(tmp_path / "run", include_base=False)[0].name == "checkpoint-2"


def test_discover_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_checkpoints(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert discover_checkpoints(empty, include_base=False) == []
    with pytest.raises(ValueError, match="base model"):
        discover_checkpoints(empty)


@pytest.fixture
def ckpts(tmp_path: Path) -> list[CheckpointRef]:
    return discover_checkpoints(_run(tmp_path, [5, 10, 15, 20, 25], {"base_model": "org/base"}))


def _names(cs: list[CheckpointRef]) -> list[str]:
    return [c.name for c in cs]


def test_select_every_n_plus_base_and_final(ckpts: list[CheckpointRef]) -> None:
    assert _names(select_checkpoints(ckpts, every_n_steps=10)) == [
        "base",
        "checkpoint-10",
        "checkpoint-20",
        "checkpoint-25",
    ]


def test_select_none_keeps_everything(ckpts: list[CheckpointRef]) -> None:
    assert select_checkpoints(ckpts, every_n_steps=None) == ckpts


def test_select_always_controls_endpoints(ckpts: list[CheckpointRef]) -> None:
    assert _names(select_checkpoints(ckpts, every_n_steps=10, always=())) == [
        "checkpoint-10",
        "checkpoint-20",
    ]
    assert _names(select_checkpoints(ckpts, every_n_steps=10, always=("final",))) == [
        "checkpoint-10",
        "checkpoint-20",
        "checkpoint-25",
    ]
    assert _names(select_checkpoints(ckpts, every_n_steps=20, always=("checkpoint-5",))) == [
        "checkpoint-5",
        "checkpoint-20",
    ]
    with pytest.raises(ValueError):
        select_checkpoints(ckpts, every_n_steps=0)
