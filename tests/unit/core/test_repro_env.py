"""repro: git info, provenance, offline mode, seeding and RNG preservation (all torch-free)."""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from slmforge import __version__, repro
from slmforge.repro import (
    OFFLINE_ENV_VARS,
    TRACKED_PACKAGES,
    Provenance,
    collect_provenance,
    git_info,
    offline_mode,
    preserved_rng_state,
    set_determinism,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def test_git_info_outside_a_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_info(tmp_path) == {"commit": None, "dirty": None}


@needs_git
def test_git_info_in_a_repo(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("a", encoding="utf-8")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    info = git_info(tmp_path)
    assert isinstance(info["commit"], str) and len(info["commit"]) == 40
    assert info["dirty"] is False
    (tmp_path / "untracked.txt").write_text("u", encoding="utf-8")
    assert git_info(tmp_path)["dirty"] is False  # untracked files do not count
    (tmp_path / "f.txt").write_text("b", encoding="utf-8")
    assert git_info(tmp_path)["dirty"] is True


def test_git_info_without_git_binary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(repro.subprocess, "run", missing)
    assert git_info(tmp_path) == {"commit": None, "dirty": None}


def test_collect_provenance_is_fail_soft_and_serializable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    prov = collect_provenance()
    assert isinstance(prov, Provenance)
    assert prov.slmforge_version == __version__
    assert set(TRACKED_PACKAGES) <= set(prov.packages)
    assert prov.packages["numpy"] == np.__version__
    assert prov.packages["pydantic"] is not None
    assert prov.hf_hub_offline is True
    assert prov.python.count(".") == 2
    assert Provenance.model_validate_json(prov.model_dump_json()) == prov


def test_collect_provenance_without_torch_notes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    prov = collect_provenance()
    assert prov.cuda is None and prov.gpu is None
    assert any("torch not loaded" in n for n in prov.notes)


def test_provenance_and_determinism_never_import_torch() -> None:
    code = (
        "import sys\n"
        "from slmforge.repro import collect_provenance, preserved_rng_state, set_determinism\n"
        "collect_provenance()\nset_determinism(1)\n"
        "with preserved_rng_state():\n    pass\n"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_offline_mode_sets_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "keep")
    with offline_mode():
        assert all(os.environ[k] == "1" for k in OFFLINE_ENV_VARS)
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert os.environ["HF_DATASETS_OFFLINE"] == "keep"


def test_offline_mode_restores_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(RuntimeError), offline_mode():
        raise RuntimeError("boom")
    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_mode_disabled_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with offline_mode(enabled=False):
        assert "HF_HUB_OFFLINE" not in os.environ


def test_set_determinism_reproduces_python_and_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_determinism(42)
    a = (random.random(), np.random.rand(3).tolist())
    notes = set_determinism(42)
    b = (random.random(), np.random.rand(3).tolist())
    assert a == b
    assert notes[:2] == ["python", "numpy"]
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    if "torch" not in sys.modules:
        assert any("torch not loaded" in n for n in notes)


def test_set_determinism_keeps_existing_cublas_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    set_determinism(0)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    set_determinism(-1)  # negative seeds are accepted


def test_preserved_rng_state_restores_streams() -> None:
    random.seed(7)
    np.random.seed(7)
    expected = (random.random(), np.random.rand())
    random.seed(7)
    np.random.seed(7)
    with preserved_rng_state():
        random.random()
        np.random.rand(10)
    assert (random.random(), np.random.rand()) == expected


def test_preserved_rng_state_restores_on_error() -> None:
    random.seed(3)
    expected = random.random()
    random.seed(3)
    with pytest.raises(KeyError), preserved_rng_state():
        random.random()
        raise KeyError("x")
    assert random.random() == expected
