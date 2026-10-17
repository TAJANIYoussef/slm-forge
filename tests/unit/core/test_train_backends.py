"""Backend selection decides from find_spec alone and never imports torch or unsloth here."""

from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Callable

import pytest

from slmforge.train import backends
from slmforge.train.backends import (
    BackendUnavailable,
    TrainBackend,
    choose_backend,
    module_available,
    select_backend,
)

ALL = frozenset({"torch", "transformers", "peft", "bitsandbytes", "unsloth"})
HF = frozenset({"torch", "transformers", "peft"})


def installed(mods: frozenset[str]) -> Callable[[str], bool]:
    return lambda name: name in mods


def test_explicit_hf_peft_does_not_probe_cuda() -> None:
    choice = choose_backend("hf_peft", has_module=installed(HF))
    assert choice.name == "hf_peft"
    assert choice.cuda is None


def test_auto_prefers_unsloth_on_cuda() -> None:
    choice = choose_backend("auto", cuda=True, has_module=installed(ALL))
    assert (choice.name, choice.cuda) == ("unsloth", True)


def test_auto_falls_back_without_cuda() -> None:
    choice = choose_backend("auto", cuda=False, has_module=installed(ALL))
    assert choice.name == "hf_peft"
    assert "no CUDA" in choice.reason


def test_auto_falls_back_without_unsloth() -> None:
    choice = choose_backend("auto", cuda=True, has_module=installed(HF))
    assert choice.name == "hf_peft"
    assert "unsloth not installed" in choice.reason


def test_unsloth_without_cuda_is_an_environment_error() -> None:
    with pytest.raises(EnvironmentError, match=r"CUDA.*exit code 3") as e:
        choose_backend("unsloth", cuda=False, has_module=installed(ALL))
    assert isinstance(e.value, BackendUnavailable)
    assert isinstance(e.value, OSError)  # the CLI maps OSError to exit code 3


def test_unsloth_not_installed() -> None:
    with pytest.raises(BackendUnavailable, match=r"slm-forge\[unsloth\]"):
        choose_backend("unsloth", cuda=True, has_module=installed(HF))


def test_hf_peft_needs_the_hf_extra() -> None:
    with pytest.raises(BackendUnavailable, match=r"torch, transformers, peft.*slm-forge\[hf\]"):
        choose_backend("hf_peft", has_module=installed(frozenset()))


def test_qlora_without_cuda_never_downgrades() -> None:
    with pytest.raises(BackendUnavailable, match=r"load_in_4bit.*CUDA"):
        choose_backend("hf_peft", load_in_4bit=True, cuda=False, has_module=installed(ALL))
    with pytest.raises(BackendUnavailable, match="bitsandbytes"):
        choose_backend("auto", load_in_4bit=True, cuda=False, has_module=installed(HF))
    ok = choose_backend("hf_peft", load_in_4bit=True, cuda=True, has_module=installed(ALL))
    assert (ok.name, ok.cuda) == ("hf_peft", True)


def test_cuda_probe_is_lazy_and_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def probe() -> bool:
        calls.append(1)
        return False

    monkeypatch.setattr(backends, "cuda_available", probe)
    assert choose_backend("auto", has_module=installed(HF)).cuda is None  # no unsloth: no probe
    assert calls == []
    with pytest.raises(BackendUnavailable, match="load_in_4bit"):
        choose_backend("auto", load_in_4bit=True, has_module=installed(ALL))
    assert calls == [1]  # probed once for the unsloth decision, reused for the 4-bit check


def test_unknown_request() -> None:
    with pytest.raises(ValueError, match=r"train\.backend"):
        choose_backend("deepspeed", has_module=installed(ALL))  # type: ignore[arg-type]


def test_cuda_available_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "module_available", lambda name: False)
    assert backends.cuda_available() is False


def test_module_available() -> None:
    assert module_available("json")
    assert not module_available("slmforge_no_such_module")


def test_select_backend_returns_lazy_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "module_available", installed(HF))
    hf = select_backend("hf_peft")
    assert hf.name == "hf_peft"
    assert isinstance(hf, TrainBackend)
    monkeypatch.setattr(backends, "module_available", installed(ALL))
    monkeypatch.setattr(backends, "cuda_available", lambda: True)
    us = select_backend("auto")
    assert us.name == "unsloth"
    assert isinstance(us, TrainBackend)
    assert "unsloth" not in sys.modules  # instantiating the backend imports nothing


def test_backend_modules_import_nothing_heavy() -> None:
    code = (
        "import sys\n"
        "import slmforge.train.backends.hf_peft, slmforge.train.backends.unsloth, slmforge.train.worker\n"
        "import slmforge.train.lora, slmforge.train.callbacks\n"
        "print(','.join(m for m in ('torch', 'transformers', 'peft', 'unsloth') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_unsloth_module_never_imports_unsloth() -> None:
    importlib.import_module("slmforge.train.backends.unsloth")
    assert "unsloth" not in sys.modules
