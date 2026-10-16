"""The scoring core and the CLI must not import heavy ML libraries (torch-free install)."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("torch", "transformers", "peft", "unsloth", "bitsandbytes", "vllm", "scipy", "sklearn")


def _loaded_after(code: str) -> list[str]:
    probe = f"import sys\n{code}\nprint(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    return [m for m in last.split(",") if m in HEAVY]


def test_every_module_is_torch_free_at_import() -> None:
    code = (
        "import importlib, pkgutil, slmforge\n"
        "for m in pkgutil.walk_packages(slmforge.__path__, 'slmforge.'):\n"
        "    importlib.import_module(m.name)"
    )
    assert _loaded_after(code) == []


def test_cli_help_is_torch_free() -> None:
    code = (
        "import sys\nfrom slmforge import cli\nsys.argv = ['slm-forge', '--help']\n"
        "try:\n    cli.app()\nexcept SystemExit:\n    pass"
    )
    assert _loaded_after(code) == []
