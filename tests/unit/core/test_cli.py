"""CLI wiring: help, version, data validate, and stub exit codes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from slmforge import __version__
from slmforge.cli import EXIT_NOT_IMPLEMENTED, EXIT_VALIDATION, app

runner = CliRunner()
COMMANDS = (
    "finetune",
    "eval",
    "eval-run",
    "score",
    "rescore",
    "report",
    "data",
    "protocol",
    "review",
    "validate-detectors",
    "diagnose-tokenizer",
    "distill",
)


def test_help_lists_every_command() -> None:
    out = runner.invoke(app, ["--help"]).output
    for cmd in COMMANDS:
        assert cmd in out


def test_version() -> None:
    assert __version__ in runner.invoke(app, ["--version"]).output


def test_data_validate_sample(repo_root: Path) -> None:
    res = runner.invoke(app, ["data", "validate", str(repo_root / "data/eval/ground_truth.sample.jsonl")])
    assert res.exit_code == 0, res.output
    assert "ok:" in res.output


def test_data_validate_errors(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "bad-1", "prompt": "q", "language": "fr"}) + "\n", encoding="utf-8")
    res = runner.invoke(app, ["data", "validate", str(p)])
    assert res.exit_code == EXIT_VALIDATION
    assert "E03" in res.output


def test_stub_exits_with_not_implemented(tmp_path: Path, repo_root: Path) -> None:
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps({"item_id": "x", "response": "y"}) + "\n", encoding="utf-8")
    gt = repo_root / "data/eval/ground_truth.sample.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "slmforge.cli",
        "score",
        "--responses",
        str(responses),
        "--ground-truth",
        str(gt),
        "--out",
        str(tmp_path / "out"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == EXIT_NOT_IMPLEMENTED, res.stderr
    assert "[M" in res.stderr


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "slmforge.cli", *args], capture_output=True, text=True)


def test_exit_codes_for_bad_inputs(tmp_path: Path, repo_root: Path) -> None:
    from slmforge.cli import EXIT_USAGE

    gt = str(repo_root / "data/eval/ground_truth.sample.jsonl")
    bad_set = _run(
        "score", "--responses", gt, "--ground-truth", gt, "--out", str(tmp_path), "--set", "novalue"
    )
    assert bad_set.returncode == EXIT_USAGE, bad_set.stderr
    a_dir = _run("data", "validate", str(tmp_path))
    assert a_dir.returncode == EXIT_VALIDATION, a_dir.stderr
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("run: [unclosed\n", encoding="utf-8")
    assert _run("finetune", "-c", str(bad_yaml)).returncode == EXIT_VALIDATION
