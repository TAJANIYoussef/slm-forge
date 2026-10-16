"""Config schema and loader: overrides, env expansion, strictness, cross-section validators."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from slmforge.config.load import apply_overrides, dump_resolved, expand_env, load_eval_config, load_root_config
from slmforge.config.schema import DistillConfig, RootConfig

MINIMAL = {
    "run": {"name": "t"},
    "model": {"name_or_path": "tiny"},
    "data": {"train": [{"path": "data/train/sample_chat.jsonl"}]},
    "train": {"max_steps": 6, "save_steps": 2},
}


def write(tmp_path: Path, data: dict[str, object]) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_minimal_root_config(tmp_path: Path) -> None:
    cfg = load_root_config(write(tmp_path, MINIMAL))
    assert cfg.train.backend == "auto"
    assert cfg.lora.target_modules == "auto"


def test_overrides_parse_yaml_scalars() -> None:
    out = apply_overrides(MINIMAL, ["train.lr=1e-4", "train.gradient_checkpointing=false", "eval.split=dev"])
    assert out["train"]["lr"] == 1e-4
    assert out["train"]["gradient_checkpointing"] is False
    assert out["eval"] == {"split": "dev"}
    assert "lr" not in MINIMAL["train"]  # input untouched


def test_bad_override() -> None:
    with pytest.raises(ValueError, match="dotted.key=value"):
        apply_overrides({}, ["novalue"])


def test_env_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINY_MODEL_DIR", "/models/tiny")
    assert expand_env({"a": ["${TINY_MODEL_DIR}/x"]}) == {"a": ["/models/tiny/x"]}
    monkeypatch.delenv("TINY_MODEL_DIR")
    with pytest.raises(ValueError, match="TINY_MODEL_DIR"):
        expand_env("${TINY_MODEL_DIR}")


def test_typos_are_errors(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_root_config(write(tmp_path, {**MINIMAL, "train": {"max_steps": 6, "learning_rate": 1e-4}}))


def test_steps_xor_epochs() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        RootConfig.model_validate({**MINIMAL, "train": {"max_steps": 6, "epochs": 1}})


def test_eval_schedule_multiple_of_save_steps() -> None:
    bad = {**MINIMAL, "eval": {"ground_truth": "gt.jsonl", "schedule": {"every_n_steps": 3}}}
    with pytest.raises(ValidationError, match="multiple of train.save_steps"):
        RootConfig.model_validate(bad)


def test_eval_config_from_run_config(tmp_path: Path) -> None:
    p = write(tmp_path, {**MINIMAL, "eval": {"ground_truth": "gt.jsonl", "split": "dev"}})
    assert load_eval_config(p).split == "dev"
    assert str(load_eval_config(None, ground_truth=Path("x.jsonl")).ground_truth) == "x.jsonl"


def test_distill_requires_both_opt_ins() -> None:
    base = {"provider": "anthropic", "model": "m", "input": "i", "output": "o", "prompt_template": "p"}
    assert DistillConfig.model_validate(base).enabled is False
    with pytest.raises(ValidationError, match="tos_acknowledged"):
        DistillConfig.model_validate({**base, "enabled": True, "tos_acknowledged": True})


def test_dump_resolved_roundtrip(tmp_path: Path) -> None:
    cfg = RootConfig.model_validate(MINIMAL)
    out = tmp_path / "resolved.yaml"
    dump_resolved(cfg, out)
    assert load_root_config(out) == cfg
