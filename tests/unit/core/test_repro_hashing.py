"""repro: canonical JSON, file/dir hashes and the training-config hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any

import pytest
from pydantic import BaseModel

from slmforge.config.schema import RootConfig
from slmforge.repro import (
    NON_SEMANTIC_FIELDS,
    canonical_json,
    config_sha256,
    sha256_dir,
    sha256_file,
    sha256_text,
)


class Color(Enum):
    RED = "red"


class Inner(BaseModel):
    x: int
    p: Path


@dataclass(frozen=True)
class DC:
    a: int
    b: tuple[str, ...]


def test_canonical_json_is_order_independent_and_compact() -> None:
    a = canonical_json({"b": 1, "a": [1, 2], "c": {"z": None, "y": True}})
    b = canonical_json({"c": {"y": True, "z": None}, "a": [1, 2], "b": 1})
    assert a == b == '{"a":[1,2],"b":1,"c":{"y":true,"z":null}}'


def test_canonical_json_keeps_unicode() -> None:
    assert canonical_json({"q": "الجواب: 70٪ réponse"}) == '{"q":"الجواب: 70٪ réponse"}'


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("1.50"), '"1.50"'),
        (Path("a") / "b.txt", '"a/b.txt"'),
        (PureWindowsPath("C:\\x\\y"), '"C:/x/y"'),
        (date(2024, 2, 29), '"2024-02-29"'),
        (datetime(2024, 1, 2, 3, 4, 5), '"2024-01-02T03:04:05"'),
        ({3, 1, 2}, "[1,2,3]"),
        (frozenset({"b", "a"}), '["a","b"]'),
        ((1, "x"), '[1,"x"]'),
        (Color.RED, '"red"'),
        (b"\x00\xff", '"00ff"'),
    ],
)
def test_canonical_json_conversions(value: Any, expected: str) -> None:
    assert canonical_json(value) == expected


def test_canonical_json_mixed_set_is_deterministic() -> None:
    assert canonical_json({1, "a"}) == canonical_json({"a", 1})


def test_canonical_json_models_and_dataclasses() -> None:
    assert canonical_json(Inner(x=1, p=Path("d/e"))) == '{"p":"d/e","x":1}'
    assert canonical_json(DC(a=1, b=("x",))) == '{"a":1,"b":["x"]}'


def test_canonical_json_non_string_keys() -> None:
    assert canonical_json({1: "a", Path("k"): "b"}) == '{"1":"a","k":"b"}'
    with pytest.raises(ValueError, match="two keys"):
        canonical_json({1: "a", "1": "b"})


def test_canonical_json_rejects_nan_and_unknown_types() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def test_canonical_json_roundtrips_as_json() -> None:
    obj = {"a": [1, 2.5, None], "b": {"c": "d"}}
    assert json.loads(canonical_json(obj)) == obj


def test_sha256_text_known_vector() -> None:
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_text("ما عرفتش") == hashlib.sha256("ما عرفتش".encode()).hexdigest()


def test_sha256_file_streams(tmp_path: Path) -> None:
    f = tmp_path / "w.bin"
    data = bytes(range(256)) * 1000
    f.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(f) == expected
    assert sha256_file(f, chunk=7) == expected


def _write(root: Path, files: dict[str, bytes]) -> None:
    for name, content in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


FILES = {
    "adapter_config.json": b'{"r": 16}',
    "adapter_model.safetensors": b"weights",
    "tokenizer.json": b"tok",
    "tokenizer_config.json": b"tokcfg",
    "sub/model-00001.safetensors": b"shard",
}


def test_sha256_dir_is_order_independent(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, FILES)
    _write(b, dict(reversed(list(FILES.items()))))
    assert sha256_dir(a) == sha256_dir(b)


def test_sha256_dir_ignores_unmatched_files(tmp_path: Path) -> None:
    _write(tmp_path, FILES)
    before = sha256_dir(tmp_path)
    _write(tmp_path, {"optimizer.pt": b"state", "README.md": b"hi", ".eval_pending": b""})
    assert sha256_dir(tmp_path) == before


@pytest.mark.parametrize(
    "change",
    [
        {"adapter_model.safetensors": b"other"},  # content
        {"extra.safetensors": b"new"},  # added file
        {"sub/model-00001.safetensors": b"shard2"},  # nested content
    ],
)
def test_sha256_dir_detects_changes(tmp_path: Path, change: dict[str, bytes]) -> None:
    _write(tmp_path, FILES)
    before = sha256_dir(tmp_path)
    _write(tmp_path, change)
    assert sha256_dir(tmp_path) != before


def test_sha256_dir_detects_rename(tmp_path: Path) -> None:
    _write(tmp_path, FILES)
    before = sha256_dir(tmp_path)
    (tmp_path / "adapter_model.safetensors").rename(tmp_path / "adapter_model2.safetensors")
    assert sha256_dir(tmp_path) != before


def test_sha256_dir_patterns_and_errors(tmp_path: Path) -> None:
    _write(tmp_path, FILES)
    only_json = sha256_dir(tmp_path, patterns=("*.json",))
    _write(tmp_path, {"adapter_model.safetensors": b"changed"})
    assert sha256_dir(tmp_path, patterns=("*.json",)) == only_json
    with pytest.raises(NotADirectoryError):
        sha256_dir(tmp_path / "adapter_config.json")


def _root(**changes: Any) -> RootConfig:
    data: dict[str, Any] = {
        "run": {"name": "r1"},
        "model": {"name_or_path": "org/model"},
        "data": {"train": [{"path": "data/train.jsonl"}]},
        "train": {"max_steps": 10},
    }
    for dotted, value in changes.items():
        node = data
        *parents, leaf = dotted.split("__")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return RootConfig.model_validate(data)


def test_config_sha256_ignores_non_semantic_fields() -> None:
    base = config_sha256(_root())
    assert config_sha256(_root(run__name="other")) == base
    assert config_sha256(_root(run__output_dir="elsewhere")) == base
    assert config_sha256(_root(train__logging_steps=1)) == base
    assert config_sha256(_root(eval={"ground_truth": "gt.jsonl"})) == base


def test_config_sha256_tracks_semantic_fields() -> None:
    base = config_sha256(_root())
    assert config_sha256(_root(lora__r=8)) != base
    assert config_sha256(_root(run__seed=1)) != base
    assert config_sha256(_root(model__revision="abc")) != base


def test_config_sha256_custom_exclude_and_mappings() -> None:
    cfg = _root()
    assert config_sha256(cfg, exclude=()) != config_sha256(cfg)
    assert config_sha256(cfg, exclude=()) == config_sha256(cfg.model_dump(mode="json"), exclude=())
    assert config_sha256({"a": 1, "b": {"c": 2}}, exclude=("b.c", "missing.path")) == config_sha256(
        {"a": 1, "b": {}}, exclude=()
    )
    assert "eval" in NON_SEMANTIC_FIELDS
