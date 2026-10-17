"""LoRA target resolution per architecture (torch-free)."""

from __future__ import annotations

import pytest

from slmforge.train.targets import DEFAULT_TARGETS, resolve_targets

LLAMA_LIKE = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.mark.parametrize("model_type", ["llama", "mistral", "qwen2"])
def test_llama_like_targets(model_type: str) -> None:
    assert resolve_targets(model_type, "auto") == LLAMA_LIKE


def test_phi3_keeps_fused_projections() -> None:
    assert resolve_targets("phi3", "auto") == ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def test_auto_is_case_insensitive_and_returns_a_copy() -> None:
    out = resolve_targets(" Qwen2 ", "auto")
    assert out == LLAMA_LIKE
    assert isinstance(out, list)
    out.append("lm_head")
    assert "lm_head" not in DEFAULT_TARGETS["qwen2"]


def test_default_targets_is_read_only() -> None:
    with pytest.raises(TypeError):
        DEFAULT_TARGETS["gpt2"] = ("c_attn",)  # type: ignore[index]


def test_unknown_architecture_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"no default LoRA targets.*all-linear"):
        resolve_targets("gpt_neox", "auto")


def test_all_linear_passes_through() -> None:
    assert resolve_targets("anything", "all-linear") == "all-linear"


def test_explicit_list_is_deduplicated_in_order() -> None:
    requested = ["q_proj", "v_proj", "q_proj"]
    assert resolve_targets("gpt_neox", requested) == ["q_proj", "v_proj"]
    assert requested == ["q_proj", "v_proj", "q_proj"]


@pytest.mark.parametrize("bad", [[], ["q_proj", " "]])
def test_empty_explicit_list_is_an_error(bad: list[str]) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        resolve_targets("llama", bad)


def test_unknown_string_is_an_error() -> None:
    with pytest.raises(ValueError, match="auto"):
        resolve_targets("llama", "q_proj")  # type: ignore[arg-type]
