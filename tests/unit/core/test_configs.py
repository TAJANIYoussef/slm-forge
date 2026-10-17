"""Every shipped config validates, and the fictional sample data has the advertised shape."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

from slmforge.config.load import load_eval_config, load_root_config, load_yaml
from slmforge.config.schema import DistillConfig, RootConfig
from slmforge.data.loader import ChatRecord
from slmforge.eval.dataset import load_ground_truth
from slmforge.train.backends import choose_backend
from slmforge.train.targets import resolve_targets
from slmforge.types import CORE_BUCKETS

REPO = Path(__file__).resolve().parents[3]
RUN_CONFIGS = {
    "qwen2.5-7b-darija.yaml": ("Qwen/Qwen2.5-7B-Instruct", "qwen2"),
    "phi-4-mini-darija.yaml": ("microsoft/Phi-4-mini-instruct", "phi3"),
    "mistral-7b-darija.yaml": ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    "smollm2-1.7b-darija.yaml": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "llama"),
    "tiny-cpu-test.yaml": ("/models/tiny", "llama"),
}
SAMPLE_GT = REPO / "data/eval/ground_truth.sample.jsonl"
SAMPLE_CHAT = REPO / "data/train/sample_chat.jsonl"


@pytest.fixture(autouse=True)
def _tiny_model_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINY_MODEL_DIR", "/models/tiny")


def test_every_config_file_is_covered() -> None:
    shipped = {p.relative_to(REPO / "configs").as_posix() for p in (REPO / "configs").rglob("*.yaml")}
    assert shipped == {*RUN_CONFIGS, "distill.yaml", "eval/factual-reliability.yaml"}


@pytest.mark.parametrize("name", sorted(RUN_CONFIGS))
def test_run_configs_validate(name: str) -> None:
    cfg = load_root_config(REPO / "configs" / name)
    model_id, model_type = RUN_CONFIGS[name]
    assert cfg.model.name_or_path == model_id
    assert cfg.model.revision is None  # pinned by the user; the comment says how
    assert cfg.run.name == name.removesuffix(".yaml")
    assert cfg.eval is not None
    assert cfg.eval.schedule.mode == "posthoc"  # inline eval is off by default
    assert cfg.eval.judge.enabled is False
    assert cfg.distill is None
    assert cfg.lora.target_modules == "auto"
    assert isinstance(resolve_targets(model_type, cfg.lora.target_modules), list)
    assert load_eval_config(REPO / "configs" / name) == cfg.eval
    assert RootConfig.model_validate(cfg.model_dump(mode="json")) == cfg


def test_revision_comment_tells_users_to_pin() -> None:
    for name in RUN_CONFIGS:
        if name == "tiny-cpu-test.yaml":
            continue
        text = (REPO / "configs" / name).read_text(encoding="utf-8")
        assert re.search(r"revision: null\s+# pin a 40-char commit SHA", text), name


def test_tiny_cpu_config() -> None:
    cfg = load_root_config(REPO / "configs/tiny-cpu-test.yaml")
    assert cfg.train.backend == "hf_peft"
    assert cfg.model.dtype == "fp32"
    assert (cfg.train.max_steps, cfg.train.save_steps) == (6, 2)
    assert cfg.quant.load_in_4bit is False
    assert cfg.eval is not None
    assert cfg.eval.subsample.fraction is None
    assert cfg.eval.ground_truth == Path("data/eval/ground_truth.sample.jsonl")
    assert [s.path for s in cfg.data.train] == [Path("data/train/sample_chat.jsonl")]
    # 4-bit is off, so the backend choice needs no GPU and never probes CUDA
    choice = choose_backend(cfg.train.backend, load_in_4bit=cfg.quant.load_in_4bit, has_module=lambda m: True)
    assert (choice.name, choice.cuda) == ("hf_peft", None)


def test_tiny_cpu_config_needs_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINY_MODEL_DIR")
    with pytest.raises(ValueError, match="TINY_MODEL_DIR"):
        load_root_config(REPO / "configs/tiny-cpu-test.yaml")


def test_eval_config_validates() -> None:
    cfg = load_eval_config(REPO / "configs/eval/factual-reliability.yaml")
    assert cfg.suite == ["factual-reliability"]
    assert cfg.split == "test"
    assert cfg.offline is True
    assert cfg.subsample.fraction is None


def test_distill_config_is_off() -> None:
    cfg = DistillConfig.model_validate(load_yaml(REPO / "configs/distill.yaml"))
    assert (cfg.enabled, cfg.tos_acknowledged, cfg.allow_send_data) == (False, False, False)
    assert cfg.terms_reviewed_on is None and cfg.acknowledged_by is None
    assert cfg.redact_pii is True


def test_protocols_readme_exists() -> None:
    assert "protocol lock" in (REPO / "configs/protocols/README.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- sample ground truth


def test_sample_ground_truth_loads_cleanly() -> None:
    gt = load_ground_truth(SAMPLE_GT, allowed_domains={"finance", "legal"})
    assert not [i for i in gt.issues if i.level == "error"]
    assert len(gt) == 24
    assert Counter(i.lang_script for i in gt.items) == dict.fromkeys(CORE_BUCKETS, 6)
    assert {i.domain for i in gt.items} == {"finance", "legal"}
    for bucket in CORE_BUCKETS:
        assert {i.domain for i in gt.items if i.lang_script == bucket} == {"finance", "legal"}


def test_sample_unanswerable_share() -> None:
    items = load_ground_truth(SAMPLE_GT).items
    unans = [i for i in items if not i.answerable]
    assert len(unans) == 6
    assert all(i.unanswerable_reason is not None for i in unans)
    assert {i.unanswerable_reason for i in unans} == {"nonexistent_entity", "false_premise"}
    for bucket in CORE_BUCKETS:
        share = sum(1 for i in unans if i.lang_script == bucket) / 6
        assert 0.15 <= share <= 0.5, bucket  # W08 range


def test_sample_facts_are_shared_across_buckets_and_agree() -> None:
    gt = load_ground_truth(SAMPLE_GT)
    facts = gt.facts()
    assert len(facts) == 8
    assert sum(1 for members in facts.values() if len({m.lang_script for m in members}) == 4) == 4
    for fid, members in facts.items():
        assert len(members) >= 2, fid
        assert len({m.lang_script for m in members}) == len(members), fid  # one member per bucket
        for attr in ("answerable", "domain", "answer_type", "split", "numeric", "date_", "citation"):
            assert len({json.dumps(getattr(m, attr), default=str, sort_keys=True) for m in members}) == 1, (
                fid,
                attr,
            )


def test_sample_typed_answers() -> None:
    items = [i for i in load_ground_truth(SAMPLE_GT).items if i.answerable]
    types = Counter(i.answer_type for i in items)
    assert {"percent", "currency", "numeric", "date", "citation", "boolean"} <= set(types)
    for i in items:
        assert i.answer_type is not None
        if i.answer_type in {"percent", "currency", "numeric"}:
            assert i.numeric is not None
            assert i.known_false, i.id
        assert i.source is not None and i.source.as_of is not None  # W06
    assert any(i.split == "dev" for i in items)


def test_sample_darija_aliases_in_both_scripts() -> None:
    arabic = re.compile(r"[؀-ۿ]")
    latin = re.compile(r"[A-Za-z]")
    for i in load_ground_truth(SAMPLE_GT).items:
        if i.language != "darija" or not i.answerable:
            continue
        forms = [i.expected_answer or "", *i.aliases]
        assert any(arabic.search(a) for a in forms), i.id
        assert any(latin.search(a) for a in forms), i.id
        other = latin if i.script == "arab" else arabic  # W02: an alias in the other script
        assert any(other.search(a) for a in i.aliases), i.id


def test_sample_is_fictional() -> None:
    text = SAMPLE_GT.read_text(encoding="utf-8")
    assert text.startswith("// slm-forge sample ground truth: 24 FICTIONAL items")
    for i in load_ground_truth(SAMPLE_GT).items:
        assert "fictional" in i.tags
        assert any(k in i.prompt for k in ("Atlas", "أطلس")), i.id


# --------------------------------------------------------------------------- sample training data


def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.casefold())
    return {tuple(toks[k : k + n]) for k in range(len(toks) - n + 1)}


def test_sample_chat_does_not_overlap_eval_prompts() -> None:
    records = [
        ChatRecord.model_validate(json.loads(line)) for line in SAMPLE_CHAT.read_text("utf-8").splitlines()
    ]
    train = set().union(*(_ngrams(m["content"]) for r in records for m in r.messages))
    by_bucket: dict[str, int] = defaultdict(int)
    for item in load_ground_truth(SAMPLE_GT).items:
        assert not (_ngrams(item.prompt) & train), item.id
        by_bucket[item.lang_script] += 1
    assert set(by_bucket) == set(CORE_BUCKETS)


def test_yaml_files_parse_as_mappings() -> None:
    for p in (REPO / "configs").rglob("*.yaml"):
        assert isinstance(yaml.safe_load(p.read_text(encoding="utf-8")), dict), p
