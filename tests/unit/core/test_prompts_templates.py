"""prompts: packaged templates, validation, per-item selection and message building."""

from __future__ import annotations

import shutil
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import pytest

from slmforge.eval.dataset import GroundTruthItem
from slmforge.eval.generate import GenRequest
from slmforge.eval.prompts import PromptBuilder, PromptTemplateSet, RenderedPrompt
from slmforge.repro import sha256_text
from slmforge.types import CORE_BUCKETS

MARKERS = {"en": "Final answer:", "fr": "Réponse finale :", "darija_arab": "الجواب:", "darija_latn": "Jawab:"}


def _item(language: str = "en", script: str | None = None, **kw: Any) -> GroundTruthItem:
    data: dict[str, Any] = {
        "id": f"it-{language}-{script or 'x'}",
        "prompt": "What is the rate?",
        "language": language,
        "script": script,
        "expected_answer": "70%",
    }
    data.update(kw)
    return GroundTruthItem.model_validate(data)


ITEMS = {
    "en": _item("en"),
    "fr": _item("fr"),
    "darija_arab": _item("darija", "arab"),
    "darija_latn": _item("darija", "latn"),
}


@pytest.fixture(scope="module")
def templates() -> PromptTemplateSet:
    return PromptTemplateSet.load()


def test_default_set_loads_every_core_bucket(templates: PromptTemplateSet) -> None:
    assert templates.id == "default_v1"
    assert set(templates.templates) == set(CORE_BUCKETS)
    assert len(templates.sha256) == 64
    assert PromptTemplateSet.load().sha256 == templates.sha256


@pytest.mark.parametrize("bucket", CORE_BUCKETS)
def test_markers_and_canonical_abstentions(templates: PromptTemplateSet, bucket: str) -> None:
    t = templates.templates[bucket]
    assert t.final_answer_marker == MARKERS[bucket]
    assert t.lang_script == bucket and t.id == "default_v1"
    assert t.canonical_abstention.strip()


@pytest.mark.parametrize("bucket", CORE_BUCKETS)
def test_for_item_picks_the_bucket(templates: PromptTemplateSet, bucket: str) -> None:
    assert templates.for_item(ITEMS[bucket]).lang_script == bucket


def test_mixed_script_uses_fallback(templates: PromptTemplateSet) -> None:
    mixed = _item("darija", "mixed")
    assert templates.for_item(mixed).lang_script == "darija_arab"
    assert templates.for_item(mixed, mixed_fallback="darija_latn").lang_script == "darija_latn"


@pytest.mark.parametrize("bucket", CORE_BUCKETS)
def test_messages_abstain_ok(templates: PromptTemplateSet, bucket: str) -> None:
    t = templates.templates[bucket]
    msgs = PromptBuilder(templates).messages(ITEMS[bucket])
    assert [m["role"] for m in msgs] == ["system", "user"]
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert t.final_answer_marker in system
    assert t.canonical_abstention in system  # the abstention instruction quotes it
    assert "{" not in system and "}" not in system
    assert user == "What is the rate?"


@pytest.mark.parametrize("bucket", CORE_BUCKETS)
def test_messages_neutral_omits_abstention_instruction(templates: PromptTemplateSet, bucket: str) -> None:
    t = templates.templates[bucket]
    system = PromptBuilder(templates, mode="neutral").messages(ITEMS[bucket])[0]["content"]
    assert t.final_answer_marker in system
    assert t.canonical_abstention not in system


def test_context_is_inserted_when_present(templates: PromptTemplateSet) -> None:
    item = _item("fr", context="Le taux est fixé par la loi.")
    user = PromptBuilder(templates).messages(item)[1]["content"]
    assert "Le taux est fixé par la loi." in user
    assert user.index("Le taux est fixé") < user.index("What is the rate?")
    assert user.endswith("What is the rate?")


def test_braces_in_prompt_are_preserved(templates: PromptTemplateSet) -> None:
    item = _item("en", prompt="Is {x} or {0} a field?", context="ctx {context}")
    user = PromptBuilder(templates).messages(item)[1]["content"]
    assert "Is {x} or {0} a field?" in user and "ctx {context}" in user


def test_request_carries_item_identity(templates: PromptTemplateSet) -> None:
    item = ITEMS["darija_latn"]
    req = PromptBuilder(templates).request(item)
    assert isinstance(req, GenRequest)
    assert (req.item_id, req.content_hash, req.lang_script) == (item.id, item.content_hash, "darija_latn")
    assert list(req.messages) == PromptBuilder(templates).messages(item)


def test_fingerprint_depends_on_mode(templates: PromptTemplateSet) -> None:
    a, b = PromptBuilder(templates), PromptBuilder(templates, mode="neutral")
    assert a.fingerprint == PromptBuilder(templates).fingerprint
    assert a.fingerprint != b.fingerprint
    with pytest.raises(ValueError):
        PromptBuilder(templates, mode="chatty")  # type: ignore[arg-type]


def test_rendered_prompt_hash() -> None:
    r = RenderedPrompt.from_text("it-1", "<s>hello")
    assert r.sha256 == sha256_text("<s>hello") and r.method == "chat_template"


# --------------------------------------------------------------------------- custom roots and validation


@pytest.fixture
def custom_root(tmp_path: Path) -> Path:
    with as_file(files("slmforge").joinpath("resources", "prompts", "default_v1")) as src:
        shutil.copytree(src, tmp_path / "mine")
    return tmp_path


def test_custom_root_loads_and_hash_tracks_content(custom_root: Path, templates: PromptTemplateSet) -> None:
    mine = PromptTemplateSet.load("mine", root=custom_root)
    assert mine.id == "mine"
    assert mine.sha256 != templates.sha256  # the id is part of the hash
    f = custom_root / "mine" / "en.yaml"
    before = mine.sha256
    f.write_text(f.read_text(encoding="utf-8") + "\n# a comment only\n", encoding="utf-8")
    assert PromptTemplateSet.load("mine", root=custom_root).sha256 == before
    f.write_text(f.read_text(encoding="utf-8").replace("I don't know.", "No idea."), encoding="utf-8")
    assert PromptTemplateSet.load("mine", root=custom_root).sha256 != before


def _break(root: Path, old: str, new: str) -> None:
    f = root / "mine" / "en.yaml"
    text = f.read_text(encoding="utf-8")
    assert old in text
    f.write_text(text.replace(old, new), encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('user_template: "{context_block}{question}"', 'user_template: "{question}"', "context_block"),
        (
            'user_template: "{context_block}{question}"',
            'user_template: "{context_block}{query}"',
            "unknown field",
        ),
        ('"Final answer:"', '""', "non-empty"),
        ("schema_version: 1", "schema_version: 2", "schema_version"),
        (
            'canonical_abstention: "I don\'t know."',
            'canonical_abstention: "I don\'t know."\nextra: 1',
            "unknown keys",
        ),
        (
            'context_template: "Use the following context:\\n{context}\\n\\n"',
            'context_template: "{context"',
            "context",
        ),
    ],
)
def test_invalid_templates_are_rejected(custom_root: Path, old: str, new: str, match: str) -> None:
    _break(custom_root, old, new)
    with pytest.raises(ValueError, match=match):
        PromptTemplateSet.load("mine", root=custom_root)


def test_missing_bucket_file(custom_root: Path) -> None:
    (custom_root / "mine" / "darija_latn.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="darija_latn"):
        PromptTemplateSet.load("mine", root=custom_root)


@pytest.mark.parametrize("bad_id", ["../etc", "", "a/b"])
def test_template_id_is_validated(bad_id: str) -> None:
    with pytest.raises(ValueError):
        PromptTemplateSet.load(bad_id)
