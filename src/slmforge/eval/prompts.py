"""Per-locale prompt templates, message building and (M2) chat-template rendering.

A template set (``resources/prompts/<id>/{en,fr,darija_arab,darija_latn}.yaml``) gives each
core bucket a system prompt, an abstention instruction, a final-answer marker, the user and
context templates and the locale's canonical abstention. The prompt contract (plan §2.4):
every answer ends with a localized final line (``Final answer:``, ``Réponse finale :``,
``الجواب:``, ``Jawab:``) that ``eval.extract.find_answer_region`` looks for.

Template strings use ``str.format`` fields; the allowed fields per string are in
:data:`TEMPLATE_FIELDS` and are checked at load time. ``darija_mixed`` items use the
``mixed_fallback`` bucket (Arabic script by default).

The set hash covers the *parsed* template content (canonical JSON), so comments and YAML
formatting do not change the protocol but any change to what the model sees does.
"""

from __future__ import annotations

import dataclasses
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import yaml

from slmforge._stub import todo
from slmforge.eval.generate import GenRequest
from slmforge.repro import canonical_json, sha256_text
from slmforge.types import CORE_BUCKETS, LangScript

if TYPE_CHECKING:
    from slmforge.eval.dataset import GroundTruthItem

PromptMode = Literal["abstain_ok", "neutral"]
RenderMethod = Literal["chat_template", "folded_system", "plain"]

TEMPLATE_SCHEMA_VERSION: Final = 1
_SHARED: Final = frozenset({"marker", "canonical_abstention"})
TEMPLATE_FIELDS: Final[Mapping[str, frozenset[str]]] = {
    "system": _SHARED,
    "abstention_instruction": _SHARED,
    "final_answer_marker": frozenset(),
    "user_template": _SHARED | {"question", "context_block"},
    "context_template": _SHARED | {"context"},
    "canonical_abstention": frozenset(),
}
"""Format fields each template string may use."""
REQUIRED_FIELDS: Final[Mapping[str, frozenset[str]]] = {
    "user_template": frozenset({"question", "context_block"}),
    "context_template": frozenset({"context"}),
}
_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PromptTemplate:
    """Per-locale system prompt, abstention instruction and final-answer marker."""

    id: str
    lang_script: LangScript
    system: str
    abstention_instruction: str
    final_answer_marker: str
    user_template: str
    context_template: str
    canonical_abstention: str

    @property
    def shared_fields(self) -> dict[str, str]:
        """Values of the fields every template string may reference."""
        return {"marker": self.final_answer_marker, "canonical_abstention": self.canonical_abstention}


def _format_fields(template: str) -> set[str]:
    """Field names used by a ``str.format`` template (raises ValueError on malformed braces).

    Only plain ``{name}`` fields are allowed: a format spec or conversion (``{q:>10}``,
    ``{q!r}``, nested ``{q:{q}}``) would make the rendered prompt depend on item values.
    """
    names: set[str] = set()
    for _, name, spec, conversion in string.Formatter().parse(template):
        if name is None:
            continue
        if spec or conversion:
            raise ValueError(f"field {{{name}}} must not use a format spec or conversion")
        if not name.isidentifier():
            raise ValueError(f"field {{{name}}} must be a plain name (no indexing or attributes)")
        names.add(name)
    return names


def _parse_template(template_id: str, lang_script: LangScript, text: str, label: str) -> PromptTemplate:
    """Validate one locale YAML and build its :class:`PromptTemplate`."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{label}: top level must be a mapping")
    if data.get("schema_version") != TEMPLATE_SCHEMA_VERSION:
        raise ValueError(f"{label}: schema_version must be {TEMPLATE_SCHEMA_VERSION}")
    keys = set(data) - {"schema_version"}
    missing, unknown = set(TEMPLATE_FIELDS) - keys, keys - set(TEMPLATE_FIELDS)
    if missing or unknown:
        raise ValueError(f"{label}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    for key, allowed in TEMPLATE_FIELDS.items():
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}: {key} must be a non-empty string")
        try:
            used = _format_fields(value)
        except ValueError as e:
            raise ValueError(f"{label}: {key}: {e}") from None
        if bad := used - allowed:
            raise ValueError(
                f"{label}: {key} uses unknown field(s) {sorted(bad)}; allowed: {sorted(allowed)}"
            )
        if lacking := REQUIRED_FIELDS.get(key, frozenset()) - used:
            raise ValueError(f"{label}: {key} must contain {sorted('{' + f + '}' for f in lacking)}")
    return PromptTemplate(id=template_id, lang_script=lang_script, **{k: data[k] for k in TEMPLATE_FIELDS})


class PromptTemplateSet:
    """All locale templates of one template id; hashed into the protocol."""

    def __init__(self, template_id: str, templates: Mapping[str, PromptTemplate]) -> None:
        self.id = template_id
        self.templates: dict[str, PromptTemplate] = dict(templates)
        content = {
            "id": template_id,
            "templates": {k: dataclasses.asdict(t) for k, t in self.templates.items()},
        }
        self.sha256 = sha256_text(canonical_json(content))

    @classmethod
    def load(cls, template_id: str = "default_v1", root: Path | None = None) -> PromptTemplateSet:
        """Load ``<root>/<template_id>/<bucket>.yaml`` for every core bucket (packaged when root is None)."""
        if not _ID_RE.match(template_id):
            raise ValueError(f"invalid prompt template id {template_id!r}")
        base: Traversable | Path = (
            files("slmforge").joinpath("resources", "prompts", template_id)
            if root is None
            else Path(root) / template_id
        )
        templates: dict[str, PromptTemplate] = {}
        for bucket in CORE_BUCKETS:
            f = base.joinpath(f"{bucket}.yaml")
            if not f.is_file():
                raise FileNotFoundError(f"prompt template {template_id!r}: missing {bucket}.yaml in {base}")
            templates[bucket] = _parse_template(
                template_id, bucket, f.read_text(encoding="utf-8"), f"{template_id}/{bucket}.yaml"
            )
        return cls(template_id, templates)

    def for_item(
        self, item: GroundTruthItem, *, mixed_fallback: LangScript = "darija_arab"
    ) -> PromptTemplate:
        """Template in the item's language and script (``darija_mixed`` -> ``mixed_fallback``)."""
        bucket = mixed_fallback if item.lang_script == "darija_mixed" else item.lang_script
        try:
            return self.templates[bucket]
        except KeyError:
            raise KeyError(f"prompt template set {self.id!r} has no template for {bucket!r}") from None


@dataclass(frozen=True)
class RenderedPrompt:
    """Final prompt text (what the model sees) and its hash."""

    item_id: str
    text: str
    sha256: str
    method: RenderMethod = "chat_template"

    @classmethod
    def from_text(cls, item_id: str, text: str, *, method: RenderMethod = "chat_template") -> RenderedPrompt:
        """Build a rendered prompt, hashing ``text``."""
        return cls(item_id=item_id, text=text, sha256=sha256_text(text), method=method)


class PromptBuilder:
    """Builds messages (torch-free) and renders them with a tokenizer's chat template (M2)."""

    def __init__(
        self,
        templates: PromptTemplateSet,
        *,
        mode: PromptMode = "abstain_ok",
        mixed_fallback: LangScript = "darija_arab",
    ) -> None:
        if mode not in ("abstain_ok", "neutral"):
            raise ValueError(f"unknown prompt mode {mode!r}")
        self.templates = templates
        self.mode: PromptMode = mode
        self.mixed_fallback: LangScript = mixed_fallback

    def messages(self, item: GroundTruthItem) -> list[dict[str, str]]:
        """System + user turns; context block inserted when present; abstain_ok adds the instruction."""
        t = self.templates.for_item(item, mixed_fallback=self.mixed_fallback)
        shared = t.shared_fields
        system = t.system.format_map(shared).strip()
        if self.mode == "abstain_ok":
            system = f"{system} {t.abstention_instruction.format_map(shared).strip()}"
        context_block = (
            "" if item.context is None else t.context_template.format_map({**shared, "context": item.context})
        )
        user = t.user_template.format_map({**shared, "question": item.prompt, "context_block": context_block})
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def request(self, item: GroundTruthItem) -> GenRequest:
        """The generation request for one item (messages + content hash + bucket)."""
        return GenRequest(
            item_id=item.id,
            content_hash=item.content_hash,
            lang_script=item.lang_script,
            messages=tuple(self.messages(item)),
        )

    def render(self, item_id: str, messages: Sequence[Mapping[str, str]], tokenizer: Any) -> RenderedPrompt:
        """Chat template with a generation prompt; fold system into user on error; plain-text fallback."""
        todo("M2", "PromptBuilder.render")

    @property
    def fingerprint(self) -> str:
        """Template set id + sha + mode (part of the generator/cache identity)."""
        return sha256_text(
            canonical_json(
                {
                    "template_id": self.templates.id,
                    "template_sha256": self.templates.sha256,
                    "mode": self.mode,
                    "mixed_fallback": self.mixed_fallback,
                }
            )
        )
