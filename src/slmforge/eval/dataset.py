"""Ground-truth dataset: schema, loading, validation and subsampling.

File format (``data/eval/ground_truth.jsonl``): UTF-8, one JSON object per line; blank
lines and lines starting with ``//`` are ignored. The full field reference and the
E/W validation rules are in ``docs/dataset.md``.

Key semantics (these resolve the README's open questions):

* ``expected_answer`` is a **required key** that may be ``null``; ``null`` means the item
  is *unanswerable* and the correct behaviour is to abstain or reject the premise. A
  forgotten field therefore cannot silently become an unanswerable item.
* ``allow_abstention`` only applies to answerable items: ``true`` (default) makes
  abstaining neutral (ABSTAIN_ACCEPTED, utility 0); ``false`` marks required knowledge
  (ABSTAIN_UNWARRANTED, utility -0.5). Abstaining never counts as correct for accuracy.
* ``fact_id`` groups paraphrases/translations of one fact. It is the cluster key for the
  bootstrap, the subsample and cross-lingual metrics.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from slmforge._stub import todo
from slmforge.types import (
    NUMERIC_TYPES,
    AnswerType,
    Language,
    LangScript,
    Script,
    Split,
    UnanswerableReason,
    lang_script_of,
)

_ID_RE = r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,63}$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class NumericSpec(_Model):
    """Gold quantity with tolerances: ``|a - g| <= max(abs_tol, rel_tol * |g|)``."""

    value: Decimal
    unit: str | None = None  # "%", "MAD", "EUR", "USD", "days", "months", "years" or None
    abs_tol: Decimal = Field(Decimal(0), ge=0)
    rel_tol: Decimal = Field(Decimal(0), ge=0)
    accept_units: list[str] = []
    accept_fraction_for_percent: bool = False  # accept 0.7 for 70%


class DateSpec(_Model):
    """Gold partial ISO date: ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``."""

    value: str = Field(pattern=r"^\d{4}(-\d{2}(-\d{2})?)?$")

    @property
    def granularity(self) -> Literal["year", "month", "day"]:
        """Finest component present."""
        return ("year", "month", "day")[self.value.count("-")]  # type: ignore[return-value]


class CitationSpec(_Model):
    """Gold legal reference, e.g. article 12 of law 31-08."""

    article: str
    instrument: str | None = None
    instrument_aliases: list[str] = []


class SourceRef(_Model):
    """Where the fact comes from, and as of when it was true."""

    citation: str
    url: str | None = None
    as_of: date | None = None


class GroundTruthItem(_Model):
    """One labelled eval item. See ``docs/dataset.md`` for every field."""

    schema_version: Literal[1] = 1
    id: str = Field(pattern=_ID_RE)
    fact_id: str | None = None  # resolved to id
    prompt: str = Field(min_length=1, max_length=4000)
    language: Language
    script: Script | None = None  # resolved: fr/en -> latn; darija must declare it until M1 auto-detects
    code_switch: list[Literal["darija", "msa", "fr", "en"]] = []
    domain: str = "general"
    expected_answer: str | None  # required key; null == unanswerable
    answerable: bool | None = None  # resolved from expected_answer; an explicit value must agree
    answer_type: AnswerType | None = None  # inferred in M1 when None
    aliases: list[str] = []
    known_false: list[str] = []  # plausible distractors; any present -> decisive MISMATCH
    numeric: NumericSpec | None = None
    date_: DateSpec | None = Field(None, alias="date")
    citation: CitationSpec | None = None
    choices: list[str] = []
    allow_abstention: bool = True
    unanswerable_reason: UnanswerableReason | None = None
    premise_rejection_cues: list[str] = []
    context: str | None = None  # optional grounding passage -> item is "grounded"
    source: SourceRef | None = None
    anchor_terms: list[str] = []  # entities for the fact_seen contamination check
    split: Split = "test"  # only dev may be used to tune lexicons, tau, weights
    reviewed: bool = True  # teacher-assisted paraphrases enter as false
    label_version: int = 1
    annotators: list[str] = []
    tags: list[str] = []
    notes: str | None = None
    meta: dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy(cls, data: Any) -> Any:
        """Upgrade README-format records ``{prompt, expected_answer, language, domain?, allow_abstention?}``."""
        if not isinstance(data, dict) or "schema_version" in data or "id" in data:
            return data
        data = dict(data)
        key = f"{' '.join(str(data.get('prompt', '')).casefold().split())}|{data.get('language')}"
        data["id"] = "legacy-" + hashlib.sha256(key.encode()).hexdigest()[:12]
        data["tags"] = [*data.get("tags", []), "legacy"]
        return data

    @model_validator(mode="after")
    def _check(self) -> GroundTruthItem:
        """Per-item rules (E03-E09, E14) and derived fields."""
        set_ = object.__setattr__  # frozen model: resolve derived fields once, here
        if self.fact_id is None:
            set_(self, "fact_id", self.id)
        if self.script is None:
            if self.language == "darija":
                raise ValueError("E07: Darija items must declare script (arab, latn or mixed)")
            set_(self, "script", "latn")
        elif self.language != "darija" and self.script == "arab":
            raise ValueError("E07: fr/en items cannot have script=arab")

        if self.expected_answer is not None and not self.expected_answer.strip():
            raise ValueError("E03: expected_answer is empty; use null for unanswerable items")
        derived = self.expected_answer is not None
        if self.answerable is not None and self.answerable != derived:
            raise ValueError("E04: answerable disagrees with expected_answer (null <=> unanswerable)")
        set_(self, "answerable", derived)

        if not derived:
            if self.unanswerable_reason is None:
                raise ValueError("E05: unanswerable items need unanswerable_reason")
            if self.aliases or self.numeric or self.date_ or self.citation:
                raise ValueError("E05: unanswerable items cannot have aliases/numeric/date/citation")
            if not self.allow_abstention:
                raise ValueError("E05: unanswerable items must have allow_abstention=true")
        else:
            if self.unanswerable_reason is not None or self.premise_rejection_cues:
                raise ValueError("E06: answerable items cannot set unanswerable_reason/premise_rejection_cues")
            if self.answer_type in NUMERIC_TYPES and self.numeric is None:
                raise ValueError(f"E08: answer_type={self.answer_type} needs a numeric spec")
            if self.answer_type == "percent" and self.numeric and self.numeric.unit not in (None, "%"):
                raise ValueError("E08: percent items need numeric.unit '%'")
            if self.answer_type == "date" and self.date_ is None:
                raise ValueError("E09: answer_type=date needs a date spec")
            if self.answer_type == "citation" and self.citation is None:
                raise ValueError("E09: answer_type=citation needs a citation spec")
            if self.answer_type == "choice" and self.expected_answer not in self.choices:
                raise ValueError("E09: choice items need choices containing expected_answer")
        if any(not a.strip() for a in self.aliases):
            raise ValueError("E14: empty alias")
        return self

    @property
    def lang_script(self) -> LangScript:
        """Bucket key (``darija_arab``, ``darija_latn``, ``darija_mixed``, ``fr``, ``en``)."""
        return lang_script_of(self.language, self.script)

    @property
    def code_switched(self) -> bool:
        """Whether the prompt mixes in other languages."""
        return bool(self.code_switch)

    @property
    def grounded(self) -> bool:
        """Whether a grounding context passage is supplied."""
        return self.context is not None

    @property
    def content_hash(self) -> str:
        """Hash of scoring-relevant fields (part of the generation-cache key)."""
        d = self.model_dump(mode="json", by_alias=True, exclude={"notes", "tags", "annotators", "meta"})
        return hashlib.sha256(_canonical(d).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class DatasetIssue:
    """A validation error or warning with its location."""

    level: Literal["error", "warning"]
    code: str
    message: str
    line: int | None = None
    item_id: str | None = None

    def __str__(self) -> str:
        where = f"line {self.line}" if self.line is not None else (self.item_id or "dataset")
        return f"{self.level.upper()} {self.code} [{where}]: {self.message}"


class GroundTruthError(Exception):
    """Raised by :func:`load_ground_truth` when the file has errors."""

    def __init__(self, issues: Sequence[DatasetIssue]) -> None:
        self.issues = list(issues)
        errors = [i for i in self.issues if i.level == "error"]
        super().__init__(f"{len(errors)} error(s) in ground truth:\n" + "\n".join(map(str, errors[:20])))


@dataclass(frozen=True)
class GroundTruthSet:
    """A validated set of items with its order-independent hash."""

    items: tuple[GroundTruthItem, ...]
    path: Path | None = None
    issues: tuple[DatasetIssue, ...] = ()
    sha256: str = field(default="")

    def __post_init__(self) -> None:
        if not self.sha256:
            object.__setattr__(self, "sha256", dataset_sha256(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def by_id(self) -> dict[str, GroundTruthItem]:
        """Index items by id."""
        return {i.id: i for i in self.items}

    def facts(self) -> dict[str, list[GroundTruthItem]]:
        """Group items by fact_id."""
        out: dict[str, list[GroundTruthItem]] = defaultdict(list)
        for i in self.items:
            out[i.fact_id or i.id].append(i)
        return dict(out)

    def filter(
        self,
        *,
        splits: Collection[str] = ("test",),
        require_reviewed: bool = False,
        exclude_ids: Collection[str] = (),
    ) -> GroundTruthSet:
        """Subset by split (``"all"`` keeps both), review status and explicit exclusions."""
        keep = tuple(
            i
            for i in self.items
            if ("all" in splits or i.split in splits)
            and (i.reviewed or not require_reviewed)
            and i.id not in exclude_ids
        )
        return GroundTruthSet(keep, self.path, self.issues)

    def subsample(self, fraction: float, *, seed: int, min_per_stratum: int = 20) -> GroundTruthSet:
        """Cluster-level hash-threshold sample over fact_id, topped up per stratum.

        Keep a fact iff ``u(seed, fact_id) < fraction`` where ``u`` is the first 8 bytes of
        ``sha256(f"{seed}:{fact_id}")`` scaled to [0, 1); then add facts in ascending ``u``
        order to any (lang_script, domain, answerable) stratum below ``min_per_stratum``.
        Deterministic, independent of file order, and stable when items are appended.
        """
        todo("M3", "GroundTruthSet.subsample")

    def coverage_table(self) -> dict[tuple[str, str, bool], int]:
        """Item counts per (lang_script, domain, answerable)."""
        out: dict[tuple[str, str, bool], int] = defaultdict(int)
        for i in self.items:
            out[(i.lang_script, i.domain, bool(i.answerable))] += 1
        return dict(out)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def dataset_sha256(items: Sequence[GroundTruthItem]) -> str:
    """Canonical, order-independent dataset hash."""
    lines = sorted(_canonical(i.model_dump(mode="json", by_alias=True)) for i in items)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _rule_code(msg: str) -> str:
    m = re.search(r"\b([EW]\d{2})\b", msg)
    return m.group(1) if m else "E01"


def parse_lines(lines: Sequence[str]) -> tuple[list[GroundTruthItem], list[DatasetIssue]]:
    """Parse JSONL lines into items, collecting per-line issues instead of stopping."""
    items: list[GroundTruthItem] = []
    issues: list[DatasetIssue] = []
    for n, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            issues.append(DatasetIssue("error", "E01", f"invalid JSON: {e.msg}", line=n))
            continue
        try:
            item = GroundTruthItem.model_validate(obj)
        except ValidationError as e:
            for err in e.errors():
                msg = str(err.get("msg", ""))
                loc = ".".join(str(p) for p in err.get("loc", ()))
                code = "E03" if loc == "expected_answer" and err.get("type") == "missing" else _rule_code(msg)
                if loc == "id" and code == "E01":
                    code = "E02"
                issues.append(
                    DatasetIssue("error", code, f"{loc}: {msg}" if loc else msg, line=n,
                                 item_id=obj.get("id") if isinstance(obj, dict) else None)
                )
            continue
        if "legacy" in item.tags and "schema_version" not in obj:
            issues.append(DatasetIssue("warning", "W01", "legacy record upgraded; derived id", n, item.id))
        items.append(item)
    return items, issues


def validate_set(items: Sequence[GroundTruthItem], *, allowed_domains: Collection[str] | None = None) -> list[DatasetIssue]:
    """Cross-item rules: E02 duplicate ids, E12 fact consistency, E13 fact split leakage, domain check."""
    issues: list[DatasetIssue] = []
    seen: set[str] = set()
    for i in items:
        if i.id in seen:
            issues.append(DatasetIssue("error", "E02", f"duplicate id {i.id}", item_id=i.id))
        seen.add(i.id)
        if allowed_domains is not None and i.domain not in allowed_domains:
            issues.append(DatasetIssue("error", "E01", f"domain {i.domain!r} not in {sorted(allowed_domains)}",
                                       item_id=i.id))
    by_fact: dict[str, list[GroundTruthItem]] = defaultdict(list)
    for i in items:
        by_fact[i.fact_id or i.id].append(i)
    for fid, members in by_fact.items():
        for attr in ("answerable", "domain", "answer_type"):
            if len({getattr(m, attr) for m in members}) > 1:
                issues.append(DatasetIssue("error", "E12", f"fact {fid}: members disagree on {attr}"))
        if len({m.split for m in members}) > 1:
            issues.append(DatasetIssue("error", "E13", f"fact {fid} appears in both dev and test"))
    # E10, E11 and warnings W02-W13 need the text layer: M1.
    return issues


def load_ground_truth(
    path: Path, *, strict: bool = False, allowed_domains: Collection[str] | None = None
) -> GroundTruthSet:
    """Parse and validate a ground-truth file; raise :class:`GroundTruthError` on errors."""
    text = Path(path).read_text(encoding="utf-8-sig")
    items, issues = parse_lines(text.splitlines())
    issues += validate_set(items, allowed_domains=allowed_domains)
    if strict:
        issues = [DatasetIssue("error", i.code, i.message, i.line, i.item_id) for i in issues]
    if any(i.level == "error" for i in issues):
        raise GroundTruthError(issues)
    return GroundTruthSet(tuple(items), Path(path), tuple(issues))


def dataset_stats(gt: GroundTruthSet, *, icc: float | None = None) -> dict[str, Any]:
    """Coverage table, per-stratum CI half-widths, design effect and minimum detectable effect."""
    todo("M5", "dataset_stats")
