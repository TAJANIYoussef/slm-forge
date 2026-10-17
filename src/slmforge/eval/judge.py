"""Detector validation gates (M7) and the optional local same-answer judge (M9).

``validate-detectors`` scores human-labelled *real* responses
(``data/eval/labels/responses_gold.jsonl``) and compares every component with the human
labels, per lang_script stratum: abstention and hedge detectors (precision/recall), the
rule matcher on decisive verdicts (agreement on the **non-exact** subset), the residual
commitment rule and, if configured, the judge (Cohen's kappa and false-match rate on
human-labelled UNRESOLVED items, the only items it ever sees; critique 22). A failing
component is never used silently: its strata are listed as unvalidated and flagged.

The judge is off by default, local only (``allow_remote`` must be set explicitly), greedy
and JSON-only; it runs after the evaluated model is unloaded and its id/revision go into
the protocol. Only :class:`LocalHFJudge` may import torch/transformers, inside its methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from slmforge._stub import todo
from slmforge.config.schema import GatesConfig

if TYPE_CHECKING:
    from slmforge.config.schema import EvalConfig
    from slmforge.eval.dataset import GroundTruthItem
    from slmforge.eval.factual_reliability import FactualReliabilitySuite

JUDGE_PROMPT_VERSION: Final = "judge_v1"
JudgeLabel = Literal["match", "mismatch", "abstain", "asserted", "unclear"]


@dataclass(frozen=True)
class JudgeVerdict:
    """Parsed judge output; a parse failure is ``unclear``."""

    verdict: JudgeLabel
    raw: str


class Judge(Protocol):
    """Same-answer judge consulted only on rule-UNRESOLVED items."""

    id: str

    def judge(self, item: GroundTruthItem, response: str) -> JudgeVerdict:
        """Verdict for one (item, response)."""
        ...


class LocalHFJudge:
    """Local HF model (path or cached ``id@revision``), greedy, JSON-only; remote only with allow_remote."""

    def __init__(
        self,
        model: str,
        revision: str | None,
        *,
        prompt_version: str = JUDGE_PROMPT_VERSION,
        dtype: str = "auto",
        allow_remote: bool = False,
    ) -> None:
        self.model = model
        self.revision = revision
        self.prompt_version = prompt_version
        self.dtype = dtype
        self.allow_remote = allow_remote
        self.id = f"{model}@{revision or 'unpinned'}:{prompt_version}"

    def judge(self, item: GroundTruthItem, response: str) -> JudgeVerdict:
        """Prompt with ``resources/prompts/judge/<prompt_version>.yaml`` and parse ``{"verdict": ...}``."""
        todo("M9", "LocalHFJudge.judge")


Gates = GatesConfig
"""Pass thresholds per core bucket (plan §2.14); configured under ``eval.gates``."""

DEFAULT_GATES: Final = Gates()


@dataclass(frozen=True)
class ComponentValidation:
    """Agreement of one component with human labels (overall, per class and per stratum)."""

    component: str
    n: int
    kappa: float | None
    per_class: dict[str, dict[str, float]]
    false_match_rate: float | None
    per_stratum: dict[str, ComponentValidation] = field(default_factory=dict)
    passed: bool = False


@dataclass(frozen=True)
class ValidationReport:
    """All component validations; its sha goes into the protocol (``validation_report_sha256``)."""

    components: dict[str, ComponentValidation]
    labels_sha256: str
    sha256: str
    unvalidated_strata: dict[str, list[str]] = field(default_factory=dict)  # component -> failing buckets


class JudgeNotValidated(Exception):
    """Raised when the judge fails its gates (unless ``judge.force_unvalidated``, which flags it)."""


def validate_components(
    labels: Path,
    *,
    cfg: EvalConfig | None = None,
    suite: FactualReliabilitySuite | None = None,
    judge: Judge | None = None,
    gates: Gates = DEFAULT_GATES,
) -> ValidationReport:
    """Score gold-labelled responses with the suite built from ``cfg``; compare per stratum with gates."""
    todo("M7", "validate_components")


def assert_judge_usable(report: ValidationReport, gates: Gates = DEFAULT_GATES) -> None:
    """Raise :class:`JudgeNotValidated` if the judge component misses a gate."""
    todo("M7", "assert_judge_usable")
