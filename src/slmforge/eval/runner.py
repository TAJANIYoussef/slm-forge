"""Evaluation orchestration: cached generation -> scoring -> statistics -> ``metrics.json`` and reports.

Layout (plan §6.1): a run's evals go to ``runs/<name>/eval/<suite>/<protocol8>/`` with one
directory per checkpoint (``base/``, ``checkpoint-<step>/``) holding ``generations.jsonl``,
``scored.jsonl`` and ``metrics.json``, plus ``run_metrics.json``, ``trend.csv`` and the
reports. A standalone ``eval`` writes ``eval_out/<ckpt>/<suite>/<protocol8>/`` and never
writes into the checkpoint or the HF cache.

Rules the runner enforces:

* Every evaluated checkpoint, base and final included, is scored on the **same** fixed
  subsample (``subsample_ids.json``) for the trend; full-set results at ``full_at``
  endpoints are separate tables (critique 8).
* The base model is loaded once per eval-run and adapters are hot-swapped.
* Generations are cached by content hash (``eval.generate.cache_key``); rescoring and tau
  sweeps never regenerate.
* A fake generator (``SLMFORGE_FAKE_RESPONSES``, honoured only with ``generation.backend:
  fake``) is stamped ``is_fake`` in ``metrics.json`` and never writes into a run directory
  (critique 21).
* ``metrics.json`` carries four identity hashes: training config, protocol, dataset and
  generator fingerprint (critique 18), plus the provenance and effective decoding config.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from slmforge._stub import todo
from slmforge.config.schema import EvalConfig, FlagsConfig, StatsConfig
from slmforge.eval.factual_reliability import MetricValue, ReliabilitySummary
from slmforge.eval.outcomes import ItemResult, Outcome
from slmforge.eval.stats import PairedComparison
from slmforge.protocol import Protocol
from slmforge.repro import Provenance

if TYPE_CHECKING:
    from slmforge.eval.checkpoints import CheckpointRef
    from slmforge.eval.generate import Generator

METRICS_SCHEMA_VERSION: Final = 1
FAKE_RESPONSES_ENV: Final = "SLMFORGE_FAKE_RESPONSES"
GENERATIONS_FILE: Final = "generations.jsonl"
SCORED_FILE: Final = "scored.jsonl"
METRICS_FILE: Final = "metrics.json"
RUN_METRICS_FILE: Final = "run_metrics.json"
SUBSAMPLE_FILE: Final = "subsample_ids.json"
PROTOCOL_FILE: Final = "protocol.json"
EVAL_CONFIG_FILE: Final = "eval_config.yaml"  # what ``slm-forge rescore DIR`` reloads

Metric = MetricValue
"""One metric in ``metrics.json`` (value, num, den, CIs, bounds, low_n); defined in factual_reliability."""

FlagCode = Literal[
    "REGRESSION",
    "BUCKET_REGRESSION",
    "TRADEOFF",
    "TAU_SENSITIVE",
    "HEDGE_INFLATION",
    "OVER_ABSTENTION",
    "DEGENERATION",
    "LANGUAGE_DRIFT",
    "RESOLUTION_INSUFFICIENT",
    "FORMAT_NONCOMPLIANT",
    "LOW_N",
    "MACRO_INCOMPLETE",
    "STALE_FACTS",
    "JUDGE_UNVALIDATED",
    "DETECTOR_UNVALIDATED",
    "SUBSAMPLED",
    "CONF_UNAVAILABLE",
    "CI_FEW_CLUSTERS",
    "ONSET_UNRESOLVED_SENSITIVE",
    "TRUNCATION_RISE",
    "UNRESOLVED_HIGH",
    "FAKE_GENERATOR",
]


class _Doc(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Flag(_Doc):
    """A raised checkpoint or run flag (thresholds in ``FlagsConfig``; see docs/metrics.md)."""

    code: FlagCode
    checkpoint: str
    metric: str | None = None
    bucket: str | None = None
    detail: dict[str, Any] = {}


class CheckpointInfo(_Doc):
    """``metrics.json`` -> ``checkpoint``: what was evaluated (None for external responses)."""

    name: str
    path: str | None
    step: int | None
    kind: Literal["base", "adapter", "full"]
    base_model: str
    base_revision: str | None
    fingerprint: str
    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None


class ProtocolStamp(_Doc):
    """``metrics.json`` -> ``protocol``: the protocol, its hash and whether it was checked against a lock."""

    sha256: str
    short: str
    locked: bool = False
    spec: Protocol


class RunInfo(_Doc):
    """``metrics.json`` -> ``run``: the training run this checkpoint belongs to, if any."""

    config_sha256: str | None = None
    seed: int
    run_dir: str | None = None


class SubsampleInfo(_Doc):
    """The fixed cluster-level subsample used for trend points."""

    fraction: float | None
    seed: int
    min_per_stratum: int
    ids_sha256: str | None
    is_full: bool


class DatasetInfo(_Doc):
    """``metrics.json`` -> ``dataset``."""

    path: str | None
    sha256: str
    split: str
    n_total: int
    n_scored: int
    excluded_contaminated: int = 0
    subsample: SubsampleInfo | None = None


class GenerationInfo(_Doc):
    """``metrics.json`` -> ``generation``: backend identity and the effective decoding config."""

    backend: Literal["hf", "vllm", "replay", "fake", "external"]
    generator_fingerprint: str
    is_fake: bool = False
    effective_generation_config: dict[str, Any] = {}
    batch_size: int | None = None
    dtype: str | None = None
    quantization: str | None = None
    device: str | None = None
    finish_reasons: dict[str, int] = {}
    n_cached: int = 0


class Timing(_Doc):
    """``metrics.json`` -> ``timing`` (excluded from the golden-file canonical projection)."""

    load_s: float = 0.0
    generation_s: float = 0.0
    scoring_s: float = 0.0
    items_per_s: float | None = None
    new_tokens: int = 0


class CheckpointMetrics(_Doc):
    """Exactly the ``metrics.json`` document (plan §6.2)."""

    schema_version: Literal[1] = 1
    suite: str
    created_at: str
    checkpoint: CheckpointInfo | None
    protocol: ProtocolStamp
    run: RunInfo
    dataset: DatasetInfo
    generation: GenerationInfo
    provenance: Provenance
    reliability: ReliabilitySummary
    reliability_subsample: ReliabilitySummary | None = (
        None  # endpoint checkpoints recomputed on the subsample
    )
    detector_validation: dict[str, Any] = {}  # {"report_sha256": ..., "unvalidated_strata": [...]}
    contamination: dict[str, Any] = {}  # {"policy": ..., "n_flagged": ...}
    task: dict[str, float | None] = {}  # eval_loss, chrf_pp (the "task" suite)
    flags: list[Flag] = []
    timing: Timing = Timing()


class TrendPoint(_Doc):
    """One checkpoint's value of one metric in one bucket (``run_metrics.json`` trend)."""

    checkpoint: str
    step: int | None
    value: float | None
    ci95: tuple[float | None, float | None] | None = None
    bounds: tuple[float | None, float | None] | None = None
    den: int = 0


class CostEstimate(_Doc):
    """Dry-run estimate (plan §4.5): generations, token upper bounds and, with a throughput probe, seconds."""

    n_checkpoints: int
    n_generations: int
    est_prompt_tokens: int
    est_new_tokens: int
    est_seconds: float | None = None
    n_cached: int = 0


class RunMetrics(_Doc):
    """``run_metrics.json``: trend, comparisons, flags, onset, best checkpoint, flips, train-log join."""

    schema_version: Literal[1] = 1
    suite: str
    protocol_sha256: str
    reference: str
    steps: list[int | None]
    checkpoints: list[str]  # metrics.json paths relative to the protocol dir, in step order
    trend: dict[
        str, dict[str, list[TrendPoint]]
    ] = {}  # metric -> bucket ("micro", "macro", lang_script) -> points
    comparisons: list[PairedComparison] = []
    flags: list[Flag] = []
    degradation_onset: dict[str, int | None] = {}
    best_checkpoint: str | None = None
    flip_matrices: dict[str, dict[str, dict[str, int]]] = {}  # "<ref>-><other>" -> from -> to -> count
    flip_examples: dict[str, list[dict[str, Any]]] = {}
    train_log: dict[int, dict[str, float]] = {}
    seeds: dict[str, Any] | None = None


# --------------------------------------------------------------------------- runner


class EvalRunner:
    """Orchestrates generation (cached), scoring, statistics and reports for one eval config."""

    def __init__(
        self, cfg: EvalConfig, *, generator_factory: Callable[[CheckpointRef], Generator] | None = None
    ) -> None:
        self.cfg = cfg
        self.generator_factory = generator_factory
        self._protocol: Protocol | None = None

    @property
    def protocol(self) -> Protocol:
        """Built lazily from the config and ground truth; checked against ``cfg.protocol_lock`` if set."""
        todo("M2", "EvalRunner.protocol")

    def eval_checkpoint(
        self,
        ckpt: CheckpointRef,
        *,
        out_dir: Path,
        generator: Generator | None = None,
        full: bool | None = None,
    ) -> CheckpointMetrics:
        """Evaluate one checkpoint: cached generation, scoring, summary, ``metrics.json``."""
        todo("M2", "EvalRunner.eval_checkpoint")

    def eval_run(
        self,
        run_dir: Path,
        *,
        steps: Sequence[int] | None = None,
        watch: bool = False,
        poll_s: int = 60,
        resume: bool = True,
    ) -> RunMetrics:
        """Base once + adapter hot-swap over a run's scheduled checkpoints, then the trend report."""
        todo("M3", "EvalRunner.eval_run")

    def score_external(self, responses: Path, *, out_dir: Path) -> CheckpointMetrics:
        """Score an ``{item_id, response}`` JSONL from any pipeline (no model, no torch)."""
        todo("M2", "EvalRunner.score_external")

    def rescore(self, eval_dir: Path, *, tau: float | None = None) -> CheckpointMetrics:
        """CPU-only rescore from ``generations.jsonl`` (lexicons, tau, human verdicts); never regenerates."""
        todo("M2", "EvalRunner.rescore")

    def estimate_cost(self, n_checkpoints: int, *, generator: Generator | None = None) -> CostEstimate:
        """Plan §4.5 cost model: generations, token upper bounds, seconds from a cached throughput probe."""
        todo("M3", "EvalRunner.estimate_cost")


def default_generator_factory(cfg: EvalConfig) -> Callable[[CheckpointRef], Generator]:
    """Generator per ``generation.backend``: hf, vllm (M9), replay or fake (needs SLMFORGE_FAKE_RESPONSES)."""
    todo("M2", "default_generator_factory")


def refuse_fake_in_run_dir(generator: Generator, out_dir: Path) -> None:
    """Raise if a fake generator would write under a training run dir (one with ``run_manifest.json``)."""
    todo("M2", "refuse_fake_in_run_dir")


def write_metrics(metrics: CheckpointMetrics, path: Path) -> Path:
    """Write ``metrics.json`` (UTF-8, sorted keys, stable float formatting)."""
    todo("M2", "write_metrics")


def read_metrics(path: Path) -> CheckpointMetrics:
    """Load and validate a ``metrics.json``."""
    todo("M2", "read_metrics")


def metrics_json_schema() -> dict[str, Any]:
    """JSON Schema of ``metrics.json`` (exported to docs and validated in tests)."""
    todo("M2", "metrics_json_schema")


# --------------------------------------------------------------------------- comparisons and flags


def compare_checkpoints(
    per_ckpt: Mapping[str, Sequence[ItemResult]],
    *,
    reference: str,
    steps: Mapping[str, int | None],
    stats: StatsConfig,
) -> list[PairedComparison]:
    """Paired cluster bootstrap vs the reference + Holm per primary metric; bucket comparisons exploratory."""
    todo("M5", "compare_checkpoints")


def compute_flags(
    per_ckpt: Mapping[str, Sequence[ItemResult]],
    comps: Sequence[PairedComparison],
    *,
    train_log: Mapping[int, Mapping[str, float]],
    cfg: FlagsConfig,
    stats: StatsConfig,
) -> list[Flag]:
    """All checkpoint flags of plan §2.12 (regression, tradeoff, hedge inflation, drift, truncation, ...)."""
    todo("M5", "compute_flags")


def degradation_onset(comps: Sequence[PairedComparison], metric: str) -> int | None:
    """First step whose harmful change is Holm-rejected on both UNRESOLVED bounds (else None)."""
    todo("M5", "degradation_onset")


def flip_matrix(ref: Sequence[ItemResult], other: Sequence[ItemResult]) -> dict[str, dict[str, int]]:
    """Outcome transitions ``ref outcome -> other outcome`` over shared item ids."""
    todo("M5", "flip_matrix")


def flip_examples(
    ref: Sequence[ItemResult],
    other: Sequence[ItemResult],
    *,
    transitions: Sequence[tuple[Outcome, Outcome]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Example items for the given transitions (e.g. CORRECT -> CONFABULATION)."""
    todo("M5", "flip_examples")


def aggregate_seeds(runs: Sequence[RunMetrics]) -> dict[str, Any]:
    """Mean and between-seed SD per step and metric (the bootstrap does not cover training variance)."""
    todo("M5", "aggregate_seeds")
