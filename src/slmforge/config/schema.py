"""Training and evaluation configuration (pydantic v2).

Only pydantic and stdlib types here: this module must stay importable without torch,
transformers or unsloth, because the CLI resolves the training backend *before*
importing any of them (Unsloth patches transformers/peft on import).

Precedence: CLI flags > ``--set dotted.key=value`` > YAML > defaults. The fully resolved
config is written next to every output (see ``config.load.dump_resolved``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from slmforge.types import DEFAULT_DOMAINS, Language, Script

Dtype = Literal["auto", "bf16", "fp16", "fp32"]

DEFAULT_MAX_NEW_TOKENS: Final[Mapping[str, int]] = MappingProxyType(
    {"default": 128, "en": 96, "fr": 112, "darija_latn": 160, "darija_arab": 192, "darija_mixed": 192}
)
"""Per lang_script token budgets (Darija tokenizes into 2-4x more tokens than English)."""

DEFAULT_UTILITY_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "correct": 1.0,
        "correct_hedged": 1.0,
        "abstain_correct": 1.0,
        "abstain_accepted": 0.0,
        "non_answer_answerable": -0.25,
        "non_answer_unanswerable": 0.0,
        "abstain_unwarranted": -0.5,
        "wrong_hedged": -0.5,
        "wrong_low_conf": -0.75,
        "confabulation": -1.0,
        "invalid": -0.5,
    }
)
"""Utility weight per outcome (NON_ANSWER split by answerability; UNRESOLVED has none)."""


def _merge_over_defaults(
    v: Any, defaults: Mapping[str, Any], where: str, *, positive_int: bool = False
) -> Any:
    """Merge a partial mapping over its defaults and reject unknown keys and bad values.

    ``--set scoring.weights.confabulation=-2`` then changes one weight instead of
    replacing the whole mapping, and a typo such as ``confabulaton`` is an error.
    """
    if not isinstance(v, Mapping):
        return v  # let pydantic report the type error
    if unknown := sorted(set(v) - set(defaults)):
        raise ValueError(f"{where}: unknown key(s) {unknown}; expected a subset of {sorted(defaults)}")
    merged = {**defaults, **v}
    for key, value in merged.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{where}.{key} must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{where}.{key} must be finite, got {value!r}")
        if positive_int and (not float(value).is_integer() or value <= 0):
            raise ValueError(f"{where}.{key} must be a positive integer, got {value!r}")
    return merged


class _Strict(BaseModel):
    """Base for config sections: unknown keys are errors (catches typos), values are frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- training


class RunConfig(_Strict):
    """Run identity and seeding."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    output_dir: Path = Path("runs")
    seed: int = 1234
    data_seed: int | None = None  # None -> seed
    strict_determinism: bool = False  # torch.use_deterministic_algorithms(warn_only=not strict)


class ModelConfig(_Strict):
    """Base model to fine-tune or evaluate."""

    name_or_path: str
    revision: str | None = None  # pin a commit SHA for citable runs; branch names warn
    dtype: Dtype = "auto"
    max_seq_length: int = Field(2048, gt=0)
    chat_template: Literal["auto"] | Path = "auto"
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"  # recorded; affects numerics
    trust_remote_code: bool = False


class QuantConfig(_Strict):
    """4-bit QLoRA settings (bitsandbytes, CUDA only)."""

    load_in_4bit: bool = False
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True
    compute_dtype: Literal["bf16", "fp16"] = "bf16"


class LoraSpec(_Strict):
    """LoRA adapter hyper-parameters."""

    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.0, ge=0, lt=1)
    bias: Literal["none"] = "none"
    use_rslora: bool = False
    target_modules: list[str] | Literal["auto", "all-linear"] = "auto"


class DataSpec(_Strict):
    """One training data file."""

    path: Path
    format: Literal["auto", "chat", "prompt_response", "alpaca"] = "auto"
    weight: float = Field(1.0, gt=0)


class AbstentionMix(_Strict):
    """Unanswerable prompts paired with the locale's canonical abstention, mixed into training."""

    path: Path
    ratio: float = Field(0.05, ge=0, le=0.5)


class ContaminationGuard(_Strict):
    """Refuse (or drop/warn on) training records that overlap eval prompts."""

    ground_truth: Path | None = None  # defaults to eval.ground_truth
    policy: Literal["error", "drop", "warn"] = "error"
    ngram_n: int = 8
    ngram_threshold: float = Field(0.5, ge=0, le=1)
    jaccard_threshold: float = Field(0.5, ge=0, le=1)  # near-duplicate (MinHash) threshold, M8


class DataConfig(_Strict):
    """Training data and language balancing."""

    train: list[DataSpec] = Field(min_length=1)
    eval_loss: Path | None = None  # held-out chat data for eval_loss (needed for the TRADEOFF flag)
    completion_only_loss: bool = True  # labels = -100 outside assistant turns
    max_samples: int | None = None
    language_sampling_alpha: float = Field(1.0, gt=0, le=1)  # p_l ∝ n_l^alpha; < 1 upsamples Darija
    sampling_key: Literal["language", "lang_script"] = "lang_script"
    abstention_mix: AbstentionMix | None = None
    contamination_guard: ContaminationGuard = ContaminationGuard()


class TrainConfig(_Strict):
    """Trainer settings shared by both backends."""

    backend: Literal["auto", "unsloth", "hf_peft"] = "auto"  # auto: unsloth iff CUDA + importable
    max_steps: int | None = Field(None, gt=0)
    epochs: float | None = Field(None, gt=0)
    per_device_batch_size: int = Field(2, gt=0)
    grad_accum: int = Field(8, gt=0)
    lr: float = Field(2e-4, gt=0)
    scheduler: str = "cosine"
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    gradient_checkpointing: bool = True
    save_steps: int = Field(100, gt=0)
    save_total_limit: int | None = None  # keep every adapter for the trend
    logging_steps: int = 10
    eval_loss_steps: int | None = None  # None -> save_steps

    @model_validator(mode="after")
    def _steps_or_epochs(self) -> TrainConfig:
        if (self.max_steps is None) == (self.epochs is None):
            raise ValueError("set exactly one of train.max_steps or train.epochs")
        return self


# --------------------------------------------------------------------------- evaluation


class ScheduleConfig(_Strict):
    """When checkpoints are evaluated."""

    mode: Literal["posthoc", "inline"] = "posthoc"  # posthoc: separate process after/while training
    every_n_steps: int | None = None  # must be a multiple of train.save_steps
    include_base: bool = True
    include_final: bool = True
    after_training: bool = True


class SubsampleConfig(_Strict):
    """Fixed cluster-level subsample, identical at every evaluated checkpoint (trend items)."""

    fraction: float | None = Field(None, gt=0, le=1)  # None -> full set everywhere
    min_per_stratum: int = 20
    seed: int = 7
    full_at: list[Literal["base", "final"]] = ["base", "final"]  # extra full-set endpoint tables


class GenerationConfig(_Strict):
    """Greedy decoding for eval. A fresh GenerationConfig is built; shipped defaults never leak in."""

    backend: Literal["hf", "vllm", "replay", "fake"] = "hf"
    prompt_template: str = "default_v1"
    prompt_mode: Literal["abstain_ok", "neutral"] = "abstain_ok"
    batch_size: int = Field(16, gt=0)
    dtype: Dtype = "auto"
    quantization: Literal["match_training", "none", "4bit", "8bit"] = "match_training"
    device: str = "auto"
    base_model_override: str | None = None
    max_new_tokens: dict[str, int] = dict(DEFAULT_MAX_NEW_TOKENS)  # partial mappings merge over these
    stop_strings: list[str] = []
    cache: bool = True

    @field_validator("max_new_tokens", mode="before")
    @classmethod
    def _merge_max_new_tokens(cls, v: Any) -> Any:
        return _merge_over_defaults(v, DEFAULT_MAX_NEW_TOKENS, "generation.max_new_tokens", positive_int=True)


class MatchSection(_Strict):
    """Matcher switches; mirrors ``eval.match.MatchConfig``."""

    accept_locale_ambiguity: bool = True  # "1,500" in fr/en: accept either reading, flag it
    darija_money_scale: bool = True  # colloquial "melyoun" = 10,000 DH reading
    colloquial_units: bool = True  # ryal = 0.05 DH, centime = 0.01 DH
    containment_min_chars: int = 3
    contrast_window_tokens: int = 3
    skeleton_enabled: bool = True
    skeleton_threshold: float = 0.85
    skeleton_review_band: float = 0.70


class ConfidenceConfig(_Strict):
    """Answer-span confidence from a teacher-forced pass; tau is fixed per protocol."""

    tau: float = Field(0.5, ge=0, le=1)
    tau_source: Literal["fixed", "base_dev"] = (
        "fixed"  # base_dev: calibrate on the base model's dev split, then lock
    )
    tau_sweep: list[float] = [0.0, 0.3, 0.5, 0.7, 0.9]
    first_k: int = 32


class ScoringConfig(_Strict):
    """Everything that changes per-item scoring (hashed into the protocol)."""

    lexicon_version: str = "v1"
    extra_lexicons: list[Path] = []
    matching: MatchSection = MatchSection()
    min_content_tokens: int = 4
    unanswerable_no_cue_policy: Literal["residual", "unresolved"] = "residual"
    confidence: ConfidenceConfig = ConfidenceConfig()
    unresolved_flag_threshold: float = 0.05
    weights: dict[str, float] = dict(DEFAULT_UTILITY_WEIGHTS)  # partial mappings merge over these

    @field_validator("weights", mode="before")
    @classmethod
    def _merge_weights(cls, v: Any) -> Any:
        return _merge_over_defaults(v, DEFAULT_UTILITY_WEIGHTS, "scoring.weights")


class StatsConfig(_Strict):
    """Uncertainty and checkpoint comparison."""

    bootstrap_resamples: int = 2000
    alpha: float = 0.05
    cluster_strata: list[Literal["answerable", "domain"]] = ["answerable", "domain"]
    reference: str = "base"
    primary_metrics: list[str] = ["factual_accuracy", "confabulation_rate", "abstention_rate"]
    headline: Literal["macro", "micro"] = "macro"
    min_bucket_n: int = 30


class FlagsConfig(_Strict):
    """Thresholds for checkpoint flags (see docs/metrics.md)."""

    hedge_inflation_pts: float = 0.10
    over_abstention_pts: float = 0.10
    degeneration_pts: float = 0.05
    language_drift_pts: float = 0.10
    format_compliance_min: float = 0.80
    stale_months: int = 18
    tradeoff_signal: Literal["eval_loss", "chrf_pp"] = "eval_loss"


class GatesConfig(_Strict):
    """Detector/judge validation gates per core bucket (``validate-detectors``, M7)."""

    abstain_precision: float = 0.95
    abstain_recall: float = 0.90
    hedge_pr: float = 0.85  # precision and recall
    matcher_agreement: float = 0.97  # decisive verdicts, non-exact subset
    residual_precision: float = 0.90  # failing strata switch to the "unresolved" no-cue policy
    min_kappa: float = 0.75  # judge, overall
    min_stratum_kappa: float = 0.60  # judge, per bucket
    max_false_match: float = 0.05  # judge


class JudgeConfig(_Strict):
    """Optional local judge, consulted only on rule-UNRESOLVED items after validation (M9)."""

    enabled: bool = False
    model: str | None = None
    revision: str | None = None
    allow_remote: bool = False
    validation_report: Path | None = None
    force_unvalidated: bool = False


class EvalConfig(_Strict):
    """Evaluation config; usable standalone (``configs/eval/*.yaml``) or under ``eval:`` in a run config."""

    schema_version: Literal[1] = 1
    suite: list[Literal["factual-reliability", "task"]] = ["factual-reliability"]
    ground_truth: Path
    split: Literal["dev", "test", "all"] = "test"
    require_reviewed: bool = False
    allowed_domains: list[str] = list(DEFAULT_DOMAINS)
    protocol_lock: Path | None = None
    allow_unlocked: bool = True  # set false for published test-split results (M7)
    offline: bool = True
    seed: int = 1234
    schedule: ScheduleConfig = ScheduleConfig()
    subsample: SubsampleConfig = SubsampleConfig()
    generation: GenerationConfig = GenerationConfig()
    scoring: ScoringConfig = ScoringConfig()
    stats: StatsConfig = StatsConfig()
    flags: FlagsConfig = FlagsConfig()
    gates: GatesConfig = GatesConfig()
    judge: JudgeConfig = JudgeConfig()
    verdict_cache: Path = Path("runs/_shared/verdicts.sqlite")
    report_formats: list[Literal["md", "html"]] = ["md", "html"]


# --------------------------------------------------------------------------- distillation


class DistillTarget(_Strict):
    """A language/script to generate distillation data for."""

    language: Language
    script: Script | None = None


class DistillConfig(_Strict):
    """Teacher-model distillation. Off by default; two explicit opt-ins are required (M9)."""

    enabled: bool = False
    provider: Literal["anthropic", "openai"]
    model: str
    tos_acknowledged: bool = False  # you reviewed the provider's terms (output use, competing models)
    terms_reviewed_on: date | None = None
    acknowledged_by: str | None = None
    allow_send_data: bool = False  # prompts leave the machine (Moroccan Law 09-08 may apply to personal data)
    input: Path
    output: Path
    prompt_template: Path
    targets: list[DistillTarget] = []
    max_requests: int = 1000
    rate_limit_rps: float = Field(2.0, gt=0)
    temperature: float = 0.7
    redact_pii: bool = True
    require_langid_match: bool = True  # drop teacher outputs not in the target language/script
    exclude_overlap_with: Path | None = None

    @model_validator(mode="after")
    def _gates(self) -> DistillConfig:
        if self.enabled and not (self.tos_acknowledged and self.terms_reviewed_on and self.allow_send_data):
            raise ValueError(
                "distill.enabled requires tos_acknowledged, terms_reviewed_on and allow_send_data "
                "(see docs/distillation-terms.md)"
            )
        return self


# --------------------------------------------------------------------------- root


class RootConfig(_Strict):
    """A complete fine-tuning run (``slm-forge finetune --config``)."""

    schema_version: Literal[1] = 1
    run: RunConfig
    model: ModelConfig
    quant: QuantConfig = QuantConfig()
    lora: LoraSpec = LoraSpec()
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig | None = None
    distill: DistillConfig | None = None

    @model_validator(mode="after")
    def _cross_section(self) -> RootConfig:
        if self.eval is not None:
            every = self.eval.schedule.every_n_steps
            if every is not None and every % self.train.save_steps != 0:
                raise ValueError(
                    f"eval.schedule.every_n_steps ({every}) must be a multiple of train.save_steps "
                    f"({self.train.save_steps})"
                )
        return self
