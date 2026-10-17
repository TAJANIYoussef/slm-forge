"""Factual-reliability metrics and the suite that produces them (plan §2.9-§2.10, critique 1/6/7).

Every rate is one row of the declarative table :data:`METRICS`: a numerator (an outcome set
and/or a named item predicate) counted **within** a denominator (a named item filter and
optionally an outcome set). So ``0 <= num <= den`` always holds and ``metrics.json`` carries
``num``/``den`` next to each value, plus the table itself (:func:`metrics_table`).

Conventions:

* INVALID and UNRESOLVED stay in every denominator they belong to: an unusable or
  undecidable answer to an answerable item counts against accuracy, and a checkpoint that
  rambles cannot look better by dropping items.
* A zero denominator gives ``None``, never 0.
* Primary (pre-registered, Holm-corrected): ``factual_accuracy`` = CORRECTS / answerable,
  ``confabulation_rate`` = CONFABULATION / all scored, ``abstention_rate`` =
  ABSTAIN_CORRECT / unanswerable.
* Confidence-gated rates (those that separate CONFABULATION from WRONG_LOW_CONF) are
  withheld (``value is None``, ``withheld == "CONF_UNAVAILABLE"``) when an unhedged error
  had no confidence, because the tau gate could not be applied. ``unhedged_error_rate`` is
  tau- and tokenizer-free and is always reported; it is the headline trend, since
  fine-tuning sharpens confidence and moves confab(tau) even when errors do not.
* ``confabulation_rate = error_rate * p_unhedged_given_error * p_confident_given_unhedged_error``.
* UNRESOLVED bounds substitute each UNRESOLVED item's best or worst case, whichever
  minimizes or maximizes the metric (:func:`compute_bounds`).
"""

from __future__ import annotations

import dataclasses
import math
import operator
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict

from slmforge._stub import todo
from slmforge.config.schema import DEFAULT_UTILITY_WEIGHTS, ScoringConfig, StatsConfig
from slmforge.eval.outcomes import ABSTAINED, ANSWERED, CORRECTS, WRONG, ItemResult, Outcome, reclassify
from slmforge.types import CORE_BUCKETS

if TYPE_CHECKING:
    from slmforge.eval.dataset import GroundTruthItem
    from slmforge.eval.generate import GenerationRecord
    from slmforge.eval.judge import Judge
    from slmforge.eval.review import VerdictCache
    from slmforge.protocol import Protocol
    from slmforge.text.langid import LangIdBackend
    from slmforge.text.lexicons import LexiconSet
    from slmforge.text.numbers import UnitTable

SCORER_VERSION: Final = "1.1.0"
CONF_UNAVAILABLE: Final = "CONF_UNAVAILABLE"
DARIJA_SCRIPT_BUCKETS: Final[tuple[str, ...]] = ("darija_arab", "darija_latn")
UNHEDGED_WRONG: Final = frozenset({Outcome.CONFABULATION, Outcome.WRONG_LOW_CONF})


# --------------------------------------------------------------------------- ratio and table


@dataclass(frozen=True)
class Ratio:
    """Numerator/denominator pair; ``value`` is ``None`` on a zero denominator or when withheld."""

    num: float
    den: int
    withheld: str | None = None  # flag code explaining a withheld value (CONF_UNAVAILABLE)

    @property
    def value(self) -> float | None:
        """``num / den``, or ``None``."""
        if self.den == 0 or self.withheld is not None:
            return None
        return self.num / self.den

    def to_json(self) -> dict[str, Any]:
        """``{"value", "num", "den"}`` plus ``withheld`` when set."""
        d: dict[str, Any] = {"value": self.value, "num": self.num, "den": self.den}
        if self.withheld is not None:
            d["withheld"] = self.withheld
        return d


ItemFilter = Callable[[ItemResult], bool]

FILTERS: Final[Mapping[str, ItemFilter]] = MappingProxyType(
    {
        "all": lambda r: True,
        "answerable": lambda r: r.answerable,
        "unanswerable": lambda r: not r.answerable,
        "answerable_required": lambda r: r.answerable and not r.allow_abstention,
        "hedged": lambda r: r.hedged,
        "truncated": lambda r: r.evidence.finish_reason == "length",
        "final_line": lambda r: r.region_method == "final_line",
        "language_scored": lambda r: r.language_fidelity is not None,
        "language_faithful": lambda r: r.language_fidelity is True,
        "script_scored": lambda r: r.script_fidelity is not None and r.lang_script in DARIJA_SCRIPT_BUCKETS,
        "script_faithful": lambda r: r.script_fidelity is True,
    }
)
"""Named item predicates used by :class:`MetricSpec` (names appear in ``metrics.json``)."""


@dataclass(frozen=True)
class MetricSpec:
    """One row of the metric table: items matching the numerator, counted within the denominator."""

    name: str
    den_filter: str = "all"
    num_outcomes: frozenset[Outcome] | None = None  # None: any outcome
    num_filter: str | None = None
    den_outcomes: frozenset[Outcome] | None = None  # None: any outcome
    tier: Literal["primary", "secondary"] = "secondary"
    conf_gated: bool = False  # depends on the tau gate (CONFABULATION vs WRONG_LOW_CONF)
    description: str = ""

    def __post_init__(self) -> None:
        for f in (self.den_filter, self.num_filter):
            if f is not None and f not in FILTERS:
                raise ValueError(f"metric {self.name!r}: unknown filter {f!r}")

    def contribution(self, r: ItemResult, outcome: Outcome | None = None) -> tuple[int, int]:
        """``(num, den)`` contribution of one item, optionally as if its outcome were ``outcome``."""
        o = r.outcome if outcome is None else outcome
        if not FILTERS[self.den_filter](r) or (self.den_outcomes is not None and o not in self.den_outcomes):
            return (0, 0)
        hit = (self.num_outcomes is None or o in self.num_outcomes) and (
            self.num_filter is None or FILTERS[self.num_filter](r)
        )
        return (int(hit), 1)

    def point_contribution(self, r: ItemResult) -> tuple[int, int]:
        """Contribution to the point estimate.

        An UNRESOLVED item counts as UNRESOLVED (neither correct nor erroneous), unless its
        best and worst cases contribute the same, e.g. to ``coverage`` both are answered;
        then that common contribution is used, so the point always lies within the bounds.
        """
        if r.outcome is Outcome.UNRESOLVED and self.substitutes:
            best, worst = self.contribution(r, r.best_case), self.contribution(r, r.worst_case)
            if best == worst:
                return best
        return self.contribution(r)

    @property
    def substitutes(self) -> bool:
        """Whether UNRESOLVED items are substituted (not for metrics that count UNRESOLVED itself)."""
        return not any(
            s is not None and Outcome.UNRESOLVED in s for s in (self.num_outcomes, self.den_outcomes)
        )

    def ratio(self, results: Sequence[ItemResult], *, conf_missing: bool | None = None) -> Ratio:
        """Point estimate over ``results``; ``conf_missing`` short-cuts :func:`conf_unavailable`."""
        num = den = 0
        for r in results:
            n, d = self.point_contribution(r)
            num += n
            den += d
        if self.conf_gated and (conf_unavailable(results) if conf_missing is None else conf_missing):
            return Ratio(num, den, CONF_UNAVAILABLE)
        return Ratio(num, den)

    def __call__(self, results: Sequence[ItemResult]) -> Ratio:
        return self.ratio(results)

    def to_json(self) -> dict[str, Any]:
        """Declarative description for ``metrics.json`` and the protocol."""
        return {
            "tier": self.tier,
            "numerator_outcomes": None
            if self.num_outcomes is None
            else sorted(o.value for o in self.num_outcomes),
            "numerator_filter": self.num_filter,
            "denominator_filter": self.den_filter,
            "denominator_outcomes": None
            if self.den_outcomes is None
            else sorted(o.value for o in self.den_outcomes),
            "conf_gated": self.conf_gated,
            "description": self.description,
        }


# fmt: off
_O = Outcome
_SPECS: Final[tuple[MetricSpec, ...]] = (
    # primary
    MetricSpec("factual_accuracy", "answerable", CORRECTS, tier="primary",
               description="CORRECT + CORRECT_HEDGED over answerable items"),
    MetricSpec("confabulation_rate", "all", frozenset({_O.CONFABULATION}), tier="primary", conf_gated=True,
               description="confidently false (unhedged, conf >= tau) over all scored items"),
    MetricSpec("abstention_rate", "unanswerable", frozenset({_O.ABSTAIN_CORRECT}), tier="primary",
               description="abstained or rejected the premise, over unanswerable items"),
    # answering behaviour
    MetricSpec("selective_accuracy", "answerable", CORRECTS, den_outcomes=ANSWERED,
               description="correct among answered answerable items"),
    MetricSpec("coverage", "answerable", ANSWERED,
               description="answered share of answerable items"),
    MetricSpec("hedge_rate", "all", num_filter="hedged", den_outcomes=ANSWERED,
               description="hedged share of answered items"),
    # errors
    MetricSpec("unhedged_error_rate", "all", UNHEDGED_WRONG,
               description="CONFABULATION + WRONG_LOW_CONF over all (tau- and tokenizer-free)"),
    MetricSpec("error_rate", "all", WRONG,
               description="any wrong answer over all"),
    MetricSpec("confabulation_rate_answerable", "answerable", frozenset({_O.CONFABULATION}), conf_gated=True,
               description="CONFABULATION over answerable items"),
    MetricSpec("unanswerable_confabulation_rate", "unanswerable", frozenset({_O.CONFABULATION}),
               conf_gated=True, description="CONFABULATION over unanswerable items"),
    MetricSpec("confab_given_answered", "all", frozenset({_O.CONFABULATION}), den_outcomes=ANSWERED,
               conf_gated=True, description="CONFABULATION among answered items"),
    MetricSpec("confident_error_share", "all", frozenset({_O.CONFABULATION}), den_outcomes=WRONG,
               conf_gated=True, description="CONFABULATION among wrong answers"),
    MetricSpec("p_unhedged_given_error", "all", UNHEDGED_WRONG, den_outcomes=WRONG,
               description="unhedged share of wrong answers"),
    MetricSpec("p_confident_given_unhedged_error", "all", frozenset({_O.CONFABULATION}),
               den_outcomes=UNHEDGED_WRONG, conf_gated=True,
               description="conf >= tau share of unhedged wrong answers"),
    # abstention
    MetricSpec("raw_abstention_rate", "all", ABSTAINED,
               description="any abstention over all"),
    MetricSpec("abstention_precision", "all", frozenset({_O.ABSTAIN_CORRECT}), den_outcomes=ABSTAINED,
               description="abstentions that were on unanswerable items"),
    MetricSpec("over_abstention_rate", "answerable", frozenset({_O.ABSTAIN_ACCEPTED, _O.ABSTAIN_UNWARRANTED}),
               description="abstentions on answerable items"),
    MetricSpec("unwarranted_abstention_rate", "answerable_required", frozenset({_O.ABSTAIN_UNWARRANTED}),
               description="abstentions on required-knowledge items (allow_abstention=false)"),
    MetricSpec("non_answer_rate", "all", frozenset({_O.NON_ANSWER}),
               description="evasive non-answers over all"),
    # scoring health
    MetricSpec("invalid_rate", "all", frozenset({_O.INVALID}),
               description="INVALID over all"),
    MetricSpec("unresolved_rate", "all", frozenset({_O.UNRESOLVED}),
               description="UNRESOLVED over all"),
    MetricSpec("truncation_rate", "all", num_filter="truncated",
               description="finish_reason=length over all"),
    MetricSpec("format_compliance_rate", "all", num_filter="final_line",
               description="responses with a localized final-answer line"),
    MetricSpec("language_fidelity_rate", "language_scored", num_filter="language_faithful",
               description="response in the item's language, over responses with >= 3 content tokens"),
    MetricSpec("script_fidelity_rate", "script_scored", num_filter="script_faithful",
               description="response in the item's script, Darija arab/latn items only"),
)
# fmt: on

METRICS: Final[Mapping[str, MetricSpec]] = MappingProxyType({s.name: s for s in _SPECS})
"""Every counting metric of plan §2.9, in report order."""

PRIMARY_METRICS: Final = ("factual_accuracy", "confabulation_rate", "abstention_rate")
UTILITY_METRIC: Final = "reliability_utility"

# fmt: off
_HARMFUL: dict[str, Literal[1, -1]] = {
    "factual_accuracy": -1, "confabulation_rate": 1, "abstention_rate": -1,
    "selective_accuracy": -1, "coverage": -1, "unhedged_error_rate": 1, "error_rate": 1,
    "confabulation_rate_answerable": 1, "unanswerable_confabulation_rate": 1, "confab_given_answered": 1,
    "confident_error_share": 1, "p_unhedged_given_error": 1, "p_confident_given_unhedged_error": 1,
    "abstention_precision": -1, "over_abstention_rate": 1, "unwarranted_abstention_rate": 1,
    "non_answer_rate": 1, "invalid_rate": 1, "unresolved_rate": 1, "truncation_rate": 1,
    "format_compliance_rate": -1, "language_fidelity_rate": -1, "script_fidelity_rate": -1,
    UTILITY_METRIC: -1,
}
# fmt: on
HARMFUL_DIRECTION: Final[Mapping[str, Literal[1, -1]]] = MappingProxyType(_HARMFUL)
"""Sign of a harmful change (+1: an increase is harmful). raw_abstention_rate and hedge_rate have none."""

DEFAULT_WEIGHTS: Final[Mapping[str, float]] = DEFAULT_UTILITY_WEIGHTS
UTILITY_KEYS: Final[frozenset[str]] = frozenset(
    {o.value for o in Outcome if o not in (Outcome.NON_ANSWER, Outcome.UNRESOLVED)}
    | {"non_answer_answerable", "non_answer_unanswerable"}
)


def metrics_table() -> dict[str, dict[str, Any]]:
    """The metric definitions as data (for ``metrics.json`` and the protocol hash)."""
    return {name: spec.to_json() for name, spec in METRICS.items()}


# --------------------------------------------------------------------------- point estimates


def conf_unavailable(results: Sequence[ItemResult]) -> bool:
    """True when an item that is (or, if UNRESOLVED, could be) CONFABULATION has no confidence."""
    return any(
        r.evidence.confidence is None and Outcome.CONFABULATION in (r.outcome, r.worst_case) for r in results
    )


def outcome_counts(results: Sequence[ItemResult]) -> dict[str, int]:
    """Count per outcome, every outcome present (zero-filled), in taxonomy order."""
    counts = dict.fromkeys((o.value for o in Outcome), 0)
    for r in results:
        counts[r.outcome.value] += 1
    return counts


def compute_point_metrics(
    results: Sequence[ItemResult], *, weights: Mapping[str, float] | None = None
) -> dict[str, Ratio]:
    """Every :data:`METRICS` rate plus ``reliability_utility``, as point estimates."""
    missing = conf_unavailable(results)
    out = {name: spec.ratio(results, conf_missing=missing) for name, spec in METRICS.items()}
    out[UTILITY_METRIC] = utility_ratio(results, DEFAULT_WEIGHTS if weights is None else weights)
    return out


def hedge_correctness_gap(results: Sequence[ItemResult]) -> float | None:
    """``P(correct | answered, unhedged) - P(correct | answered, hedged)`` over answerable items."""
    groups: dict[bool, list[bool]] = {False: [], True: []}
    for r in results:
        if r.answerable and r.outcome in ANSWERED:
            groups[r.hedged].append(r.outcome in CORRECTS)
    if not groups[False] or not groups[True]:
        return None
    return sum(groups[False]) / len(groups[False]) - sum(groups[True]) / len(groups[True])


# --------------------------------------------------------------------------- utility


def _weight(weights: Mapping[str, float], r: ItemResult, outcome: Outcome) -> float:
    if outcome is Outcome.NON_ANSWER:
        key = "non_answer_answerable" if r.answerable else "non_answer_unanswerable"
        if key not in weights and "non_answer" in weights:
            key = "non_answer"
    else:
        key = outcome.value
    try:
        return float(weights[key])
    except KeyError:
        raise KeyError(
            f"utility weights have no entry for {key!r} (expected keys: {sorted(UTILITY_KEYS)})"
        ) from None


def utility_ratio(results: Sequence[ItemResult], weights: Mapping[str, float] = DEFAULT_WEIGHTS) -> Ratio:
    """Sum of outcome weights over resolved items, and their count (UNRESOLVED excluded)."""
    ws = [_weight(weights, r, r.outcome) for r in results if r.outcome is not Outcome.UNRESOLVED]
    return Ratio(math.fsum(ws), len(ws))


def utility(results: Sequence[ItemResult], weights: Mapping[str, float] = DEFAULT_WEIGHTS) -> float | None:
    """Mean outcome weight over resolved items (``reliability_utility``, range [-1, 1] by default)."""
    return utility_ratio(results, weights).value


# --------------------------------------------------------------------------- bounds and tau


def at_tau(results: Sequence[ItemResult], tau: float) -> list[ItemResult]:
    """Re-derive every outcome at ``tau`` from stored evidence (each item keeps its scoring policy)."""
    return [reclassify(r, tau=tau) for r in results]


# Ordering of (num, den) contributions by their effect on a ratio: adding (0, 1) never raises
# it, (0, 0) leaves it, (1, 1) never lowers it - whatever the other items are. So choosing per
# item by this rank reaches the global minimum/maximum over all best/worst substitutions.
_RANK: Final[Mapping[tuple[int, int], int]] = MappingProxyType({(0, 1): 0, (0, 0): 1, (1, 1): 2})


def compute_bounds(
    results: Sequence[ItemResult],
    metric: str,
    *,
    tau: float | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[float | None, float | None]:
    """Lower/upper value over every best/worst substitution of the UNRESOLVED items.

    ``tau`` re-derives outcomes first (:func:`at_tau`). For ``reliability_utility`` the
    substitution includes every item while the point estimate averages resolved items
    only, so the returned interval is widened to contain the point estimate.
    """
    rs = list(results) if tau is None else at_tau(results, tau)
    if metric == UTILITY_METRIC:
        w = DEFAULT_WEIGHTS if weights is None else weights
        lo: list[float] = []
        hi: list[float] = []
        for r in rs:
            opts = (r.best_case, r.worst_case) if r.outcome is Outcome.UNRESOLVED else (r.outcome,)
            vals = [_weight(w, r, o) for o in opts]
            lo.append(min(vals))
            hi.append(max(vals))
        low_v, high_v = Ratio(math.fsum(lo), len(lo)).value, Ratio(math.fsum(hi), len(hi)).value
        point = utility(rs, w)
        if point is not None and low_v is not None and high_v is not None:
            low_v, high_v = min(low_v, point), max(high_v, point)
        return (low_v, high_v)
    spec = _spec(metric)
    if spec.conf_gated and conf_unavailable(rs):
        return (None, None)
    lo_n = lo_d = hi_n = hi_d = 0
    any_hit_option = any_miss_option = False  # an UNRESOLVED option contributing (1, 1) / (0, 1)
    for r in rs:
        if r.outcome is Outcome.UNRESOLVED and spec.substitutes:
            contribs = (spec.contribution(r, r.best_case), spec.contribution(r, r.worst_case))
            low, high = min(contribs, key=_RANK.__getitem__), max(contribs, key=_RANK.__getitem__)
            any_hit_option |= (1, 1) in contribs
            any_miss_option |= (0, 1) in contribs
        else:
            low = high = spec.contribution(r)
        lo_n, lo_d, hi_n, hi_d = lo_n + low[0], lo_d + low[1], hi_n + high[0], hi_d + high[1]
    low_v, high_v = Ratio(lo_n, lo_d).value, Ratio(hi_n, hi_d).value
    # The rank order maximizes/minimizes the ratio but may leave the denominator empty. Then the
    # extreme over *defined* substitutions is reached by including a single forced option.
    if low_v is None and any_hit_option:
        low_v = 1.0  # every option that enters the denominator is a hit
    if high_v is None and any_miss_option:
        high_v = 0.0  # every option that enters the denominator is a miss
    return (low_v, high_v)


def tau_key(tau: float) -> str:
    """Canonical JSON key for a tau value (``0.0 -> "0"``, ``0.5 -> "0.5"``)."""
    return format(float(tau), "g")


def tau_sweep(results: Sequence[ItemResult], taus: Sequence[float]) -> dict[str, float | None]:
    """``confabulation_rate`` at each tau (keys from :func:`tau_key`); non-increasing in tau."""
    spec = METRICS["confabulation_rate"]
    return {tau_key(t): spec.ratio(at_tau(results, t)).value for t in taus}


# --------------------------------------------------------------------------- buckets and macro

_NOT_GROUPABLE: Final = frozenset({"evidence", "flags"})
GROUP_KEYS: Final[frozenset[str]] = frozenset(
    f.name for f in dataclasses.fields(ItemResult) if f.name not in _NOT_GROUPABLE
)


def _label(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return "none" if v is None else str(v)


def group_by(
    results: Sequence[ItemResult], key: str | Callable[[ItemResult], object]
) -> dict[str, list[ItemResult]]:
    """Bucket results by an ItemResult field (or a function); labels are strings, sorted.

    Booleans become ``"true"``/``"false"`` and ``None`` becomes ``"none"``.
    """
    get: Callable[[ItemResult], object]
    if isinstance(key, str):
        if key not in GROUP_KEYS:
            raise ValueError(f"cannot group by {key!r}; expected one of {sorted(GROUP_KEYS)}")
        get = operator.attrgetter(key)
    else:
        get = key
    out: dict[str, list[ItemResult]] = {}
    for r in results:
        out.setdefault(_label(get(r)), []).append(r)
    return dict(sorted(out.items()))


def _spec(metric: str) -> MetricSpec:
    try:
        return METRICS[metric]
    except KeyError:
        raise KeyError(f"unknown metric {metric!r}; expected one of {[*METRICS, UTILITY_METRIC]}") from None


def _metric_fn(
    metric: str | Callable[[Sequence[ItemResult]], Ratio], weights: Mapping[str, float] | None
) -> Callable[[Sequence[ItemResult]], Ratio]:
    if callable(metric):
        return metric
    if metric == UTILITY_METRIC:
        w = DEFAULT_WEIGHTS if weights is None else weights
        return lambda rs: utility_ratio(rs, w)
    return _spec(metric)


def macro(
    results: Sequence[ItemResult],
    metric: str | Callable[[Sequence[ItemResult]], Ratio],
    *,
    buckets: Sequence[str] = CORE_BUCKETS,
    min_n: int = 30,
    weights: Mapping[str, float] | None = None,
) -> tuple[float | None, list[str]]:
    """Unweighted mean over lang_script buckets whose denominator is >= ``min_n``; returns excluded buckets.

    The value is ``None`` when no bucket qualifies, or when the metric is withheld in any
    qualifying bucket (a partial confidence-gated macro would be misleading).
    """
    fn = _metric_fn(metric, weights)
    groups = group_by(results, "lang_script")
    values: list[float] = []
    excluded: list[str] = []
    withheld = False
    for b in buckets:
        ratio = fn(groups.get(b, []))
        if ratio.den < min_n or ratio.den == 0:
            excluded.append(b)
        elif ratio.withheld is not None:
            withheld = True
        else:
            values.append(ratio.num / ratio.den)
    if withheld or not values:
        return (None, excluded)
    return (math.fsum(values) / len(values), excluded)


def gaps(results: Sequence[ItemResult], metrics: Sequence[str] = PRIMARY_METRICS) -> dict[str, float | None]:
    """``darija_gap_<m>`` = m(fr+en pooled) - m(darija pooled); ``script_gap_<m>`` = m(arab) - m(latn)."""
    todo("M4", "gaps")


def cross_lingual(results: Sequence[ItemResult]) -> dict[str, Ratio]:
    """``consistency`` (same coarse label in every language) and ``confab_divergence`` over facts."""
    todo("M4", "cross_lingual")


# --------------------------------------------------------------------------- summary model


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Interval = tuple[float | None, float | None]


class MetricValue(_Out):
    """One micro or per-bucket metric as written to ``metrics.json``."""

    value: float | None
    num: float
    den: int
    withheld: str | None = None
    ci95: Interval | None = None
    wilson95: tuple[float, float] | None = None
    bounds: Interval | None = None
    low_n: bool = False

    @classmethod
    def from_ratio(cls, r: Ratio, **extra: Any) -> MetricValue:
        """Lift a :class:`Ratio`; ``extra`` sets the interval fields."""
        return cls(value=r.value, num=r.num, den=r.den, withheld=r.withheld, **extra)


class Estimate(_Out):
    """A derived value with an optional bootstrap CI (gaps)."""

    value: float | None
    ci95: Interval | None = None


class MacroValue(Estimate):
    """Macro (unweighted bucket mean) headline value."""

    buckets_used: list[str] = []
    excluded: list[str] = []


class MetricFlag(_Out):
    """A checkpoint flag (``LOW_N``, ``MACRO_INCOMPLETE``, ``CONF_UNAVAILABLE``, ...)."""

    code: str
    metric: str | None = None
    bucket: str | None = None
    detail: dict[str, Any] = {}


class ReliabilitySummary(_Out):
    """Aggregates for one checkpoint (embedded in ``metrics.json``)."""

    schema_version: Literal[1] = 1
    scorer_version: str = SCORER_VERSION
    tau: float
    n_scored: int
    is_subsample: bool = False
    outcome_counts: dict[str, int]
    micro: dict[str, MetricValue]
    macro: dict[str, MacroValue] = {}
    by_bucket: dict[str, dict[str, dict[str, MetricValue]]] = {}  # key -> bucket label -> metric
    gaps: dict[str, Estimate] = {}
    cross_lingual: dict[str, MetricValue] = {}
    calibration: dict[str, Any] | None = None
    tau_sweep: dict[str, float | None] = {}
    hedge_correctness_gap: float | None = None
    verdict_sources: dict[str, int] = {}
    match_flags: dict[str, int] = {}
    flags: list[MetricFlag] = []


# --------------------------------------------------------------------------- suite


class FactualReliabilitySuite:
    """The factual-reliability suite: per-item scoring and per-checkpoint summaries."""

    name: ClassVar[str] = "factual-reliability"

    def __init__(
        self,
        protocol: Protocol,
        cfg: ScoringConfig,
        *,
        lexicons: LexiconSet,
        units: UnitTable,
        langid: LangIdBackend,
        verdict_cache: VerdictCache | None = None,
        judge: Judge | None = None,
    ) -> None:
        self.protocol = protocol
        self.cfg = cfg
        self.lexicons = lexicons
        self.units = units
        self.langid = langid
        self.verdict_cache = verdict_cache
        self.judge = judge

    def score_item(self, item: GroundTruthItem, rec: GenerationRecord) -> ItemResult:
        """One item: clean, invalid check, signals, region, candidates, match, confidence, decide."""
        todo("M1", "FactualReliabilitySuite.score_item")

    def score(
        self, items: Sequence[GroundTruthItem], recs: Mapping[str, GenerationRecord]
    ) -> list[ItemResult]:
        """Score every item that has a record (a missing record is a generation_error INVALID)."""
        todo("M1", "FactualReliabilitySuite.score")

    def summarize(
        self,
        results: Sequence[ItemResult],
        *,
        stats: StatsConfig,
        subsample_ids: Collection[str] | None = None,
    ) -> ReliabilitySummary:
        """Micro/macro/bucket rates with CIs and bounds, calibration, tau sweep, gaps, flags.

        M1 fills point estimates, bounds, macro, tau sweep and flags that need no resampling;
        M5 adds the cluster-bootstrap CIs and calibration.
        """
        todo("M1", "FactualReliabilitySuite.summarize")
