"""Uncertainty and checkpoint comparison (plan §2.11, critique 10/11/19).

* Per-checkpoint CIs: stratified **cluster** bootstrap. Clusters are ``fact_id`` (all
  translations/paraphrases of one fact move together); strata are ``(answerable, domain)``
  at cluster level. Indices come from ``PCG64(seed).random_raw()`` with our own reduction,
  so resamples are stable across NumPy versions.
* Checkpoint vs reference: the same resampled clusters applied to both checkpoints over
  their shared item ids; exact McNemar on discordant pairs as a secondary test.
* Multiplicity: Holm step-down within each primary metric's family of checkpoints.
* Wilson intervals are for display only; pass ``deff`` to use ``n_eff = n / DEFF``.

Everything here is pure stdlib + NumPy; SciPy is a test oracle only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Final

from slmforge._stub import todo

if TYPE_CHECKING:
    import numpy as np

    from slmforge.eval.outcomes import ItemResult

Z95: Final = 1.959964


@dataclass(frozen=True)
class CI:
    """Bootstrap interval; resamples with a zero denominator are dropped and counted."""

    point: float | None
    low: float | None
    high: float | None
    n_resamples: int
    n_degenerate: int = 0


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar on discordant pairs: ``b01`` = ref no / other yes, ``b10`` = ref yes / other no."""

    b01: int
    b10: int
    p_value: float


@dataclass(frozen=True)
class PairedComparison:
    """One checkpoint vs the reference on one metric (``bucket=None`` is the headline)."""

    metric: str
    bucket: str | None
    ref: str
    other: str
    step: int | None
    delta: float | None
    ci: CI
    p_value: float
    holm_reject: bool | None  # None for exploratory (unadjusted) comparisons
    n_items: int
    mcnemar: McNemarResult | None = None


def cluster_bootstrap_indices(
    clusters: Sequence[str], strata: Sequence[str], *, n_resamples: int, seed: int
) -> Iterator[np.ndarray]:
    """Resample clusters with replacement within each stratum; yield the item indices of each resample."""
    todo("M5", "cluster_bootstrap_indices")


def bootstrap_ci(
    results: Sequence[ItemResult],
    fn: Callable[[Sequence[ItemResult]], float | None],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int,
    strata_keys: Sequence[str] = ("answerable", "domain"),
) -> CI:
    """Stratified cluster-bootstrap percentile CI of ``fn``."""
    todo("M5", "bootstrap_ci")


def paired_bootstrap(
    ref: Sequence[ItemResult],
    other: Sequence[ItemResult],
    fn: Callable[[Sequence[ItemResult]], float | None],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int,
) -> tuple[float | None, CI, float]:
    """Delta ``fn(other) - fn(ref)``, its CI and a two-sided p on shared item ids with shared resamples."""
    todo("M5", "paired_bootstrap")


def mcnemar_from_counts(b01: int, b10: int) -> McNemarResult:
    """Exact two-sided McNemar: binomial(n = b01 + b10, 1/2) on the smaller discordant count."""
    if b01 < 0 or b10 < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b01 + b10
    if n == 0:
        return McNemarResult(b01, b10, 1.0)
    k = min(b01, b10)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2**n)
    return McNemarResult(b01, b10, float(min(Fraction(1), 2 * tail)))


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> McNemarResult:
    """Exact two-sided McNemar test on paired binary indicators (``a`` = reference, ``b`` = other)."""
    if len(a) != len(b):
        raise ValueError(f"paired sequences differ in length ({len(a)} != {len(b)})")
    b01 = sum(1 for x, y in zip(a, b, strict=True) if not x and y)
    b10 = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    return mcnemar_from_counts(b01, b10)


def holm(pvals: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm step-down: reject ``H_(i)`` iff ``p_(j) <= alpha / (m - j + 1)`` for every ``j <= i``.

    NaN p-values are never rejected; ties are ordered by key so the result is deterministic.
    """
    order = sorted(pvals, key=lambda k: (math.isnan(pvals[k]), pvals[k], k))
    m = len(order)
    out = dict.fromkeys(pvals, False)
    for j, key in enumerate(order):
        p = pvals[key]
        if math.isnan(p) or p > alpha / (m - j):
            break
        out[key] = True
    return out


def holm_adjusted(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values ``max_{j<=i} min(1, (m - j + 1) p_(j))`` (rejected iff <= alpha)."""
    order = sorted(pvals, key=lambda k: (math.isnan(pvals[k]), pvals[k], k))
    m = len(order)
    out: dict[str, float] = {}
    running = 0.0
    for j, key in enumerate(order):
        p = pvals[key]
        if math.isnan(p):
            out[key] = math.nan
            continue
        running = max(running, min(1.0, (m - j) * p))
        out[key] = running
    return {k: out[k] for k in pvals}


def wilson(k: float, n: float, z: float = Z95, *, deff: float = 1.0) -> tuple[float, float] | None:
    """Wilson score interval for ``k`` successes in ``n``; ``deff > 1`` uses ``n_eff = n / deff``.

    ``center = (p + z²/2n) / (1 + z²/n)``, ``half = z sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)``.
    Returns ``None`` when ``n == 0``.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if deff <= 0:
        raise ValueError(f"deff must be positive, got {deff}")
    if n == 0:
        return None
    p = k / n
    n_eff = n / deff
    z2 = z * z
    denom = 1 + z2 / n_eff
    center = (p + z2 / (2 * n_eff)) / denom
    half = z * math.sqrt(p * (1 - p) / n_eff + z2 / (4 * n_eff * n_eff)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def icc_binary(values: Sequence[bool], clusters: Sequence[str]) -> float:
    """One-way ANOVA ICC of a binary outcome by cluster (``fact_id``)."""
    todo("M5", "icc_binary")


def design_effect(cluster_sizes: Sequence[int], icc: float) -> float:
    """Kish design effect ``1 + (mbar - 1) * icc`` with ``mbar`` the mean cluster size."""
    if not cluster_sizes:
        raise ValueError("design_effect needs at least one cluster")
    if any(s <= 0 for s in cluster_sizes):
        raise ValueError("cluster sizes must be positive")
    mbar = sum(cluster_sizes) / len(cluster_sizes)
    return 1.0 + (mbar - 1.0) * icc


def mde_paired(n: int, discordance: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Connor's paired minimum detectable difference for ``n`` pairs and discordance ``psi``."""
    todo("M5", "mde_paired")
