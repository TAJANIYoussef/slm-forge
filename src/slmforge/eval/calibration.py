"""Confidence calibration of answered answerable items (plan §2.13, critique 11).

Eligible items: answerable, outcome in ANSWERED, ``confidence is not None`` and a decisive
verdict; ``correct := outcome in CORRECTS``. Numbers are comparable only within one
tokenizer/base model, because span confidence depends on tokenization.

* ``ECE = sum_b (n_b / n) |acc_b - conf_b|`` over 10 equal-mass bins (``np.array_split`` of
  the confidence-sorted items). Binned ECE is biased upward on finite samples.
* ``Brier = mean((conf - correct)²)``; ``AUROC`` = Mann-Whitney U / (n_pos n_neg), ties 0.5.
* ``AURC = (1/n) sum_k (errors in the top-k by confidence) / k``.
* ``overconfidence_gap = mean(conf) - selective_accuracy``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from slmforge._stub import todo

if TYPE_CHECKING:
    import numpy as np

    from slmforge.eval.outcomes import ItemResult


def ece_equal_mass(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float | None:
    """Expected calibration error over equal-mass bins; ``None`` on empty input."""
    todo("M5", "ece_equal_mass")


def brier(conf: np.ndarray, correct: np.ndarray) -> float | None:
    """Mean squared error between confidence and correctness."""
    todo("M5", "brier")


def auroc(conf: np.ndarray, correct: np.ndarray) -> float | None:
    """Mann-Whitney AUROC of confidence for correctness, ties counted 0.5; ``None`` without both classes."""
    todo("M5", "auroc")


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Risk (error share) vs coverage when answering the top-k items by confidence."""
    todo("M5", "risk_coverage")


def aurc(conf: np.ndarray, correct: np.ndarray) -> float | None:
    """Area under the risk-coverage curve."""
    todo("M5", "aurc")


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability-diagram bins: ``{n, conf_mean, acc, conf_low, conf_high}`` per equal-mass bin."""
    todo("M5", "reliability_bins")


def calibration_summary(results: Sequence[ItemResult]) -> dict[str, Any]:
    """ECE, Brier, AUROC, AURC, overconfidence gap, n and bins for the eligible items."""
    todo("M5", "calibration_summary")
