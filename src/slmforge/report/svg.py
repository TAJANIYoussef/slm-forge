"""Dependency-free inline SVG charts for the HTML report (M5).

Every function returns an ``<svg>`` element string with a ``viewBox`` (it scales with the
page), a ``<title>`` for accessibility and colours from CSS custom properties defined by
the report, so light and dark themes both work. ``None`` values are gaps, never zeros.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from slmforge._stub import todo


def line_chart(
    series: Mapping[str, Sequence[tuple[int, float | None]]],
    *,
    bands: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    markers: Sequence[int] = (),
    y_label: str,
    width: int = 720,
    height: int = 280,
) -> str:
    """Trend per series (e.g. per lang_script) over steps, with CI bands and vertical markers (onset)."""
    todo("M5", "line_chart")


def stacked_bars(counts: Mapping[str, Mapping[str, int]], *, order: Sequence[str]) -> str:
    """Outcome composition per checkpoint (shows abstentions turning into confabulations)."""
    todo("M5", "stacked_bars")


def scatter(
    points: Sequence[tuple[str, float, float, tuple[float, float] | None]], *, x_label: str, y_label: str
) -> str:
    """Pareto view: fluency (eval_loss or chrF++) vs confabulation rate, with CI error bars."""
    todo("M5", "scatter")


def reliability_diagram(bins_by_ckpt: Mapping[str, Sequence[Mapping[str, float]]]) -> str:
    """Calibration diagram (confidence vs accuracy per bin) for base vs final."""
    todo("M5", "reliability_diagram")
