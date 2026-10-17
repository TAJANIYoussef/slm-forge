"""Markdown and HTML reports for one checkpoint or a whole run (plan §6.4).

The HTML is self-contained: no JavaScript, no CDN, inline SVG charts (``report.svg``) and
every model response or Arabic string wrapped in ``<bdi dir="auto">`` so mixed-direction
text renders correctly. Reports refuse to compare confabulation rates across checkpoints
whose ``confidence_source`` differs and say so in the table instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final

from slmforge._stub import todo

if TYPE_CHECKING:
    from slmforge.eval.runner import CheckpointMetrics, RunMetrics

REPORT_FORMATS: Final[tuple[str, ...]] = ("md", "html")


def load_run_metrics(
    run_dir: Path, *, suite: str = "factual-reliability", protocol: str | None = None
) -> RunMetrics:
    """Load ``eval/<suite>/<protocol8>/run_metrics.json`` (newest protocol unless ``protocol`` is given)."""
    todo("M3", "load_run_metrics")


def render_markdown(metrics: RunMetrics | CheckpointMetrics) -> str:
    """Markdown report: one checkpoint's tables (M1) or a run's headline, flags and trend tables."""
    todo("M1", "render_markdown")


def render_html(run: RunMetrics) -> str:
    """Self-contained HTML: no JS/CDN, inline SVG, Arabic in ``<bdi dir='auto'>``."""
    todo("M5", "render_html")


def write_trend_csv(run: RunMetrics, path: Path) -> Path:
    """Step x metric x bucket CSV with values, CIs and bounds."""
    todo("M5", "write_trend_csv")


def render_run_report(
    run_dir: Path, *, suite: str = "factual-reliability", formats: Sequence[str] = REPORT_FORMATS
) -> list[Path]:
    """Write ``report.md`` / ``report.html`` next to ``run_metrics.json``; returns the written paths."""
    todo("M3", "render_run_report")
