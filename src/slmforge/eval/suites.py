"""Registry of evaluation suites, resolved lazily so naming a suite never imports more than it needs."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, Final, Protocol


class Suite(Protocol):
    """A named evaluation suite."""

    name: ClassVar[str]


SUITES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "factual-reliability": "slmforge.eval.factual_reliability:FactualReliabilitySuite",
        "task": "slmforge.eval.task_metrics:TaskMetricsSuite",
    }
)
"""Suite name -> ``module:attribute`` import path."""


def get_suite(name: str) -> type:
    """Import and return the suite class registered under ``name``."""
    try:
        target = SUITES[name]
    except KeyError:
        raise ValueError(f"unknown suite {name!r}; known suites: {', '.join(sorted(SUITES))}") from None
    module_name, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module_name), attr)
    if not isinstance(cls, type) or getattr(cls, "name", None) != name:
        raise TypeError(f"{target} is not a suite class named {name!r}")
    return cls
