"""Wilson, Holm, exact McNemar and design effect against reference values."""

from __future__ import annotations

import math

import pytest

from slmforge.eval.stats import design_effect, holm, holm_adjusted, mcnemar_exact, mcnemar_from_counts, wilson


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [(0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (20, 200, 0.0657, 0.1494)],
)
def test_wilson_reference(k: int, n: int, lo: float, hi: float) -> None:
    got = wilson(k, n)
    assert got is not None
    assert got[0] == pytest.approx(lo, abs=5e-5) and got[1] == pytest.approx(hi, abs=5e-5)


def test_wilson_edge_cases() -> None:
    assert wilson(0, 0) is None
    narrow, wide = wilson(20, 200), wilson(20, 200, deff=2.0)
    assert narrow is not None and wide is not None
    assert wide[1] - wide[0] > narrow[1] - narrow[0]
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_mcnemar_reference() -> None:
    assert mcnemar_from_counts(8, 2).p_value == pytest.approx(0.109375)  # scipy binomtest(2, 10)
    assert mcnemar_from_counts(0, 0).p_value == 1.0
    assert mcnemar_from_counts(5, 5).p_value == 1.0
    r = mcnemar_exact([True, True, False, False], [False, True, True, True])
    assert (r.b01, r.b10) == (2, 1)


def test_holm_hand_example() -> None:
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    # sorted: d .005<=.0125 a .01<=.0167 c .03>.025 -> stop
    assert holm(p) == {"a": True, "b": False, "c": False, "d": True}
    adj = holm_adjusted(p)
    assert adj["d"] == pytest.approx(0.02) and adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert {k for k, v in adj.items() if v <= 0.05} == {k for k, v in holm(p).items() if v}


def test_holm_nan_never_rejected() -> None:
    assert holm({"a": math.nan, "b": 0.001}) == {"a": False, "b": True}


def test_design_effect() -> None:
    assert design_effect([4, 4, 4], 0.3) == pytest.approx(1.9)
    assert design_effect([1, 1], 0.9) == 1.0
