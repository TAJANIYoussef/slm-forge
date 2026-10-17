"""The instrument pattern covers the forms used in the sample data and docs (L1 text)."""

from __future__ import annotations

import pytest

from slmforge.text.citations import INSTRUMENT_PATTERN


@pytest.mark.parametrize(
    ("text", "groups"),
    [
        ("law no. 77-19", ("77", "19")),
        ("loi n° 31.08", ("31", "08")),
        ("القانون رقم 77-19", ("77", "19")),
        ("l9anoun 77-19", ("77", "19")),
        ("dahir 1-58", ("1", "58")),
    ],
)
def test_instrument_pattern(text: str, groups: tuple[str, str]) -> None:
    m = INSTRUMENT_PATTERN.search(text)
    assert m is not None and m.groups() == groups
