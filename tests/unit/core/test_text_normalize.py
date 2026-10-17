"""L0 normalization, digit folding, script profiles and the L1 character oracle."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from slmforge.text.normalize import fold_digits, l1_char_problem, normalize_light, script_profile


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("مـــدونة", "مدونة"),  # tatweel removed, letters untouched
        ("‏ما عرفتش‎", "ما عرفتش"),  # RLM/LRM stripped
        ("Prêt  logement", "Prêt logement"),  # NBSP run collapsed, accent kept (L0)
        ("1 000,50 DH", "1 000,50 DH"),  # NNBSP group separator becomes a space
        ("  a \t b \n c  ", "a b c"),
        ("﻿bonjour", "bonjour"),
        ("أَلْفُ", "أَلْفُ"),  # L0 never strips harakat
    ],
)
def test_normalize_light(raw: str, expected: str) -> None:
    assert normalize_light(raw) == expected


def test_normalize_light_keep_newlines() -> None:
    assert normalize_light("Explication.\n\n  Réponse finale : 70 %", keep_newlines=True) == (
        "Explication.\nRéponse finale : 70 %"
    )


@given(st.text(alphabet=st.characters(codec="utf-8"), max_size=50))
def test_normalize_light_idempotent(s: str) -> None:
    once = normalize_light(s)
    assert normalize_light(once) == once


def test_fold_digits() -> None:
    assert fold_digits("٧٠٪ و ۱۲") == "70٪ و 12"


@pytest.mark.parametrize(
    ("text", "dominant"),
    [
        ("شحال هو أقصى LTV ديال كريدي السكن", "arab"),  # LTV acronym ignored
        ("ch7al howa l'LTV l9ousswa dyal credit", "latn"),
        ("Quel est le ratio LTV maximal ?", "latn"),
        ("واش credit sakan ديال BAM", "mixed"),
        ("70 % — 2024", None),  # no letters
        ("LTV BAM", "latn"),  # all-caps only: acronyms count rather than leaving nothing
    ],
)
def test_script_profile(text: str, dominant: str | None) -> None:
    assert script_profile(text).dominant == dominant


@pytest.mark.parametrize(
    ("ch", "ok"),
    [
        ("a", True),
        ("ه", True),
        ("ي", True),
        ("%", True),
        ("é", False),
        ("ة", False),
        ("ى", False),
        ("أ", False),
        ("٧", False),
        ("A", False),
        (" ", False),
        ("ـ", False),
        ("ﻻ", False),
    ],
)
def test_l1_char_problem(ch: str, ok: bool) -> None:
    assert (l1_char_problem(ch) is None) is ok
