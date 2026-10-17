"""Arabizi (Darija in Latin script): digit roles and the cross-script consonant skeleton (L2).

Arabizi writes some Arabic consonants with digits (3 = ع, 7 = ح, 9 = ق, ...), so "3la"
is a word and "3 ans" is a number. :func:`digit_roles` decides which, and
:func:`mask_arabizi_digits` hides letter-digits from number extraction while keeping
offsets valid. :func:`skeleton` maps Arabic-script and Arabizi Darija to one consonant
skeleton, so a single lexicon regex covers both scripts and ``skeleton_ratio`` can match
"Bank Al-Maghrib" against بنك المغرب.

Skeleton invariants (tested in M4): skel("ma3reftch") == skel("ma3rftch") == "mErftC",
skel("ما عرفتش") == "m ErftC", skel("dyal") == skel("ديال") == "dl",
skel("Bank Al-Maghrib") == skel("بنك المغرب") == "bnk lmGrb".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from slmforge._stub import todo

DigitRole = Literal["letter", "numeral"]

# Digits that may stand for a letter: 2=ء 3=ع 5=خ 7=ح 8=غ 9=ق.
LETTER_DIGITS: Final = frozenset("235789")

# A lowercased token matching this is a numeral even if it contains letters (2eme, 3ans, 70dh, 10k).
NUMERAL_TOKEN_RE: Final = re.compile(
    r"^\d+([.,]\d+)?(e|er|re|eme|ieme|st|nd|rd|th|h|min|k|m|mds?|dh|dhs|mad|ans?|mois|jours?|%|€|\$)$"
)

# Maximal Arabizi token: ASCII letters, digits and the apostrophe of 3' / 7'.
ARABIZI_TOKEN_RE: Final = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class Skeleton:
    """Consonant skeleton plus offsets back to the source text."""

    text: str
    offsets: tuple[int, ...]  # offsets[i]: source offset of skeleton char i


def digit_roles(token: str) -> list[DigitRole | None]:
    """Per-char role for digits in an Arabizi token.

    A digit is a *letter* iff it is in :data:`LETTER_DIGITS`, the token has at least one
    ASCII letter, a neighbour is a letter (or the apostrophe of 3'/7'), and the lowercased
    token does not match :data:`NUMERAL_TOKEN_RE`. Non-digits get None.
    """
    todo("M4", "digit_roles")


def mask_arabizi_digits(s: str, mask_char: str = " ") -> str:
    """Length-preserving mask of letter-digits."""
    todo("M4", "mask_arabizi_digits")


def skeleton(s: str, *, keep_spaces: bool = True) -> Skeleton:
    """L2 skeleton shared by Arabic-script and Arabizi Darija.

    Computed from L0 text. Arabizi: lowercase, strip accents; digraphs longest first
    (3'→G, 7'→x, kh→x, gh→G, ch/sh→C, dj→j, th→t, ou→u); letter-digits 2→"" 3→E 5→x
    7→H 8→G 9→q; drop aeiou; drop w/y unless token-initial; collapse runs. Arabic:
    ا→"" ب b, ت/ث/ط t, ج j, ح H, خ x, د/ذ/ض/ظ d, ر r, ز z, س/ص s, ش C, ع E, غ G, ف f,
    ق q, ك k, ل l, م m, ن n, ه h, و w, ي y; ة→""; word-final ه→"" once the token has >= 2
    consonants; drop و/ي unless token-initial; collapse runs.
    """
    todo("M4", "skeleton")


def skeleton_ratio(a: str, b: str) -> float:
    """1 - normalized Levenshtein distance of two skeletons."""
    todo("M4", "skeleton_ratio")
