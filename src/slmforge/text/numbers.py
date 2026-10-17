"""Locale-aware quantities: numbers, percents, money and durations, plus tolerance checks.

Quantities are parsed from L0 text (so NBSP/NNBSP group separators survive) with
citation, date and Arabizi letter-digit spans masked first ("loi 31.08" is never the
number 31.08, "3la" is never 3). Every plausible reading goes into
``Quantity.interpretations`` with a flag, and the matcher accepts a candidate if *any*
reading is within the gold tolerance:

* ``locale_ambiguous`` -- a lone ``1,500`` / ``1.500`` (fr/darija: 1500 vs 1.5; en: reverse)
* ``fr_billion`` -- French "billion" = 1e12
* ``colloquial_unit_conversion`` -- ryal = 0.05 MAD, centime/frank = 0.01 MAD
* ``darija_money_scale`` -- "melyoun" with no currency (or in centimes/ryal) also reads as
  1e4 MAD, because 1 melyoun (centimes) = 10,000 DH colloquially

Units, multipliers and conversions are data (``resources/units.yaml``, loaded by
:class:`UnitTable`); number words are in ``resources/numbers/<bucket>.yaml``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import MAX_EMAX, MAX_PREC, MIN_EMIN, Context, Decimal, InvalidOperation
from importlib.resources import files
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from slmforge._stub import todo
from slmforge.text.normalize import l1_char_problem
from slmforge.types import Language, LexBucket, Span

QuantityKind = Literal["number", "percent", "money", "duration"]
UNIT_KINDS: Final[tuple[QuantityKind, ...]] = ("percent", "money", "duration")
LANG_HINTS: Final = frozenset({"darija", "fr", "en"})

# Grouped-number patterns (applied to L0 text with masks), most specific first.
GROUPED_NUMBER_PATTERNS: Final[Mapping[str, str]] = {
    "fr": r"\d{1,3}(?:[    .]\d{3})+(?:,\d+)?",
    "en": r"\d{1,3}(?:,\d{3})+(?:\.\d+)?",
    "ar": r"\d{1,3}(?:٬\d{3})+(?:٫\d+)?",
    "plain": r"[-+−]?\d+(?:[.,٫]\d+)?",
}

# Unrounded arithmetic for tolerance checks: add/subtract/multiply are exact under it.
_EXACT: Final = Context(prec=MAX_PREC, Emax=MAX_EMAX, Emin=MIN_EMIN)


class UnitTableError(ValueError):
    """A units file is malformed; the message names the file and the offending key."""


@dataclass(frozen=True)
class Quantity:
    """A parsed quantity with every plausible interpretation and flags."""

    value: Decimal  # the reading preferred for the lang hint
    kind: QuantityKind
    unit: str | None  # canonical unit code ("%", "MAD", "EUR", "USD", "days", "months", "years")
    span: Span
    raw: str
    interpretations: tuple[Decimal, ...]  # includes value
    flags: frozenset[str]


@dataclass(frozen=True)
class UnitDef:
    """A canonical unit and the surface forms (L1) that denote it."""

    code: str
    kind: QuantityKind
    surfaces: tuple[str, ...]


@dataclass(frozen=True)
class Multiplier:
    """Words that scale the preceding number, optionally implying a unit (``mdh`` = x1e6 MAD)."""

    surfaces: tuple[str, ...]
    factor: Decimal
    unit: str | None = None
    langs: frozenset[str] | None = None  # lang hints this reading applies to; None = every hint
    flag: str | None = None


@dataclass(frozen=True, eq=False)
class UnitTable:
    """Units, multipliers and colloquial conversions from resources/units.yaml."""

    version: int
    units: Mapping[str, UnitDef]
    scales: tuple[Multiplier, ...]  # k, mille, million, milliard, billion (per lang) ...
    compounds: tuple[Multiplier, ...]  # unit-bearing scales: MDH = 1e6 MAD, MMDH = 1e9 MAD
    colloquial: tuple[Multiplier, ...]  # ryal = 0.05 MAD, centime = 0.01 MAD
    darija_money_scale: Multiplier | None  # melyoun -> extra x1e4 MAD reading
    conversions: Mapping[tuple[str, str], Decimal]  # (from, to) -> factor, e.g. (years, months) -> 12
    by_surface: Mapping[str, str]  # unit surface -> unit code
    sha256: str  # of the file bytes

    @classmethod
    def load(cls, path: Path | None = None) -> UnitTable:
        """Load the packaged or a custom table."""
        if path is None:
            data = files("slmforge").joinpath("resources", "units.yaml").read_bytes()
            label = "resources/units.yaml"
        else:
            data = Path(path).read_bytes()
            label = str(path)
        return _parse_units(data, label)


def _fail(label: str, msg: str) -> UnitTableError:
    return UnitTableError(f"{label}: {msg}")


def _mapping(obj: Any, label: str, where: str, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise _fail(label, f"{where} must be a mapping, got {type(obj).__name__}")
    if unknown := set(obj) - allowed:
        raise _fail(label, f"{where}: unknown key(s) {sorted(map(str, unknown))}")
    if missing := required - set(obj):
        raise _fail(label, f"{where}: missing key(s) {sorted(missing)}")
    return obj


def _surfaces(obj: Any, label: str, where: str) -> tuple[str, ...]:
    if not isinstance(obj, list) or not obj:
        raise _fail(label, f"{where}.surfaces must be a non-empty list")
    out: list[str] = []
    for s in obj:
        if not isinstance(s, str) or not s.strip() or s != s.strip():
            raise _fail(label, f"{where}.surfaces: {s!r} must be a non-empty string without outer spaces")
        for ch in s:
            if problem := l1_char_problem(ch):
                raise _fail(label, f"{where}.surfaces: {s!r} is not in L1 form ({ch!r}: {problem})")
        out.append(s)
    return tuple(out)


def _decimal(obj: Any, label: str, where: str) -> Decimal:
    if isinstance(obj, bool) or not isinstance(obj, int | str):
        raise _fail(label, f"{where} must be an integer or a quoted decimal string, got {obj!r}")
    try:
        d = Decimal(obj)
    except InvalidOperation:
        raise _fail(label, f"{where}: {obj!r} is not a decimal number") from None
    if not d.is_finite() or d <= 0:
        raise _fail(label, f"{where} must be a finite positive number, got {obj!r}")
    return d


def _multiplier(obj: Any, label: str, where: str, units: Mapping[str, UnitDef]) -> Multiplier:
    m = _mapping(obj, label, where, {"surfaces", "factor", "unit", "langs", "flag"}, {"surfaces", "factor"})
    unit = m.get("unit")
    if unit is not None and unit not in units:
        raise _fail(label, f"{where}.unit {unit!r} is not defined under units")
    langs = m.get("langs")
    if langs is not None:
        if not isinstance(langs, list) or not langs or not set(langs) <= LANG_HINTS:
            raise _fail(label, f"{where}.langs must be a non-empty subset of {sorted(LANG_HINTS)}")
        langs = frozenset(langs)
    flag = m.get("flag")
    if flag is not None and not (isinstance(flag, str) and flag):
        raise _fail(label, f"{where}.flag must be a non-empty string")
    return Multiplier(
        _surfaces(m["surfaces"], label, where),
        _decimal(m["factor"], label, f"{where}.factor"),
        unit,
        langs,
        flag,
    )


def _parse_units(data: bytes, label: str) -> UnitTable:
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise _fail(label, f"cannot parse YAML: {e}") from None
    top = _mapping(
        raw,
        label,
        "top level",
        {"version", "units", "scales", "compounds", "colloquial", "darija_money_scale", "conversions"},
        {"version", "units"},
    )
    version = top["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise _fail(label, f"version must be an integer, got {version!r}")

    if not isinstance(top["units"], dict) or not top["units"]:
        raise _fail(label, "units must be a non-empty mapping")
    units: dict[str, UnitDef] = {}
    by_surface: dict[str, str] = {}
    for code, spec in top["units"].items():
        where = f"units[{code!r}]"
        u = _mapping(spec, label, where, {"kind", "surfaces"}, {"kind", "surfaces"})
        if u["kind"] not in UNIT_KINDS:
            raise _fail(label, f"{where}.kind must be one of {list(UNIT_KINDS)}, got {u['kind']!r}")
        units[str(code)] = UnitDef(str(code), u["kind"], _surfaces(u["surfaces"], label, where))
        for s in units[str(code)].surfaces:
            if s in by_surface:
                raise _fail(label, f"surface {s!r} maps to both {by_surface[s]!r} and {code!r}")
            by_surface[s] = str(code)

    def mults(key: str) -> tuple[Multiplier, ...]:
        seq = top.get(key, [])
        if not isinstance(seq, list):
            raise _fail(label, f"{key} must be a list")
        return tuple(_multiplier(m, label, f"{key}[{i}]", units) for i, m in enumerate(seq))

    scales, compounds, colloquial = mults("scales"), mults("compounds"), mults("colloquial")
    for group, items in (("compounds", compounds), ("colloquial", colloquial)):
        for m in items:
            if m.unit is None:
                raise _fail(label, f"{group} entries need a unit ({m.surfaces[0]!r})")
            if clash := set(m.surfaces) & by_surface.keys():
                raise _fail(label, f"{group} surface(s) {sorted(clash)} are also unit surfaces")
    # A scale surface may carry different factors per lang hint, but the lang sets must not overlap.
    seen: dict[str, list[frozenset[str] | None]] = {}
    for m in scales:
        for s in m.surfaces:
            for other in seen.get(s, []):
                if other is None or m.langs is None or other & m.langs:
                    raise _fail(label, f"scale surface {s!r} is defined twice for the same lang hint")
            seen.setdefault(s, []).append(m.langs)

    dms = top.get("darija_money_scale")
    darija_money_scale = None if dms is None else _multiplier(dms, label, "darija_money_scale", units)

    conversions: dict[tuple[str, str], Decimal] = {}
    conv_seq = top.get("conversions", [])
    if not isinstance(conv_seq, list):
        raise _fail(label, "conversions must be a list")
    for i, c in enumerate(conv_seq):
        where = f"conversions[{i}]"
        c = _mapping(c, label, where, {"from", "to", "factor"}, {"from", "to", "factor"})
        src, dst = c["from"], c["to"]
        if src not in units or dst not in units:
            raise _fail(label, f"{where}: units {src!r} and {dst!r} must both be defined")
        if units[src].kind != units[dst].kind:
            raise _fail(label, f"{where}: cannot convert {units[src].kind} to {units[dst].kind}")
        conversions[(src, dst)] = _decimal(c["factor"], label, f"{where}.factor")

    return UnitTable(
        version=version,
        units=units,
        scales=scales,
        compounds=compounds,
        colloquial=colloquial,
        darija_money_scale=darija_money_scale,
        conversions=conversions,
        by_surface=by_surface,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def parse_number_words(tokens: Sequence[str], bucket: LexBucket) -> tuple[Decimal, int] | None:
    """Compositional number-word parse; returns (value, tokens consumed).

    Tokens are L1. Covers fr soixante-dix / quatre-vingt(s) / quatre-vingt-dix / septante /
    nonante, Darija جوج/زوج ... سبعين, مية/ميتين, wa7ed ... sb3in, mya/mitin, the
    conjunctions و / w / o, and fractions (نص/nss 0.5, ربع/rbe3 0.25, تلت/tolt 1/3, the
    latter with an automatic rel_tol of 1e-4). Word lists: ``resources/numbers/<bucket>.yaml``.
    M1 covers the en/fr buckets; M4 adds darija_arab/darija_latn.
    """
    todo("M1", "parse_number_words")


def extract_quantities(
    text: str,
    *,
    lang_hint: Language | None,
    units: UnitTable | None = None,
    mask: Sequence[Span] = (),
    darija_money_scale: bool = True,
    colloquial_units: bool = True,
) -> list[Quantity]:
    """Locale-aware numbers, percents, money and durations from L0 text.

    When both ``.`` and ``,`` appear, the last one is the decimal separator. Scale words
    and units are read from the next token(s). M1 covers en/fr; Darija/MSA number words
    and Moroccan colloquial quantities arrive in M4.
    """
    todo("M1", "extract_quantities")


def _as_decimal(x: Decimal | int | str, name: str) -> Decimal:
    if isinstance(x, float):
        raise TypeError(f"{name} must be a Decimal (got float {x!r}; use Decimal(str(x)))")
    return x if isinstance(x, Decimal) else Decimal(x)


def within(a: Decimal, gold: Decimal, abs_tol: Decimal = Decimal(0), rel_tol: Decimal = Decimal(0)) -> bool:
    """|a-g| <= max(abs_tol, rel_tol*|g|).

    Exact (unrounded) Decimal arithmetic, boundary inclusive. A non-finite value or gold is
    never within; a negative or non-finite tolerance is a ValueError.
    """
    a, gold = _as_decimal(a, "a"), _as_decimal(gold, "gold")
    abs_tol, rel_tol = _as_decimal(abs_tol, "abs_tol"), _as_decimal(rel_tol, "rel_tol")
    for name, tol in (("abs_tol", abs_tol), ("rel_tol", rel_tol)):
        if not tol.is_finite() or tol < 0:
            raise ValueError(f"{name} must be a finite non-negative Decimal, got {tol}")
    if not (a.is_finite() and gold.is_finite()):
        return False
    diff = _EXACT.abs(_EXACT.subtract(a, gold))
    bound = max(abs_tol, _EXACT.multiply(rel_tol, _EXACT.abs(gold)))
    return diff <= bound


def format_number(v: Decimal, locale: Literal["fr", "en", "ar"]) -> str:
    """Locale formatting (tests, report)."""
    todo("M4", "format_number")
