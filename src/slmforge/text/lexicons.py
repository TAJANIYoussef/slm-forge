"""Versioned cue lexicons: abstention, premise rejection, hedges, filler, contrast, stopwords, yes/no.

Files live at ``resources/lexicons/<version>/<kind>/<bucket>.yaml`` (format:
``resources/lexicons/v1/README.md``). **Every bucket is applied to every response**,
whatever the item's language, so code-switched answers ("désolé, ma3reftch") and MSA
replies to Darija prompts are covered.

Regex entries carry a ``pattern`` (on L1 text) and/or a ``skeleton`` (on the L2 consonant
skeleton, so one regex covers both Darija scripts). ``unless_followed_by`` suppresses a
hit, e.g. "I don't know *why* ..." is not an abstention. Strong vs weak cues: weak cues
(courtesy formulas such as "consult a lawyer", "الله أعلم") count as abstention only when
no answer is given and never count as a hedge.

The set is hashed (``LexiconSet.sha256``) into the evaluation protocol, so editing a
lexicon changes the protocol hash and old and new results are never mixed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Final, Literal, cast, get_args

import yaml

from slmforge._stub import todo
from slmforge.text.normalize import l1_char_problem
from slmforge.types import LEX_BUCKETS, LexBucket, Span

LexKind = Literal["abstain", "premise", "hedge", "filler", "contrast", "stopwords", "yes_no"]
REGEX_KINDS: Final[frozenset[str]] = frozenset({"abstain", "premise", "hedge", "filler", "contrast"})
WORD_KINDS: Final[frozenset[str]] = frozenset({"stopwords", "yes_no"})
MIN_SKELETON_CHARS: Final = 4  # shorter skeletons collide (momkin/makan -> mkn)

_ENTRY_KEYS: Final = frozenset(
    {"id", "category", "strength", "hedge_kind", "pattern", "skeleton", "unless_followed_by", "note"}
)
# Regex syntax stripped before counting a skeleton pattern's literal characters.
_REGEX_SYNTAX_RE: Final = re.compile(r"\\[bBdDsSwWAZ]|\(\?[:!=<]+|[\\^$.|?*+()\[\]{}\s]|\d")


class LexiconError(ValueError):
    """A lexicon file is malformed; the message names the file and entry."""


@dataclass(frozen=True)
class LexiconEntry:
    """One cue: surface regex on L1 and/or skeleton regex on L2."""

    id: str
    kind: LexKind
    bucket: LexBucket
    category: str
    strength: Literal["strong", "weak"]
    hedge_kind: Literal["epistemic", "approximation"] | None
    pattern: re.Pattern[str] | None
    skeleton: re.Pattern[str] | None
    unless_followed_by: re.Pattern[str] | None


@dataclass(frozen=True)
class LexiconMatch:
    """A cue hit in raw-response coordinates."""

    entry: LexiconEntry
    span: Span
    via: Literal["surface", "skeleton"]


@dataclass(frozen=True, eq=False)
class LexiconSet:
    """All packaged plus user lexicons; hashed into the protocol."""

    version: str
    sha256: str
    entries: tuple[LexiconEntry, ...]
    stopwords_by_bucket: Mapping[str, frozenset[str]] = field(default_factory=dict)
    yes_words: frozenset[str] = frozenset()
    no_words: frozenset[str] = frozenset()
    first_token_only: frozenset[str] = frozenset()

    @classmethod
    def load(cls, version: str = "v1", *, root: Path | None = None, extra: Sequence[Path] = ()) -> LexiconSet:
        """Load and validate a lexicon version (packaged unless ``root`` is given) plus extra files.

        Validation: known keys only, unique ids, regexes compile, surface patterns are in L1
        form, skeleton patterns have >= 4 literal characters, hedge entries declare a
        hedge_kind. The hash covers every file's relative path and bytes, in sorted order.
        """
        base: Traversable = (
            files("slmforge").joinpath("resources", "lexicons", version) if root is None else root
        )
        sources: list[tuple[str, bytes]] = []
        for kind_dir in sorted(_iterdir(base), key=lambda t: t.name):
            if not kind_dir.is_dir():
                continue
            for f in sorted(_iterdir(kind_dir), key=lambda t: t.name):
                if f.name.endswith(".yaml"):
                    sources.append((f"{kind_dir.name}/{f.name}", f.read_bytes()))
        if not sources:
            raise LexiconError(f"no lexicon files found for version {version!r} under {base}")
        sources += [(f"extra/{Path(p).name}", Path(p).read_bytes()) for p in extra]
        return _build(version, sources)

    def find(
        self, text: str, kinds: Collection[LexKind], *, buckets: Collection[LexBucket] | None = None
    ) -> list[LexiconMatch]:
        """Find cues on the L1 and skeleton views; union of all buckets by default (code-switching)."""
        todo("M1", "LexiconSet.find")

    def by_kind(self, kind: LexKind) -> list[LexiconEntry]:
        """All regex entries of one kind."""
        return [e for e in self.entries if e.kind == kind]

    def stopwords(self) -> frozenset[str]:
        """Union of stopwords across buckets (L1 form)."""
        return frozenset().union(*self.stopwords_by_bucket.values())

    def polarity(self, token: str, *, first_token: bool = True) -> Literal["yes", "no"] | None:
        """Yes/no polarity of an L1 token; ambiguous words (Latin 'la') only count as the first token."""
        if token in self.first_token_only and not first_token:
            return None
        if token in self.yes_words:
            return "yes"
        if token in self.no_words:
            return "no"
        return None


def _iterdir(t: Traversable | Path) -> Iterable[Traversable]:
    return t.iterdir() if t.is_dir() else ()


def _fail(label: str, msg: str) -> LexiconError:
    return LexiconError(f"{label}: {msg}")


def _compile(label: str, where: str, src: Any) -> re.Pattern[str]:
    if not isinstance(src, str) or not src:
        raise _fail(label, f"{where} must be a non-empty regex string")
    try:
        return re.compile(src)
    except re.error as e:
        raise _fail(label, f"{where}: invalid regex {src!r}: {e}") from None


def _check_l1_regex(label: str, where: str, src: str) -> None:
    """Surface patterns run on L1 text, so literals must already be in L1 form."""
    escaped = False
    for ch in src:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch.isascii() and not ch.isupper():
            continue
        if problem := l1_char_problem(ch):
            raise _fail(label, f"{where}: {ch!r} can never appear in L1 text ({problem})")


def _check_l1_word(label: str, where: str, word: Any) -> str:
    if not isinstance(word, str) or not word.strip() or word != word.strip():
        raise _fail(label, f"{where}: {word!r} must be a non-empty string without outer spaces")
    for ch in word:
        if problem := l1_char_problem(ch):
            raise _fail(label, f"{where}: {word!r} is not in L1 form ({ch!r}: {problem})")
    return word


def skeleton_literal_len(src: str) -> int:
    """Literal characters in a skeleton regex (syntax and digits removed); a collision heuristic."""
    return len(_REGEX_SYNTAX_RE.sub("", src))


def _build(version: str, sources: Sequence[tuple[str, bytes]]) -> LexiconSet:
    entries: list[LexiconEntry] = []
    seen_ids: set[str] = set()
    stop: dict[str, frozenset[str]] = {}
    yes: set[str] = set()
    no: set[str] = set()
    first_only: set[str] = set()
    digest = hashlib.sha256()
    for label, data in sources:
        digest.update(label.encode() + b"\0" + data + b"\0")
        try:
            doc = yaml.safe_load(data.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as e:
            raise _fail(label, f"cannot parse YAML: {e}") from None
        if not isinstance(doc, dict):
            raise _fail(label, "top level must be a mapping")
        kind, bucket = doc.get("kind"), doc.get("bucket")
        if kind not in get_args(LexKind):
            raise _fail(label, f"kind must be one of {list(get_args(LexKind))}, got {kind!r}")
        if bucket not in LEX_BUCKETS:
            raise _fail(label, f"bucket must be one of {list(LEX_BUCKETS)}, got {bucket!r}")
        if kind in WORD_KINDS:
            if kind == "stopwords":
                words = doc.get("words") or []
                stop[bucket] = stop.get(bucket, frozenset()) | {
                    _check_l1_word(label, "words", w) for w in words
                }
            else:
                yes |= {_check_l1_word(label, "yes", w) for w in doc.get("yes") or []}
                no |= {_check_l1_word(label, "no", w) for w in doc.get("no") or []}
                first_only |= {
                    _check_l1_word(label, "first_token_only", w) for w in doc.get("first_token_only") or []
                }
            continue
        raw_entries = doc.get("entries")
        if not isinstance(raw_entries, list):
            raise _fail(label, "entries must be a list")
        for i, raw in enumerate(raw_entries):
            where = f"entries[{i}]"
            if not isinstance(raw, dict):
                raise _fail(label, f"{where} must be a mapping")
            if unknown := set(raw) - _ENTRY_KEYS:
                raise _fail(label, f"{where}: unknown key(s) {sorted(unknown)}")
            eid = raw.get("id")
            if not isinstance(eid, str) or not eid:
                raise _fail(label, f"{where}: missing id")
            if eid in seen_ids:
                raise _fail(label, f"{where}: duplicate id {eid!r}")
            seen_ids.add(eid)
            if not raw.get("pattern") and not raw.get("skeleton"):
                raise _fail(label, f"{eid}: needs a pattern or a skeleton")
            strength = raw.get("strength", "strong")
            if strength not in ("strong", "weak"):
                raise _fail(label, f"{eid}: strength must be strong or weak")
            hedge_kind = raw.get("hedge_kind")
            if kind == "hedge" and hedge_kind not in ("epistemic", "approximation"):
                raise _fail(label, f"{eid}: hedge entries need hedge_kind epistemic or approximation")
            if kind != "hedge" and hedge_kind is not None:
                raise _fail(label, f"{eid}: hedge_kind is only valid for hedge entries")
            pattern = skeleton = None
            if raw.get("pattern"):
                _check_l1_regex(label, f"{eid}.pattern", raw["pattern"])
                pattern = _compile(label, f"{eid}.pattern", raw["pattern"])
            if raw.get("skeleton"):
                if skeleton_literal_len(raw["skeleton"]) < MIN_SKELETON_CHARS:
                    raise _fail(
                        label,
                        f"{eid}: skeleton patterns need >= {MIN_SKELETON_CHARS} literal characters; "
                        "use a surface pattern instead",
                    )
                skeleton = _compile(label, f"{eid}.skeleton", raw["skeleton"])
            unless = raw.get("unless_followed_by")
            entries.append(
                LexiconEntry(
                    id=eid,
                    kind=cast(LexKind, kind),
                    bucket=cast(LexBucket, bucket),
                    category=str(raw.get("category", kind)),
                    strength=strength,
                    hedge_kind=hedge_kind,
                    pattern=pattern,
                    skeleton=skeleton,
                    unless_followed_by=None
                    if unless is None
                    else _compile(label, f"{eid}.unless_followed_by", unless),
                )
            )
    return LexiconSet(
        version=version,
        sha256=digest.hexdigest(),
        entries=tuple(entries),
        stopwords_by_bucket=stop,
        yes_words=frozenset(yes),
        no_words=frozenset(no),
        first_token_only=frozenset(first_only),
    )
