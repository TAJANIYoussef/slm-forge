"""Generation contract (torch-free): requests, records, the Generator protocol, test generators, cache.

Generation and scoring talk only through :class:`GenerationRecord` rows in
``generations.jsonl``. That is what makes eval decoupled from training: any pipeline that
writes ``{"item_id": ..., "response": ...}`` lines can be scored (``slm-forge score``), and
re-scoring never regenerates (``slm-forge rescore``).

Token logprobs come from a **teacher-forced** forward pass over prompt + response (not from
``generate()`` scores, which are post-processed by logits processors and differ between
HF, Unsloth and vLLM). See ``eval.generate_hf.HFGenerator.score_logprobs``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from slmforge._stub import todo
from slmforge.eval.outcomes import FinishReason
from slmforge.types import LangScript, Span

if TYPE_CHECKING:
    from slmforge.eval.prompts import RenderedPrompt


@dataclass(frozen=True)
class DecodingParams:
    """Greedy decoding parameters. Sampling is deliberately not exposed for scored generations."""

    max_new_tokens: Mapping[str, int] = field(default_factory=lambda: {"default": 128})
    stop_strings: tuple[str, ...] = ()
    seed: int = 0
    return_logprobs: bool = True

    def for_bucket(self, bucket: str) -> int:
        """Token budget for a lang_script bucket, falling back to ``default``."""
        return int(self.max_new_tokens.get(bucket, self.max_new_tokens.get("default", 128)))

    def sha256(self) -> str:
        """Hash for the protocol and the cache key."""
        d = {
            "max_new_tokens": dict(sorted(self.max_new_tokens.items())),
            "stop": list(self.stop_strings),
            "seed": self.seed,
            "greedy": True,
            "repetition_penalty": 1.0,
        }
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class GenRequest:
    """One item to generate a response for."""

    item_id: str
    content_hash: str
    lang_script: LangScript
    messages: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class GenerationRecord:
    """One generation: the generation -> scoring contract (one JSON line in ``generations.jsonl``)."""

    item_id: str
    response: str
    finish_reason: FinishReason = "external"
    content_hash: str | None = None
    prompt_sha256: str | None = None
    token_logprobs: tuple[float, ...] | None = None  # teacher-forced, one per response token
    token_char_spans: tuple[Span, ...] | None = None  # response char span of each token
    n_prompt_tokens: int | None = None
    n_new_tokens: int | None = None
    latency_s: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        if self.token_char_spans is not None:
            d["token_char_spans"] = [list(s) for s in self.token_char_spans]
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> GenerationRecord:
        """Deserialize; ``{"item_id", "response"}`` alone is enough (external pipelines)."""
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        if kw.get("token_logprobs") is not None:
            kw["token_logprobs"] = tuple(float(x) for x in kw["token_logprobs"])
        if kw.get("token_char_spans") is not None:
            kw["token_char_spans"] = tuple((int(a), int(b)) for a, b in kw["token_char_spans"])
        return cls(**kw)


@runtime_checkable
class Generator(Protocol):
    """Any backend that turns requests into records (HF, vLLM, replay, fake)."""

    @property
    def fingerprint(self) -> str:
        """Backend + model identity + dtype/quantization/batch/device; part of the cache key."""
        ...

    def generate(self, reqs: Sequence[GenRequest], params: DecodingParams) -> list[GenerationRecord]:
        """Generate one record per request, in input order."""
        ...


class FakeGenerator:
    """Test generator: scripted responses by item_id (or a function), synthetic per-token logprobs.

    Responses are tokenized on whitespace so that confidence code paths get exercised.
    ``fail_after`` raises after that many records, to test resume.
    """

    def __init__(
        self,
        responses: Mapping[str, str] | Callable[[GenRequest], str],
        *,
        logprob: float | None = -0.1,
        fingerprint: str = "fake",
        fail_after: int | None = None,
    ) -> None:
        self._responses = responses
        self._logprob = logprob
        self._fingerprint = fingerprint
        self._fail_after = fail_after
        self._count = 0

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def generate(self, reqs: Sequence[GenRequest], params: DecodingParams) -> list[GenerationRecord]:
        out: list[GenerationRecord] = []
        for req in reqs:
            if self._fail_after is not None and self._count >= self._fail_after:
                raise RuntimeError(f"FakeGenerator: injected failure after {self._count} records")
            text = self._responses(req) if callable(self._responses) else self._responses.get(req.item_id, "")
            spans = _whitespace_spans(text)
            out.append(
                GenerationRecord(
                    item_id=req.item_id,
                    response=text,
                    finish_reason="eos",
                    content_hash=req.content_hash,
                    token_logprobs=None if self._logprob is None else tuple(self._logprob for _ in spans),
                    token_char_spans=tuple(spans),
                    n_new_tokens=len(spans),
                )
            )
            self._count += 1
        return out


class ReplayGenerator:
    """Serves responses from a JSONL file produced by any external pipeline (no logprobs needed)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records = {r.item_id: r for r in read_generations(self.path)}
        self._fp = "replay:" + hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]

    @property
    def fingerprint(self) -> str:
        return self._fp

    def generate(self, reqs: Sequence[GenRequest], params: DecodingParams) -> list[GenerationRecord]:
        missing = [r.item_id for r in reqs if r.item_id not in self._records]
        if missing:
            raise KeyError(f"{self.path}: no response for {len(missing)} item(s), e.g. {missing[:5]}")
        return [self._records[r.item_id] for r in reqs]


def read_generations(path: Path) -> list[GenerationRecord]:
    """Read a ``generations.jsonl`` / external responses file."""
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(GenerationRecord.from_json(json.loads(line)))
    return out


def write_generations(path: Path, records: Sequence[GenerationRecord]) -> None:
    """Write records as JSONL (UTF-8, no ASCII escaping)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def cache_key(
    *, checkpoint_fp: str, generator_fp: str, decoding_sha256: str, prompt_sha256: str, content_hash: str
) -> str:
    """Key for one cached generation. Never a path: ``checkpoint-500`` is reused across runs."""
    raw = "\0".join((checkpoint_fp, generator_fp, decoding_sha256, prompt_sha256, content_hash))
    return hashlib.sha256(raw.encode()).hexdigest()


class GenerationCache:
    """Append-only JSONL cache keyed by :func:`cache_key`, fsync'd per batch; makes runs resumable."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def get(self, key: str) -> GenerationRecord | None:
        """Look up one key."""
        todo("M2", "GenerationCache.get")

    def put_many(self, items: Sequence[tuple[str, GenerationRecord]]) -> None:
        """Append a batch and fsync."""
        todo("M2", "GenerationCache.put_many")

    def keys(self) -> set[str]:
        """All cached keys."""
        todo("M2", "GenerationCache.keys")


def generate_with_cache(
    gen: Generator,
    reqs: Sequence[GenRequest],
    params: DecodingParams,
    cache: GenerationCache,
    *,
    checkpoint_fp: str,
    render: Callable[[GenRequest], RenderedPrompt],
) -> list[GenerationRecord]:
    """Generate only the requests missing from the cache; return records in request order."""
    todo("M2", "generate_with_cache")


def _whitespace_spans(text: str) -> list[Span]:
    spans: list[Span] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans
