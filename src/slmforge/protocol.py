"""The evaluation protocol: everything that changes a score, its hash, the lock and drift detection.

Two results are comparable only if their protocols hash the same. The protocol covers the
scorer/normalizer/matcher/decision-table versions, the lexicon and prompt-template hashes,
decoding, tau, the scoring config, statistics settings, the dataset, split and subsample,
and the judge/validation report. Results are stored under ``eval/<suite>/<protocol8>/``,
so different protocols are never mixed in one trend or report.

Pre-registration: ``slm-forge protocol lock`` writes the protocol to a lock file; an eval
configured with ``protocol_lock`` then fails with :class:`ProtocolDrift` (listing the
differing fields) if anything in the live protocol changed.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from slmforge._stub import todo
from slmforge.repro import canonical_json, sha256_text

if TYPE_CHECKING:
    from slmforge.config.schema import EvalConfig
    from slmforge.eval.dataset import GroundTruthSet
    from slmforge.eval.prompts import PromptTemplateSet
    from slmforge.text.lexicons import LexiconSet

LOCK_SCHEMA_VERSION: Final = 1


class Protocol(BaseModel):
    """Everything that changes a score; differing hashes mean results are not comparable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = "default"
    suite: str
    # code versions
    scorer_version: str
    normalizer_version: str
    matcher_version: str
    outcome_table_version: str  # the decision table is versioned data (outcomes.OUTCOME_TABLE_VERSION)
    # resources
    lexicon_version: str
    lexicon_sha256: str
    prompt_template_id: str
    prompt_template_sha256: str
    prompt_mode: Literal["abstain_ok", "neutral"]
    # generation
    generation_backend: Literal["hf", "vllm", "replay", "fake", "external"]  # external: `slm-forge score`
    decoding_sha256: str
    # scoring
    scoring_sha256: str  # the whole resolved ScoringConfig (matching switches, thresholds, weights)
    tau: float
    tau_source: Literal["fixed", "base_dev"] = "fixed"
    tau_sweep: tuple[float, ...]
    unanswerable_no_cue_policy: Literal["residual", "unresolved"]
    utility_weights_sha256: str
    # statistics
    primary_metrics: tuple[str, ...]
    headline: Literal["macro", "micro"] = "macro"
    alpha: float
    bootstrap_resamples: int
    seed: int
    min_bucket_n: int = 30  # decides which buckets enter the macro headline
    cluster_strata: tuple[str, ...] = ("answerable", "domain")
    reference: str = "base"
    # data
    dataset_sha256: str
    split: str
    require_reviewed: bool = False
    subsample_ids_sha256: str | None = None
    # judge and detector validation
    judge_id: str | None = None
    validation_report_sha256: str | None = None

    def sha256(self) -> str:
        """Canonical-JSON hash of all fields."""
        return sha256_text(canonical_json(self.model_dump(mode="json")))

    def short(self) -> str:
        """First 8 hex chars of :meth:`sha256`; names the results directory."""
        return self.sha256()[:8]

    def diff(self, other: Protocol) -> dict[str, tuple[Any, Any]]:
        """Fields whose values differ, as ``{field: (self_value, other_value)}`` in JSON form."""
        a, b = self.model_dump(mode="json"), other.model_dump(mode="json")
        return {k: (a.get(k), b.get(k)) for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}


class ProtocolDrift(Exception):
    """Raised when a live protocol differs from a lock; the message lists the differing fields."""

    def __init__(self, fields: Mapping[str, tuple[Any, Any]], *, lock_path: Path | None = None) -> None:
        self.fields = dict(fields)
        self.lock_path = lock_path
        where = f" ({lock_path})" if lock_path is not None else ""
        lines = [f"  {k}: locked={v[0]!r} live={v[1]!r}" for k, v in self.fields.items()]
        super().__init__(
            f"protocol drift against the lock{where}: {len(self.fields)} field(s) differ\n" + "\n".join(lines)
        )


def build_protocol(
    cfg: EvalConfig,
    gt: GroundTruthSet,
    *,
    lexicons: LexiconSet | None = None,
    templates: PromptTemplateSet | None = None,
    subsample_ids: Collection[str] | None = None,
    name: str = "default",
) -> Protocol:
    """Assemble the protocol from config, data and resource hashes (lexicons/templates default to cfg's)."""
    todo("M1", "build_protocol")


def _lock_document(protocol: Protocol) -> dict[str, Any]:
    return {
        "lock_schema_version": LOCK_SCHEMA_VERSION,
        "sha256": protocol.sha256(),
        "short": protocol.short(),
        "locked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": protocol.model_dump(mode="json"),
    }


def _read_lock(path: Path) -> Protocol:
    """Parse a lock file and check that its recorded hash matches its content."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "protocol" not in doc:
        raise ValueError(f"{path}: not a protocol lock (missing 'protocol')")
    locked = Protocol.model_validate(doc["protocol"])
    recorded = doc.get("sha256")
    if recorded is not None and recorded != locked.sha256():
        raise ValueError(
            f"{path}: lock is inconsistent (recorded sha256 {recorded[:12]}..., content hashes to "
            f"{locked.sha256()[:12]}...); it was edited by hand or written by another scorer version"
        )
    return locked


def lock_protocol(protocol: Protocol, path: Path, *, overwrite: bool = False) -> None:
    """Write the lock file (pre-registration).

    Re-locking the same protocol is a no-op; replacing a lock that holds a *different*
    protocol raises :class:`ProtocolDrift` unless ``overwrite=True``.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        drift = _read_lock(path).diff(protocol)
        if not drift:
            return
        raise ProtocolDrift(drift, lock_path=path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(_lock_document(protocol), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def load_locked(path: Path, *, live: Protocol | None = None) -> Protocol:
    """Return the locked protocol, or raise :class:`ProtocolDrift` if ``live`` differs from it."""
    locked = _read_lock(path)
    drift = locked.diff(live) if live is not None else {}
    if drift:
        raise ProtocolDrift(drift, lock_path=Path(path))
    return locked
