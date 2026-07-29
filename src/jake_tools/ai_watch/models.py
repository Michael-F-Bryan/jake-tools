from __future__ import annotations

import hashlib
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class AuditStage(StrEnum):
    DISCOVERED = "discovered"
    SEEN_CHECK = "seen_check"
    FETCHED = "fetched"
    SCOUTED = "scouted"
    CURATED = "curated"
    OBSIDIAN_SYNCED = "obsidian_synced"
    DELIVERED = "delivered"


class CuratorDecisionType(StrEnum):
    SURFACE = "surface"
    SPECULATIVE_WATCH = "speculative_watch"
    REJECT = "reject"
    DUPLICATE = "duplicate"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class DigestLane(StrEnum):
    MAIN_DIGEST = "main_digest"
    SPECULATIVE_WATCH = "speculative_watch"


class ScoutRecommendation(StrEnum):
    PROMOTE_TO_CURATOR = "promote_to_curator"
    HOLD = "hold"
    REJECT = "reject"


class RunStatus(StrEnum):
    OK = "ok"
    FAIL = "fail"


def candidate_id_for(*, url: str) -> str:
    # Identity is canonical-URL-only by design: titles are mutable editorial
    # copy and must never affect which candidate a URL maps to.
    canonical = url.strip().rstrip("/").lower()
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def content_hash_for(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


class AiWatchStageError(Exception):
    """Raised when a pipeline stage finishes with invalid records.

    Carries the stage name and the full list of per-record validation errors
    so callers (CLI, runner) can report specifics instead of a flattened,
    unattributed string.
    """

    def __init__(self, *, stage: str, errors: list[str]) -> None:
        super().__init__(f"{stage}: " + "; ".join(errors))
        self.stage = stage
        self.errors = errors


class VaultPathEscapeError(ValueError):
    """Raised when a vault-relative path resolves outside the vault root."""

    def __init__(self, *, vault: Path, resolved: Path) -> None:
        super().__init__(f"note path {resolved} escapes vault root {vault}")
        self.vault = vault
        self.resolved = resolved


def resolve_vault_path(*, vault: Path, rel_path: str) -> Path:
    """Resolve `rel_path` against `vault`, rejecting absolute or `..` escapes."""
    candidate = (vault / rel_path).resolve()
    if not candidate.is_relative_to(vault.resolve()):
        raise VaultPathEscapeError(vault=vault, resolved=candidate)
    return candidate


class SearchResult(BaseModel):
    url: str
    title: str
    description: str = ""
    source: str


class ExtractResult(BaseModel):
    url: str
    title: str
    content: str
    full_text_path: str | None = None
    status: Literal["ok", "fail"] = "ok"
    error: str | None = None


class ScoutOutput(BaseModel):
    tags: list[str] = Field(default_factory=list)
    fit_score: int = Field(ge=1, le=5)
    novelty_score: int = Field(ge=1, le=5)
    practicality_score: int = Field(ge=1, le=5)
    noise_risk: int = Field(ge=1, le=5)
    evidence_quotes: list[str] = Field(default_factory=list)
    recommendation: ScoutRecommendation
    reason: str
    summary: str


class ObsidianRecommendation(BaseModel):
    should_create_note: bool
    path: str = ""
    placement_reason: str = ""


class CuratorDecision(BaseModel):
    decision: CuratorDecisionType
    lane: DigestLane | None = None
    reason: str
    digest_summary: str = ""
    reject_reason: str | None = None
    obsidian_recommendation: ObsidianRecommendation = Field(
        default_factory=lambda: ObsidianRecommendation(should_create_note=False)
    )


class StageFailure(BaseModel):
    """A single pipeline stage's failure, with its own name and cause.

    Replaces flattening every stage failure into an unattributed
    `f"{stage}: {error}"` string, which loses the ability to tell stages
    apart programmatically (e.g. in JSON output).
    """

    stage: str
    error: str


class AiWatchCommandResult(BaseModel):
    status: RunStatus
    run_id: str
    root: Path
    digest_path: Path | None = None
    summary_path: Path | None = None
    surfaced_count: int = 0
    speculative_count: int = 0
    failed_stages: list[StageFailure] = Field(default_factory=list)


class AiWatchCommandOptions(BaseModel):
    target_date: date
    base_dir: Path = Field(default_factory=lambda: Path.cwd() / "_working")
    scout_model: str = "claude-haiku-4-5"
    curator_model: str = "claude-sonnet-5"
    vault_path: Path = Path("/Users/work/Documents/Vault")
    discord_target: str = ""
    dry_run: bool = False
    max_candidates: int = 80
    surface_limit: int | None = 2
    max_article_age_days: int = 90
    cost_cap_usd: float | None = None
    calibration_only: bool = False
    force_candidate: str | None = None
