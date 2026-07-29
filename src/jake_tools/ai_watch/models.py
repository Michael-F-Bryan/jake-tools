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


def candidate_id_for(*, url: str, title: str = "") -> str:
    canonical = url.strip().rstrip("/").lower()
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def content_hash_for(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


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


class StageResult(BaseModel):
    status: RunStatus
    message: str = ""
    count: int = 0


class AiWatchCommandResult(BaseModel):
    status: RunStatus
    run_id: str
    root: Path
    digest_path: Path | None = None
    summary_path: Path | None = None
    surfaced_count: int = 0
    speculative_count: int = 0
    failed_stages: list[str] = Field(default_factory=list)


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
    save_raw: bool = False
    force_candidate: str | None = None
