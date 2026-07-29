from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from .models import (
    AuditStage,
    CuratorDecision,
    CuratorDecisionType,
    DigestLane,
    ObsidianRecommendation,
    RunStatus,
    ScoutOutput,
    ScoutRecommendation,
    StageFailure,
)


class FetchStatus(StrEnum):
    OK = "ok"
    FAIL = "fail"
    SKIPPED = "skipped"


class DeliveryStatus(StrEnum):
    SENT = "sent"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


class ObsidianSyncStatus(StrEnum):
    CREATED = "created"
    DRY_RUN = "dry_run"


class ArticleMetadata(BaseModel):
    candidate_id: str
    url: str
    title: str
    source: str
    content_hash: str
    full_text_path: str | None = None
    status: Literal["ok", "fail"] = "ok"
    error: str | None = None


class SeenCandidateRecord(BaseModel):
    candidate_id: str
    canonical_url: str
    url: str
    title: str
    sources: list[str] = Field(default_factory=list)
    first_seen_at: datetime
    last_seen_at: datetime
    latest_decision: CuratorDecisionType | None = None
    content_hash: str | None = None
    latest_content_path: str | None = None
    obsidian_note_path: str | None = None
    duplicate_of: str | None = None


class DiscoveredRecord(BaseModel):
    run_id: str
    candidate_id: str
    stage: Literal[AuditStage.DISCOVERED] = AuditStage.DISCOVERED
    timestamp: datetime
    source: str
    url: str
    title: str
    description: str = ""


class SeenCheckRecord(BaseModel):
    run_id: str
    candidate_id: str
    stage: Literal[AuditStage.SEEN_CHECK] = AuditStage.SEEN_CHECK
    timestamp: datetime
    status: Literal["already_seen"] = "already_seen"
    matched_on: str = "canonical_url"
    first_seen_at: datetime | None = None
    latest_decision: CuratorDecisionType | None = None
    action: Literal["skip_fetch"] = "skip_fetch"


class FetchRecord(BaseModel):
    run_id: str
    candidate_id: str
    stage: Literal[AuditStage.FETCHED] = AuditStage.FETCHED
    timestamp: datetime
    status: FetchStatus
    content_path: str | None = None
    metadata_path: str | None = None
    content_hash: str | None = None
    error: str | None = None
    reason: str | None = None


class ScoutEvaluationRecord(BaseModel):
    run_id: str
    candidate_id: str
    stage: Literal[AuditStage.SCOUTED] = AuditStage.SCOUTED
    timestamp: datetime
    model: str
    tags: list[str] = Field(default_factory=list)
    fit_score: int
    novelty_score: int
    practicality_score: int
    noise_risk: int
    evidence_quotes: list[str] = Field(default_factory=list)
    recommendation: ScoutRecommendation
    reason: str
    summary: str

    @classmethod
    def from_output(
        cls,
        *,
        run_id: str,
        candidate_id: str,
        timestamp: datetime,
        model: str,
        output: ScoutOutput,
    ) -> ScoutEvaluationRecord:
        return cls(
            run_id=run_id,
            candidate_id=candidate_id,
            timestamp=timestamp,
            model=model,
            **output.model_dump(),
        )


class ScoutFailureRecord(BaseModel):
    run_id: str
    candidate_id: str
    timestamp: datetime
    errors: list[str]


class CuratorDecisionRecord(BaseModel):
    run_id: str
    candidate_id: str
    stage: Literal[AuditStage.CURATED] = AuditStage.CURATED
    timestamp: datetime
    model: str
    decision: CuratorDecisionType
    lane: DigestLane | None = None
    reason: str
    digest_summary: str = ""
    reject_reason: str | None = None
    obsidian_recommendation: ObsidianRecommendation = Field(
        default_factory=lambda: ObsidianRecommendation(should_create_note=False)
    )

    @classmethod
    def from_decision(
        cls,
        *,
        run_id: str,
        candidate_id: str,
        timestamp: datetime,
        model: str,
        decision: CuratorDecision,
    ) -> CuratorDecisionRecord:
        return cls(
            run_id=run_id,
            candidate_id=candidate_id,
            timestamp=timestamp,
            model=model,
            **decision.model_dump(),
        )


class ObsidianSyncRecord(BaseModel):
    run_id: str
    candidate_id: str
    stage: Literal[AuditStage.OBSIDIAN_SYNCED] = AuditStage.OBSIDIAN_SYNCED
    timestamp: datetime
    status: ObsidianSyncStatus
    note_path: str
    placement_reason: str | None = None


class DeliveryRecord(BaseModel):
    run_id: str
    stage: Literal[AuditStage.DELIVERED] = AuditStage.DELIVERED
    timestamp: datetime
    target: str
    status: DeliveryStatus
    digest_path: str | None = None
    payload_path: str | None = None
    reason: str | None = None
    surfaced_count: int = 0
    speculative_count: int = 0


class RunManifestPaths(BaseModel):
    root: str
    digest: str
    summary: str


class RunManifestCounts(BaseModel):
    candidates: int
    fetched: int
    scouted: int
    curated: int
    surfaced: int
    speculative: int


class RunManifest(BaseModel):
    run_id: str
    status: RunStatus
    paths: RunManifestPaths
    counts: RunManifestCounts
    failed_stages: list[StageFailure] = Field(default_factory=list)
    dry_run: bool = False


class CalibrationCase(BaseModel):
    id: str
    url: str
    title_hint: str
    expected_tags: list[str] = Field(default_factory=list)
    min_fit_score: int = 4
    expected_curator_decision: CuratorDecisionType
    expected_lane: DigestLane | None = None


class ObsidianPreview(BaseModel):
    note_path: str
    preview: str
