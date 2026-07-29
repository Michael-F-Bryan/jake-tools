from __future__ import annotations

import datetime as dt
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..ai_usage import AIStageStats, AITotals


class RecordingRef(BaseModel):
    raw_link: str
    resolved_path: Path
    created_at: datetime


class SourceNote(BaseModel):
    path: Path
    title: str
    body: str
    attendees: list[str] = Field(default_factory=list)
    recordings: list[RecordingRef] = Field(default_factory=list)

    def to_source_artifact(self) -> SourceArtifact:
        return SourceArtifact(
            kind="obsidian-note",
            source_path=self.path,
            title=self.title,
            attendees=list(self.attendees),
            attachments=[recording.resolved_path for recording in self.recordings],
        )


class ConcatPlan(BaseModel):
    inputs_in_creation_order: list[Path]
    output_merged_audio: Path


class ScribeRunReport(BaseModel):
    input_audio: Path
    output_json: Path
    format: str = "json"
    warnings: list[str] = Field(default_factory=list)
    segment_count: int | None = None
    stdout: str = ""
    stderr: str = ""


class TranscriptTurn(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class TranscriptTurnsPayload(BaseModel):
    turns: list[TranscriptTurn] = Field(default_factory=list)


class SpeakerIdentity(BaseModel):
    name: str
    confidence: float
    reason: str


class SpeakerMapping(BaseModel):
    mapping: dict[str, SpeakerIdentity] = Field(default_factory=dict)
    unresolved: list[str] = Field(default_factory=list)
    notes: str = ""


class Chapter(BaseModel):
    title: str
    start: float
    end: float
    summary: str


class ChapterSummary(BaseModel):
    title: str
    start_timestamp: str
    end_timestamp: str


class ChaptersPayload(BaseModel):
    chapters: list[Chapter] = Field(default_factory=list)


class MeetingMinutes(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)


class SourceOverview(BaseModel):
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1)


class SpeakerMessageCount(BaseModel):
    speaker: str
    messages: int


class CoordinatorResult(BaseModel):
    note_path: Path
    updated: bool
    chapter_summaries: list[ChapterSummary] = Field(default_factory=list)
    ai_stage_stats: list[AIStageStats] = Field(default_factory=list)
    ai_totals: AITotals = Field(default_factory=AITotals)
    speaker_message_counts: list[SpeakerMessageCount] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def json_summary(self) -> dict[str, object]:
        return {
            "chapter_summaries": [
                summary.model_dump(mode="json") for summary in self.chapter_summaries
            ],
            "ai_stage_stats": [
                stats.model_dump(mode="json") for stats in self.ai_stage_stats
            ],
            "ai_totals": self.ai_totals.model_dump(mode="json"),
            "speaker_message_counts": [
                count.model_dump(mode="json") for count in self.speaker_message_counts
            ],
            "warnings": list(self.warnings),
        }


SourceKind = Literal["obsidian-note", "youtube", "msgraph-teams"]


class SourceArtifact(BaseModel):
    kind: SourceKind
    source_path: Path | None = None
    source_url: str | None = None
    message_id: str | None = None
    title: str | None = None
    date: dt.date | None = None
    attendees: list[str] = Field(default_factory=list)
    organisation: str | None = None
    project: str | None = None
    attachments: list[Path] = Field(default_factory=list)
    raw_text_path: Path | None = None
    # Kind-specific capture details (e.g. YoutubeCaptureMetadata for
    # kind="youtube") live here as a plain dict because different source
    # kinds populate genuinely different shapes. Callers that know the kind
    # should parse this via the matching typed model instead of indexing it
    # directly.
    metadata: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class YoutubeCaptureMetadata(BaseModel):
    """Typed view of ``SourceArtifact.metadata`` for ``kind="youtube"``."""

    video_id: str
    channel: str
    channel_id: str = ""
    duration_seconds: int = 0
    language: str = ""
    subtitle_track: str
    subtitle_kind: Literal["manual", "automatic"]
    capture_method: str = "yt-dlp"


class TranscriptSourceRef(BaseModel):
    turn_index: int
    source_ref: str


class TranscriptArtifact(BaseModel):
    turns: list[TranscriptTurn]
    source_refs: list[TranscriptSourceRef] = Field(default_factory=list)
    speakers: dict[str, SpeakerIdentity] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ChapterPlan(BaseModel):
    chapters: list[Chapter]
    boundary_source: Literal["deterministic", "llm", "manual"]


class SourceNotePlan(BaseModel):
    overview: SourceOverview
    chapters: ChapterPlan


class VerificationCheck(BaseModel):
    check_id: str
    status: Literal["pass", "fail", "blocked"]
    message: str


class VerificationReport(BaseModel):
    status: Literal["pass", "fail", "blocked"]
    checks: list[VerificationCheck]
    failed_gate_ids: list[str] = Field(default_factory=list)
    affected_artifact_paths: list[Path] = Field(default_factory=list)
    next_action: str | None = None


class RunStageStatus(BaseModel):
    stage: str
    status: Literal["pending", "running", "pass", "fail", "blocked", "skipped"]
    artefacts: list[Path] = Field(default_factory=list)


class RunManifest(BaseModel):
    run_id: str
    stages: list[RunStageStatus]
    artefact_paths: dict[str, Path] = Field(default_factory=dict)
    ai_totals: AITotals | None = None
    warnings: list[str] = Field(default_factory=list)


class YoutubeSourceNoteResult(BaseModel):
    title: str | None = None
    note: Path | None = None
    updated: bool
    out_dir: Path
    rendered_note: Path
    manifest: Path
    verification_status: str
    subtitle_track: str | None = None
    subtitle_kind: str | None = None


class TeamsMeetingResult(BaseModel):
    note: Path | None = None
    updated: bool
    out_dir: Path
    rendered_note: Path
    raw_vtt: Path
    manifest: Path
