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


class MergeReport(BaseModel):
    status: str
    transcript_heading_count: int
    chapter_count: int
    original_content_preserved: bool
    recording_embed_preserved: bool


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
        }


class SourceArtifact(BaseModel):
    kind: str
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
    metadata: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ExternalCommandMetadata(BaseModel):
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class AudioArtifact(BaseModel):
    input_files: list[Path]
    merged_audio_path: Path | None = None
    transcript_json_path: Path | None = None
    commands: list[ExternalCommandMetadata] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TranscriptSourceRef(BaseModel):
    turn_index: int
    source_ref: str


class TranscriptArtifact(BaseModel):
    turns: list[TranscriptTurn]
    source_refs: list[TranscriptSourceRef] = Field(default_factory=list)
    speakers: dict[str, SpeakerIdentity] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class PlannedChapter(BaseModel):
    start: float
    end: float
    title: str
    summary: str


class ChapterPlan(BaseModel):
    chapters: list[PlannedChapter]
    boundary_source: Literal["deterministic", "llm", "manual"]


class RenderedNote(BaseModel):
    rendered_markdown_path: Path
    destination_path: Path | None = None
    sections_included: list[str] = Field(default_factory=list)
    attachments_or_source_links: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    command_metadata: list[ExternalCommandMetadata] = Field(default_factory=list)
    ai_totals: AITotals | None = None


PUBLIC_TRANSCRIPT_ARTIFACT_MODELS: tuple[type[BaseModel], ...] = (
    SourceArtifact,
    AudioArtifact,
    TranscriptArtifact,
    ChapterPlan,
    RenderedNote,
    VerificationReport,
    RunManifest,
)

_PUBLIC_TRANSCRIPT_ARTIFACTS_BY_NAME: dict[str, type[BaseModel]] = {
    model.__name__: model for model in PUBLIC_TRANSCRIPT_ARTIFACT_MODELS
}


def list_public_transcript_artifact_models() -> tuple[type[BaseModel], ...]:
    return PUBLIC_TRANSCRIPT_ARTIFACT_MODELS


def resolve_public_transcript_artifact_model(
    model_name: str,
) -> type[BaseModel] | None:
    return _PUBLIC_TRANSCRIPT_ARTIFACTS_BY_NAME.get(model_name)


def example_for_public_transcript_artifact(model_name: str) -> BaseModel | None:
    if model_name == SourceArtifact.__name__:
        return SourceArtifact(kind="plain-text")
    if model_name == AudioArtifact.__name__:
        return AudioArtifact(input_files=[Path("recording.m4a")])
    if model_name == TranscriptArtifact.__name__:
        return TranscriptArtifact(
            turns=[
                TranscriptTurn(
                    start=0.0,
                    end=3.2,
                    speaker="Speaker 1",
                    text="Hello world.",
                )
            ]
        )
    if model_name == ChapterPlan.__name__:
        return ChapterPlan(
            boundary_source="deterministic",
            chapters=[
                PlannedChapter(
                    start=0.0,
                    end=60.0,
                    title="Introduction",
                    summary="Meeting opened and agenda confirmed.",
                )
            ],
        )
    if model_name == RenderedNote.__name__:
        return RenderedNote(rendered_markdown_path=Path("meeting.md"))
    if model_name == VerificationReport.__name__:
        return VerificationReport(
            status="pass",
            checks=[
                VerificationCheck(
                    check_id="turn-order",
                    status="pass",
                    message="Turns are ordered by start time.",
                )
            ],
        )
    if model_name == RunManifest.__name__:
        return RunManifest(
            run_id="run-001",
            stages=[RunStageStatus(stage="parse", status="pass")],
        )
    return None
