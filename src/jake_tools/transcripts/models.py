from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
