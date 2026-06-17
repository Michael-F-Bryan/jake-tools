from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class RecordingRef(BaseModel):
    raw_link: str
    resolved_path: Path
    created_at: datetime


class SourceNote(BaseModel):
    path: Path
    body: str
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


class MeetingMinutes(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class MergeReport(BaseModel):
    status: str
    transcript_heading_count: int
    chapter_count: int
    original_content_preserved: bool
    recording_embed_preserved: bool
