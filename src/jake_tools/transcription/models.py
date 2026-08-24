"""Typed intermediates for the meeting-transcription pipeline.

Every stage of the transcription pipeline (merge audio, ASR+diarise, resolve
speakers, chapterise, polish, minutes, integrate) is both a library function
and a ``jake-tools transcript`` sub-command that reads/writes JSON. These
models are the contract between those two: every one of them round-trips
through ``model_dump_json``/``model_validate_json`` unchanged, which is what
lets a plumbing sub-command re-hydrate a stage's output for the next stage.

Timestamp fields are meeting-relative seconds throughout; rendering to
``HH:MM`` happens at integrate time, not here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..cache_models import CacheEnvelope


class StageTiming(BaseModel):
    """Durable timing and provenance for one local pipeline stage."""

    stage: str
    started_at: datetime
    ended_at: datetime
    elapsed_seconds: float
    cache_hit: bool = False
    device: str | None = None
    model: str | None = None
    config: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    media_duration_seconds: float | None = None
    rtf: float | None = None
    # Local stages are explicitly zero API cost; this is not an invoice field.
    api_rate_cost: float = 0.0
    telemetry_schema_version: int = 1


class StageTimingLog(CacheEnvelope):
    """The append-only local-stage timing document in a run cache."""

    stages: list[StageTiming]


class StaleState(CacheEnvelope):
    """Downstream artefacts invalidated by an upstream human correction."""

    reason: str
    artefacts: list[str] = Field(default_factory=list)


NoteContext = Literal["meeting", "ops-log"]
"""Which kind of Obsidian note a run is producing. Detection lands in plan 002."""


class SourceClip(BaseModel):
    """One source audio file contributing to the merged recording."""

    path: str  # vault-relative or absolute path to the source clip
    offset_seconds: float  # position of this clip's start in the merged audio
    duration_seconds: float


class Utterance(BaseModel):
    """One diarised span of speech in the merged audio."""

    start: float  # seconds, meeting-relative (cumulative across clips)
    end: float
    speaker: str  # "SPEAKER_00"-style cluster id, later a real name or "Unknown"
    text: str


class RawTranscript(CacheEnvelope):
    """The output of ASR+diarisation, before speakers are resolved.

    `utterances` is a total order by non-decreasing `start` (tiebroken by
    `(start, end, speaker)`), not a guarantee of disjoint spans: genuine
    cross-talk — two diarised speakers overlapping in time — produces
    utterances that overlap too (see `transcription/asr.py`'s `align()`).
    That overlap is real information for the polish stage, not noise to be
    clamped away.
    """

    clips: list[SourceClip]
    utterances: list[Utterance]
    audio_sha256: str | None = (
        None  # hash of the merged audio; None for pre-diarised sources
    )
    source_text_sha256: str | None = None
    asr_model: str | None = None
    diarisation_model: str | None = None
    diarisation_device: str | None = None
    num_speakers: int | None = None
    asr_chunk_duration: float | None = None
    asr_chunk_overlap: float | None = None
    timings: list[StageTiming] = Field(default_factory=list)


class SpeakerAssignment(BaseModel):
    """A resolved mapping from one diarisation cluster to a human name."""

    cluster: str  # "SPEAKER_03"
    name: str  # "Nikki Staltari" or "Unknown"


class SpeakerCorrection(BaseModel):
    """A human attribution for one half-open time range of one cluster."""

    cluster: str
    start_seconds: float
    end_seconds: float
    name: str


class TimestampedSegment(BaseModel):
    """Verbatim, timestamped transcript evidence attached to a request."""

    start_seconds: float
    end_seconds: float
    speaker: str
    text: str


SnippetSegment = TimestampedSegment


class SnippetRequest(BaseModel):
    """One unresolved cluster and the evidence a human needs to identify it."""

    cluster: str
    clip_paths: list[str]  # ffmpeg-cut snippet files on disk
    context: str  # adjacent resolved dialogue, to jog memory
    segments: list[TimestampedSegment] = Field(default_factory=list)


class ChapterSpan(BaseModel):
    """One chapter's boundaries within the raw utterance sequence."""

    title: str
    start_utterance: int  # index into RawTranscript.utterances, inclusive
    end_utterance: int  # inclusive
    start_seconds: float


DroppedSourceReason = Literal[
    "filler-only backchannel",
    "immediate duplicate ASR fragment",
    "unintelligible fragment with no recoverable substance",
]


class DroppedSourceTurn(BaseModel):
    """A source turn deliberately omitted by polish, with an allowed reason."""

    source_turn_index: int
    reason: DroppedSourceReason


class PolishedTurn(BaseModel):
    """One cleaned-up turn of dialogue in a polished chapter."""

    speaker: str
    text: str
    source_turn_indices: list[int] = Field(default_factory=list)


class PolishedChapter(BaseModel):
    """A chapter after the polish stage has cleaned up its dialogue."""

    title: str
    start_seconds: float
    summary: str  # per-chapter summary callout text
    turns: list[PolishedTurn]
    dropped_source_turns: list[DroppedSourceTurn] = Field(default_factory=list)


class TranscriptProducts(BaseModel):
    """Everything the integrate step writes into the note."""

    context: NoteContext
    meeting_summary: str
    discussion_notes: str  # markdown bullet body for ## Discussion Notes
    chapters: list[PolishedChapter]
