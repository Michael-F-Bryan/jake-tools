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

from typing import Literal

from pydantic import BaseModel

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


class RawTranscript(BaseModel):
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


class SpeakerAssignment(BaseModel):
    """A resolved mapping from one diarisation cluster to a human name."""

    cluster: str  # "SPEAKER_03"
    name: str  # "Nikki Staltari" or "Unknown"


class SnippetRequest(BaseModel):
    """One unresolved cluster and the evidence a human needs to identify it."""

    cluster: str
    clip_paths: list[str]  # ffmpeg-cut snippet files on disk
    context: str  # adjacent resolved dialogue, to jog memory


class ChapterSpan(BaseModel):
    """One chapter's boundaries within the raw utterance sequence."""

    title: str
    start_utterance: int  # index into RawTranscript.utterances, inclusive
    end_utterance: int  # inclusive
    start_seconds: float


class PolishedTurn(BaseModel):
    """One cleaned-up turn of dialogue in a polished chapter."""

    speaker: str
    text: str


class PolishedChapter(BaseModel):
    """A chapter after the polish stage has cleaned up its dialogue."""

    title: str
    start_seconds: float
    summary: str  # per-chapter summary callout text
    turns: list[PolishedTurn]


class TranscriptProducts(BaseModel):
    """Everything the integrate step writes into the note."""

    context: NoteContext
    meeting_summary: str
    discussion_notes: str  # markdown bullet body for ## Discussion Notes
    chapters: list[PolishedChapter]
