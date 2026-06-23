from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from ..hermes import Hermes, Reply
from ..prompting import StructuredPrompt
from .models import (
    Chapter,
    ChaptersPayload,
    MeetingMinutes,
    SourceNote,
    SpeakerIdentity,
    SpeakerMapping,
    TranscriptTurn,
    TranscriptTurnsPayload,
)


class SpeakerMappingPrompt(StructuredPrompt[SpeakerMapping]):
    response_model: ClassVar[type[BaseModel]] = SpeakerMapping
    template: ClassVar[str] = """
You are the speaker-mapping specialist for an Obsidian meeting recording workflow.

Meeting title: {{ title }}
Attendees: {{ attendees | json }}

Infer speaker names conservatively from the transcript turns below. Prefer the attendee list when the transcript supports it. If you cannot justify a real name from the transcript or meeting context, leave the speaker unresolved.

Transcript turns:
{{ turns | json }}
"""

    title: str
    attendees: list[str]
    turns: list[TranscriptTurn]


class ChapteringPrompt(StructuredPrompt[ChaptersPayload]):
    response_model: ClassVar[type[BaseModel]] = ChaptersPayload
    template: ClassVar[str] = """
You are the chaptering specialist for an Obsidian meeting recording workflow.

Create broad thematic chapters for the transcript. Prefer topic shifts and agenda changes over rigid time slices. Chapters must be contiguous, in order, and cover the whole transcript.

Transcript turns:
{{ turns | json }}
"""

    turns: list[TranscriptTurn]


class MeetingMinutesPrompt(StructuredPrompt[MeetingMinutes]):
    response_model: ClassVar[type[BaseModel]] = MeetingMinutes
    template: ClassVar[str] = """
You are the meeting-minutes specialist for an Obsidian meeting recording workflow.

Write faithful high-level meeting notes from the transcript. Preserve uncertainty when the transcript is unclear.
Prefer concrete outcomes, instructions, appointments, and next steps over generic summary prose.

Transcript turns:
{{ turns | json }}
{% if chapters %}
Chapter plan:
{{ chapters | json }}
{% endif %}
"""

    turns: list[TranscriptTurn]
    chapters: list[Chapter] | None = None


class TranscriptPolishPrompt(StructuredPrompt[TranscriptTurnsPayload]):
    response_model: ClassVar[type[BaseModel]] = TranscriptTurnsPayload
    template: ClassVar[str] = """
You are the transcript-polishing specialist for an Obsidian meeting recording workflow.

Meeting title: {{ title }}
Attendees: {{ attendees | json }}
Speaker mapping: {{ speaker_mapping | json }}

Rewrite each transcript turn conservatively for readability.

Rules:
- keep the same order, speaker labels, `start`, and `end`
- preserve the meaning of each turn
- remove filler-noise, repeated words, and obvious ASR junk when the intended wording is clear
- improve punctuation and grammar when safe
- do not invent facts, names, diagnoses, or commitments
- if wording is uncertain, keep it close to the source rather than guessing

Transcript turns:
{{ turns | json }}
"""

    title: str
    attendees: list[str]
    speaker_mapping: dict[str, str]
    turns: list[TranscriptTurn]


def _ordered_speaker_labels(turns: list[TranscriptTurn]) -> list[str]:
    labels: list[str] = []
    for turn in turns:
        if turn.speaker not in labels:
            labels.append(turn.speaker)
    return labels


def _fallback_two_party_call_mapping(
    source: SourceNote, turns: list[TranscriptTurn]
) -> SpeakerMapping | None:
    if "call" not in source.title.lower():
        return None

    labels = _ordered_speaker_labels(turns)
    if len(labels) != 2 or len(source.attendees) != 2:
        return None

    michael = next(
        (
            attendee
            for attendee in source.attendees
            if attendee.lower() == "michael bryan"
        ),
        None,
    )
    if michael is None:
        return None

    other_attendee = next(
        attendee for attendee in source.attendees if attendee != michael
    )
    return SpeakerMapping(
        mapping={
            labels[0]: SpeakerIdentity(
                name=other_attendee,
                confidence=0.55,
                reason="fallback two-party call mapping based on attendee list and turn order",
            ),
            labels[1]: SpeakerIdentity(
                name=michael,
                confidence=0.55,
                reason="fallback two-party call mapping based on attendee list and turn order",
            ),
        },
        unresolved=[],
        notes="Used deterministic fallback speaker mapping for a two-party call.",
    )


def run_speaker_mapping_stage(
    hermes: Hermes, source: SourceNote, turns: list[TranscriptTurn]
) -> tuple[SpeakerMapping, Reply]:
    mapping, reply = hermes.run_structured(
        SpeakerMappingPrompt(
            title=source.title, attendees=source.attendees, turns=turns
        )
    )
    if mapping.mapping:
        return mapping, reply

    fallback = _fallback_two_party_call_mapping(source, turns)
    if fallback is not None:
        return fallback, reply

    return mapping, reply


def run_transcript_polish_stage(
    hermes: Hermes,
    source: SourceNote,
    turns: list[TranscriptTurn],
    speaker_mapping: SpeakerMapping,
) -> tuple[list[TranscriptTurn], Reply]:
    payload, reply = hermes.run_structured(
        TranscriptPolishPrompt(
            title=source.title,
            attendees=source.attendees,
            speaker_mapping={
                label: identity.name
                for label, identity in speaker_mapping.mapping.items()
            },
            turns=turns,
        )
    )
    if len(payload.turns) != len(turns):
        return turns, reply
    return payload.turns, reply


def run_chaptering_stage(
    hermes: Hermes, turns: list[TranscriptTurn]
) -> tuple[ChaptersPayload, Reply]:
    return hermes.run_structured(ChapteringPrompt(turns=turns))


def run_minutes_stage(
    hermes: Hermes,
    turns: list[TranscriptTurn],
    chapters: ChaptersPayload | None = None,
) -> tuple[MeetingMinutes, Reply]:
    return hermes.run_structured(
        MeetingMinutesPrompt(
            turns=turns, chapters=chapters.chapters if chapters else None
        )
    )
