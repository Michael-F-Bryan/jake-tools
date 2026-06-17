from __future__ import annotations

from ..hermes import Hermes
from .models import ChaptersPayload, MeetingMinutes, SourceNote, SpeakerIdentity, SpeakerMapping, TranscriptTurn, TranscriptTurnsPayload
from .prompts import (
    render_chaptering_prompt,
    render_minutes_prompt,
    render_speaker_mapping_prompt,
    render_transcript_polish_prompt,
)


def _serialise_turns(turns: list[TranscriptTurn]) -> list[dict]:
    return [turn.model_dump(mode="json") for turn in turns]


def _ordered_speaker_labels(turns: list[TranscriptTurn]) -> list[str]:
    labels: list[str] = []
    for turn in turns:
        if turn.speaker not in labels:
            labels.append(turn.speaker)
    return labels


def _fallback_two_party_call_mapping(source: SourceNote, turns: list[TranscriptTurn]) -> SpeakerMapping | None:
    if "call" not in source.title.lower():
        return None

    labels = _ordered_speaker_labels(turns)
    if len(labels) != 2 or len(source.attendees) != 2:
        return None

    michael = next((attendee for attendee in source.attendees if attendee.lower() == "michael bryan"), None)
    if michael is None:
        return None

    other_attendee = next(attendee for attendee in source.attendees if attendee != michael)
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


def run_speaker_mapping_stage(hermes: Hermes, source: SourceNote, turns: list[TranscriptTurn]) -> SpeakerMapping:
    mapping = hermes.oneshot_structured(
        render_speaker_mapping_prompt(source.title, source.attendees, _serialise_turns(turns)),
        SpeakerMapping,
    )
    if mapping.mapping:
        return mapping

    fallback = _fallback_two_party_call_mapping(source, turns)
    if fallback is not None:
        return fallback

    return mapping


def run_transcript_polish_stage(
    hermes: Hermes,
    source: SourceNote,
    turns: list[TranscriptTurn],
    speaker_mapping: SpeakerMapping,
) -> list[TranscriptTurn]:
    payload = hermes.oneshot_structured(
        render_transcript_polish_prompt(
            source.title,
            source.attendees,
            {label: identity.name for label, identity in speaker_mapping.mapping.items()},
            _serialise_turns(turns),
        ),
        TranscriptTurnsPayload,
    )
    if len(payload.turns) != len(turns):
        return turns
    return payload.turns


def run_chaptering_stage(hermes: Hermes, turns: list[TranscriptTurn]) -> ChaptersPayload:
    return hermes.oneshot_structured(render_chaptering_prompt(_serialise_turns(turns)), ChaptersPayload)



def run_minutes_stage(
    hermes: Hermes,
    turns: list[TranscriptTurn],
    chapters: ChaptersPayload | None = None,
) -> MeetingMinutes:
    return hermes.oneshot_structured(
        render_minutes_prompt(
            _serialise_turns(turns),
            chapters.model_dump(mode="json").get("chapters") if chapters else None,
        ),
        MeetingMinutes,
    )
