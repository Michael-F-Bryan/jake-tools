from __future__ import annotations

from ..hermes import Hermes
from .models import ChaptersPayload, MeetingMinutes, SpeakerMapping, TranscriptTurn
from .prompts import (
    render_chaptering_prompt,
    render_minutes_prompt,
    render_speaker_mapping_prompt,
)


def _serialise_turns(turns: list[TranscriptTurn]) -> list[dict]:
    return [turn.model_dump(mode="json") for turn in turns]


def run_speaker_mapping_stage(hermes: Hermes, turns: list[TranscriptTurn]) -> SpeakerMapping:
    return hermes.oneshot_structured(render_speaker_mapping_prompt(_serialise_turns(turns)), SpeakerMapping)



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
