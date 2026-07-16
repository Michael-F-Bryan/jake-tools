from __future__ import annotations

import re

from .models import TranscriptTurn

_WORD_REPEAT_RE = re.compile(
    r"\b(?P<word>[A-Za-z']+)(?:\s+(?P=word)\b)+", re.IGNORECASE
)
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:?!])")


def _normalise_text(text: str) -> str:
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    cleaned = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", cleaned)
    cleaned = _WORD_REPEAT_RE.sub(lambda match: match.group("word"), cleaned)
    return cleaned


def normalise_turns(turns: list[TranscriptTurn]) -> list[TranscriptTurn]:
    normalised: list[TranscriptTurn] = []
    for turn in turns:
        text = _normalise_text(turn.text)
        if not text:
            continue

        candidate = TranscriptTurn(
            start=turn.start,
            end=turn.end,
            speaker=turn.speaker,
            text=text,
        )
        if (
            normalised
            and normalised[-1].speaker == candidate.speaker
            and normalised[-1].text == candidate.text
            and candidate.start < normalised[-1].end
        ):
            normalised[-1] = TranscriptTurn(
                start=normalised[-1].start,
                end=max(normalised[-1].end, candidate.end),
                speaker=candidate.speaker,
                text=candidate.text,
            )
            continue

        normalised.append(candidate)

    return normalised


def _join_turn_text(current: str, following: str) -> str:
    if current.endswith(("-", "—")):
        return f"{current.rstrip()} {following}"
    if current.endswith((".", "?", "!")):
        return f"{current} {following}"
    return f"{current}. {following}"


def merge_consecutive_turns(
    turns: list[TranscriptTurn], *, max_gap_seconds: float = 3.0
) -> list[TranscriptTurn]:
    if not turns:
        return []

    merged: list[TranscriptTurn] = [turns[0]]
    for turn in turns[1:]:
        current = merged[-1]
        gap = max(0.0, turn.start - current.end)
        if current.speaker == turn.speaker and gap <= max_gap_seconds:
            merged[-1] = TranscriptTurn(
                start=current.start,
                end=max(current.end, turn.end),
                speaker=current.speaker,
                text=_join_turn_text(current.text, turn.text),
            )
            continue

        merged.append(turn)

    return merged
