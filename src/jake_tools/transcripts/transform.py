from __future__ import annotations

import re

from .errors import TranscriptError
from .models import Chapter, ChapterPlan, TranscriptArtifact, TranscriptTurn

_WORD_REPEAT_RE = re.compile(
    r"\b(?P<word>[A-Za-z']+)(?:\s+(?P=word)\b)+", re.IGNORECASE
)
_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:?!])")


class TransformPrimitiveError(TranscriptError):
    pass


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


def normalise_transcript_artifact(artifact: TranscriptArtifact) -> TranscriptArtifact:
    return artifact.model_copy(update={"turns": normalise_turns(artifact.turns)})


def merge_adjacent_turns(
    artifact: TranscriptArtifact, *, max_gap_seconds: float
) -> TranscriptArtifact:
    if max_gap_seconds < 0:
        raise TransformPrimitiveError("max_gap_seconds must be zero or greater.")
    return artifact.model_copy(
        update={
            "turns": merge_consecutive_turns(
                artifact.turns, max_gap_seconds=max_gap_seconds
            )
        }
    )


def draft_chapter_boundaries(
    artifact: TranscriptArtifact, *, window_minutes: float
) -> ChapterPlan:
    if window_minutes <= 0:
        raise TransformPrimitiveError("window_minutes must be greater than zero.")
    if not artifact.turns:
        raise TransformPrimitiveError("TranscriptArtifact has no turns for chaptering.")

    window_seconds = window_minutes * 60.0
    chapters: list[Chapter] = []
    chapter_start = artifact.turns[0].start
    chapter_turns: list[TranscriptTurn] = []

    for turn in artifact.turns:
        if chapter_turns and turn.start - chapter_start >= window_seconds:
            chapters.append(_draft_chapter(chapters, chapter_start, chapter_turns))
            chapter_start = turn.start
            chapter_turns = [turn]
        else:
            chapter_turns.append(turn)

    if chapter_turns:
        chapters.append(_draft_chapter(chapters, chapter_start, chapter_turns))

    chapters = [
        chapter.model_copy(update={"end": chapters[index + 1].start})
        if index + 1 < len(chapters)
        else chapter
        for index, chapter in enumerate(chapters)
    ]
    return ChapterPlan(chapters=chapters, boundary_source="deterministic")


def _draft_chapter(
    existing: list[Chapter], start: float, turns: list[TranscriptTurn]
) -> Chapter:
    end = max(turn.end for turn in turns)
    speakers = sorted({turn.speaker for turn in turns})
    speaker_summary = ", ".join(speakers[:3]) if speakers else "unknown speakers"
    return Chapter(
        start=start,
        end=end,
        title=f"Chapter {len(existing) + 1}",
        summary=f"Discussed turns from {speaker_summary}.",
    )
