from __future__ import annotations

from .models import (
    ChapterPlan,
    PlannedChapter,
    TranscriptArtifact,
    TranscriptTurn,
)
from .transforms import merge_consecutive_turns, normalise_turns


class TransformPrimitiveError(RuntimeError):
    pass


def normalise_transcript_artifact(artifact: TranscriptArtifact) -> TranscriptArtifact:
    return artifact.model_copy(update={"turns": normalise_turns(artifact.turns)})


def merge_adjacent_turns(
    artifact: TranscriptArtifact, *, max_gap_seconds: float
) -> TranscriptArtifact:
    if max_gap_seconds < 0:
        raise TransformPrimitiveError("--max-gap must be zero or greater.")
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
        raise TransformPrimitiveError("--window-minutes must be greater than zero.")
    if not artifact.turns:
        raise TransformPrimitiveError("TranscriptArtifact has no turns for chaptering.")

    window_seconds = window_minutes * 60.0
    chapters: list[PlannedChapter] = []
    chapter_start = artifact.turns[0].start
    chapter_turns: list = []

    for turn in artifact.turns:
        if chapter_turns and turn.start - chapter_start >= window_seconds:
            chapters.append(_planned_chapter(chapters, chapter_start, chapter_turns))
            chapter_start = turn.start
            chapter_turns = [turn]
        else:
            chapter_turns.append(turn)

    if chapter_turns:
        chapters.append(_planned_chapter(chapters, chapter_start, chapter_turns))

    chapters = [
        chapter.model_copy(update={"end": chapters[index + 1].start})
        if index + 1 < len(chapters)
        else chapter
        for index, chapter in enumerate(chapters)
    ]
    return ChapterPlan(chapters=chapters, boundary_source="deterministic")


def _planned_chapter(
    existing: list[PlannedChapter], start: float, turns: list[TranscriptTurn]
) -> PlannedChapter:
    end = max(turn.end for turn in turns)
    speakers = sorted({turn.speaker for turn in turns})
    speaker_summary = ", ".join(speakers[:3]) if speakers else "unknown speakers"
    return PlannedChapter(
        start=start,
        end=end,
        title=f"Chapter {len(existing) + 1}",
        summary=f"Discussed turns from {speaker_summary}.",
    )
