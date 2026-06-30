from __future__ import annotations

import math
import re
from pathlib import Path

from .models import (
    ChapterPlan,
    PlannedChapter,
    RunManifest,
    RunStageStatus,
    SourceArtifact,
    TranscriptArtifact,
    TranscriptTurn,
)
from .transforms import merge_consecutive_turns, normalise_turns

_BOILERPLATE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*Loading the transcript-polisher skill\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*```{3,}.*$", re.IGNORECASE),
)


class TransformPrimitiveError(RuntimeError):
    pass


def _resolve_source_text_path(source: SourceArtifact) -> Path:
    path = source.raw_text_path or source.source_path
    if path is None:
        raise TransformPrimitiveError(
            "SourceArtifact is missing both raw_text_path and source_path."
        )
    if not path.exists():
        raise TransformPrimitiveError(f"Source text path does not exist: {path}")
    return path


def strip_source_boilerplate(
    source: SourceArtifact, *, source_output_path: Path
) -> SourceArtifact:
    source_text_path = _resolve_source_text_path(source)
    lines = source_text_path.read_text(encoding="utf-8").splitlines()
    removed_count = 0
    kept_lines: list[str] = []

    for line in lines:
        if any(pattern.match(line) for pattern in _BOILERPLATE_LINE_PATTERNS):
            removed_count += 1
            continue
        kept_lines.append(line)

    cleaned_text_path = source_output_path.resolve().with_suffix(".clean.txt")
    cleaned_text_path.write_text("\n".join(kept_lines).strip() + "\n", encoding="utf-8")

    warnings = list(source.warnings)
    if removed_count:
        warnings.append(f"Removed {removed_count} boilerplate line(s).")

    return source.model_copy(
        update={
            "raw_text_path": cleaned_text_path,
            "warnings": warnings,
        }
    )


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


def split_transcript_manifest(
    artifact: TranscriptArtifact, *, target_minutes: float
) -> RunManifest:
    if target_minutes <= 0:
        raise TransformPrimitiveError("--target-minutes must be greater than zero.")
    if not artifact.turns:
        raise TransformPrimitiveError("TranscriptArtifact has no turns to split.")

    chunk_seconds = target_minutes * 60.0
    first_start = artifact.turns[0].start
    last_end = max(turn.end for turn in artifact.turns)
    chunk_count = max(1, math.ceil(max(0.0, last_end - first_start) / chunk_seconds))

    stages: list[RunStageStatus] = []
    artefact_paths: dict[str, Path] = {}
    for index in range(chunk_count):
        start = first_start + (index * chunk_seconds)
        end = start + chunk_seconds
        chunk_turns = [
            turn for turn in artifact.turns if turn.start < end and turn.end >= start
        ]
        if not chunk_turns:
            continue
        chunk_key = f"chunk-{index + 1:03d}"
        chunk_path = Path(f"{chunk_key}.json")
        artefact_paths[chunk_key] = chunk_path
        stages.append(
            RunStageStatus(stage=chunk_key, status="pass", artefacts=[chunk_path])
        )

    if not stages:
        raise TransformPrimitiveError(
            "Unable to create any chunks from transcript turns."
        )

    return RunManifest(
        run_id="split-manifest",
        stages=stages,
        artefact_paths=artefact_paths,
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
