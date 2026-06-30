from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..ai_usage import build_ai_stage_stats, build_ai_totals
from ..hermes import Hermes, Reply
from .audio import (
    AudioPipelineError,
    build_concat_plan,
    concatenate_recordings,
    run_scribe,
)
from .merge import format_timestamp, speaker_name
from .models import (
    Chapter,
    ChapterPlan,
    ChapterSummary,
    CoordinatorResult,
    RunManifest,
    RunStageStatus,
    SpeakerMapping,
    SpeakerMessageCount,
    TranscriptArtifact,
)
from .note_primitives import merge_generated_note, write_note
from .obsidian import load_source_note
from .parse_primitives import parse_scribe_transcript
from .render_primitives import render_meeting_note_markdown, render_transcript_markdown
from .source_primitives import source_from_obsidian_note
from .stage_primitives import (
    StagePrimitiveError,
    run_map_speakers_stage,
    run_minutes_stage,
    run_polish_stage,
    run_title_chapters_stage,
)
from .transform_primitives import merge_adjacent_turns, normalise_transcript_artifact
from .verify import VerificationError, verify_note

DEFAULT_STAGE_MAX_ATTEMPTS = 2
DEFAULT_MAX_GAP_SECONDS = 3.0


class RecipePrimitiveError(RuntimeError):
    pass


def obsidian_recording_recipe_plan() -> dict[str, Any]:
    return {
        "recipe": "obsidian-recording",
        "description": "Compose Obsidian recording workflow over transcript primitives.",
        "supports": {
            "show_plan": True,
            "json_plan": True,
            "workdir": True,
            "manifest": True,
            "dry_run": True,
        },
        "steps": [
            {"index": 1, "primitive": "source.obsidian-note"},
            {"index": 2, "primitive": "audio.concat-recordings"},
            {"index": 3, "primitive": "audio.scribe"},
            {"index": 4, "primitive": "parse.scribe"},
            {"index": 5, "primitive": "transform.normalise"},
            {"index": 6, "primitive": "transform.merge-adjacent"},
            {"index": 7, "primitive": "stage.map-speakers"},
            {"index": 8, "primitive": "stage.polish"},
            {"index": 9, "primitive": "stage.title-chapters"},
            {"index": 10, "primitive": "stage.minutes"},
            {"index": 11, "primitive": "render.transcript"},
            {"index": 12, "primitive": "render.meeting-note"},
            {"index": 13, "primitive": "note.merge"},
            {"index": 14, "primitive": "verify.note"},
            {"index": 15, "primitive": "note.write"},
        ],
    }


def render_obsidian_recording_recipe_plan() -> str:
    plan = obsidian_recording_recipe_plan()
    lines = [f"recipe: {plan['recipe']}", "primitive sequence:"]
    for step in plan["steps"]:
        lines.append(f"{step['index']:02d}. {step['primitive']}")
    return "\n".join(lines)


@contextmanager
def _recipe_workspace(workdir: Path | None):
    if workdir is None:
        with TemporaryDirectory() as tmp:
            yield Path(tmp)
        return

    resolved = workdir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    yield resolved


def _chapter_summaries(chapters: ChapterPlan) -> list[ChapterSummary]:
    return [
        ChapterSummary(
            title=chapter.title,
            start_timestamp=format_timestamp(chapter.start),
            end_timestamp=format_timestamp(chapter.end),
        )
        for chapter in chapters.chapters
    ]


def _speaker_message_counts(
    transcript: TranscriptArtifact, mapping: SpeakerMapping | None
) -> list[SpeakerMessageCount]:
    counts: dict[str, int] = {}
    order: list[str] = []
    for turn in transcript.turns:
        resolved_name = speaker_name(turn.speaker, mapping)
        counts[resolved_name] = counts.get(resolved_name, 0) + 1
        if resolved_name not in order:
            order.append(resolved_name)
    return [SpeakerMessageCount(speaker=name, messages=counts[name]) for name in order]


def _to_chapters(chapter_plan: ChapterPlan) -> list[Chapter]:
    return [
        Chapter(
            title=chapter.title,
            start=chapter.start,
            end=chapter.end,
            summary=chapter.summary,
        )
        for chapter in chapter_plan.chapters
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_manifest(
    *,
    manifest_path: Path,
    stage_names: list[str],
    artifact_paths: dict[str, Path],
) -> None:
    manifest = RunManifest(
        run_id="obsidian-recording-recipe",
        stages=[RunStageStatus(stage=stage, status="pass") for stage in stage_names],
        artefact_paths=artifact_paths,
    )
    _write_json(manifest_path, manifest.model_dump(mode="json"))


def run_obsidian_recording_recipe(
    hermes: Hermes,
    obsidian_note: Path,
    *,
    dry_run: bool,
    workdir: Path | None = None,
    manifest_path: Path | None = None,
) -> CoordinatorResult:
    stage_replies: list[tuple[str, Reply | None]] = []
    stage_names: list[str] = []
    artifact_paths: dict[str, Path] = {}

    try:
        with _recipe_workspace(workdir) as workspace:
            source = source_from_obsidian_note(obsidian_note)
            source_note = load_source_note(obsidian_note)
            stage_names.append("source.obsidian-note")
            artifact_paths["source"] = source.source_path or obsidian_note
            if not source_note.recordings:
                raise RecipePrimitiveError(f"No recordings found in {obsidian_note}")

            merged_audio_path = workspace / "merged.mp3"
            concat_inputs_path = workspace / "inputs.txt"
            transcript_json_path = workspace / "merged.json"
            transcript_path = workspace / "transcript.artifact.json"
            transcript_markdown_path = workspace / "transcript.md"
            polished_path = workspace / "transcript.polished.json"
            ledger_path = workspace / "transcript.polish.ledger.json"
            speaker_mapping_path = workspace / "speaker-mapping.json"
            chapters_path = workspace / "chapters.json"
            minutes_path = workspace / "minutes.json"
            note_render_path = workspace / "note.generated.md"

            concat_plan = build_concat_plan(source_note.recordings, merged_audio_path)
            concatenate_recordings(concat_plan, concat_inputs_path)
            stage_names.append("audio.concat-recordings")
            artifact_paths["audio.concat"] = concat_inputs_path
            artifact_paths["audio.merged"] = merged_audio_path

            run_scribe(merged_audio_path, transcript_json_path)
            stage_names.append("audio.scribe")
            artifact_paths["audio.transcript-json"] = transcript_json_path

            transcript = parse_scribe_transcript(transcript_json_path)
            stage_names.append("parse.scribe")
            _write_json(transcript_path, transcript.model_dump(mode="json"))
            artifact_paths["transcript.parsed"] = transcript_path

            transcript = normalise_transcript_artifact(transcript)
            stage_names.append("transform.normalise")
            transcript = merge_adjacent_turns(
                transcript, max_gap_seconds=DEFAULT_MAX_GAP_SECONDS
            )
            stage_names.append("transform.merge-adjacent")
            _write_json(transcript_path, transcript.model_dump(mode="json"))
            artifact_paths["transcript.normalised"] = transcript_path

            speaker_mapping, mapping_reply = run_map_speakers_stage(
                hermes,
                transcript,
                attendees=source.attendees,
                max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
            )
            stage_replies.append(("speaker_mapping", mapping_reply))
            stage_names.append("stage.map-speakers")
            _write_json(speaker_mapping_path, speaker_mapping.model_dump(mode="json"))
            artifact_paths["speaker-mapping"] = speaker_mapping_path

            polished, ledger, polish_reply = run_polish_stage(
                hermes,
                transcript,
                max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
            )
            stage_replies.append(("transcript_polish", polish_reply))
            stage_names.append("stage.polish")
            _write_json(polished_path, polished.model_dump(mode="json"))
            _write_json(ledger_path, ledger.model_dump(mode="json"))
            artifact_paths["transcript.polished"] = polished_path
            artifact_paths["transcript.polish-ledger"] = ledger_path

            chapter_plan, chapter_reply = run_title_chapters_stage(
                hermes,
                polished,
                draft_plan=None,
                max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
            )
            stage_replies.append(("chaptering", chapter_reply))
            stage_names.append("stage.title-chapters")
            _write_json(chapters_path, chapter_plan.model_dump(mode="json"))
            artifact_paths["chapters"] = chapters_path

            minutes, minutes_reply = run_minutes_stage(
                hermes,
                polished,
                chapters=chapter_plan,
                max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
            )
            stage_replies.append(("meeting_minutes", minutes_reply))
            stage_names.append("stage.minutes")
            _write_json(minutes_path, minutes.model_dump(mode="json"))
            artifact_paths["minutes"] = minutes_path

            transcript_markdown = render_transcript_markdown(
                polished, chapters=chapter_plan, speaker_mapping=speaker_mapping
            )
            stage_names.append("render.transcript")
            transcript_markdown_path.write_text(transcript_markdown, encoding="utf-8")
            artifact_paths["transcript.markdown"] = transcript_markdown_path

            generated_note, _sections = render_meeting_note_markdown(
                source,
                minutes,
                transcript_markdown=transcript_markdown,
                chapters=chapter_plan,
            )
            stage_names.append("render.meeting-note")
            note_render_path.write_text(generated_note, encoding="utf-8")
            artifact_paths["note.generated"] = note_render_path

            original_note = obsidian_note.read_text(encoding="utf-8")
            merged_note = merge_generated_note(original_note, generated_note)
            stage_names.append("note.merge")

            verify_note(
                merged_note,
                original_body=original_note,
                chapters=_to_chapters(chapter_plan),
            )
            stage_names.append("verify.note")

            updated = write_note(obsidian_note, merged_note, dry_run=dry_run)
            stage_names.append("note.write")
            artifact_paths["note.output"] = obsidian_note.resolve()

            if manifest_path is not None:
                _write_manifest(
                    manifest_path=manifest_path,
                    stage_names=stage_names,
                    artifact_paths=artifact_paths,
                )

            ai_stage_stats = [
                stage_stats
                for stage, reply in stage_replies
                if (stage_stats := build_ai_stage_stats(stage, reply)) is not None
            ]
            return CoordinatorResult(
                note_path=obsidian_note,
                updated=updated,
                chapter_summaries=_chapter_summaries(chapter_plan),
                ai_stage_stats=ai_stage_stats,
                ai_totals=build_ai_totals(ai_stage_stats),
                speaker_message_counts=_speaker_message_counts(
                    polished, speaker_mapping
                ),
            )
    except (
        AudioPipelineError,
        StagePrimitiveError,
        VerificationError,
        ValueError,
    ) as exc:
        raise RecipePrimitiveError(str(exc)) from exc
