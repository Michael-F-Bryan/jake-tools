from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from ..ai_usage import build_ai_totals
from .audio import build_concat_plan, concatenate_recordings, run_scribe
from .errors import TranscriptError
from .manifest import ManifestBuilder, write_json
from .merge import speaker_name
from .models import (
    ChapterPlan,
    ChapterSummary,
    CoordinatorResult,
    SpeakerMapping,
    SpeakerMessageCount,
    TranscriptArtifact,
)
from .notes import merge_generated_note, write_note
from .parse import parse_scribe_transcript
from .render import render_meeting_note_markdown, render_transcript_markdown
from .sources import load_source_note, source_from_obsidian_note
from .stages import (
    StructuredAgent,
    run_map_speakers_stage,
    run_minutes_stage,
    run_polish_stage,
    run_title_chapters_stage,
)
from .transform import merge_adjacent_turns, normalise_transcript_artifact
from .verify import verify_note

DEFAULT_STAGE_MAX_ATTEMPTS = 2
DEFAULT_MAX_GAP_SECONDS = 3.0


class RecipePrimitiveError(TranscriptError):
    pass


@contextmanager
def _recipe_workspace(workdir: Path | None) -> Generator[Path]:
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
            start_timestamp=_format_timestamp(chapter.start),
            end_timestamp=_format_timestamp(chapter.end),
        )
        for chapter in chapters.chapters
    ]


def _format_timestamp(seconds: float) -> str:
    from .merge import format_timestamp

    return format_timestamp(seconds)


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


async def run_obsidian_recording_recipe(
    agent: StructuredAgent,
    obsidian_note: Path,
    *,
    dry_run: bool,
    workdir: Path | None = None,
) -> CoordinatorResult:
    """Turn an Obsidian note's linked recordings into a verified, merged note.

    `workdir` controls whether intermediate artefacts (audio, transcripts,
    chapters, a manifest) persist: pass a directory to keep them around for
    inspection, matching the youtube/teams recipes' `--out-dir` convention.
    Leave it unset for an ephemeral run — nothing but the note itself is
    written, and no manifest is produced (there would be nothing left for it
    to point at).
    """
    manifest = ManifestBuilder(run_id="obsidian-recording")

    try:
        with _recipe_workspace(workdir) as workspace:
            manifest_path = workspace / "manifest.json" if workdir is not None else None
            try:
                with manifest.step("source.obsidian-note"):
                    source = source_from_obsidian_note(obsidian_note)
                    source_note = load_source_note(obsidian_note)
                    manifest.artefact("source", source.source_path or obsidian_note)
                    if not source_note.recordings:
                        raise RecipePrimitiveError(
                            f"No recordings found in {obsidian_note}"
                        )

                merged_audio_path = workspace / "merged.mp3"
                concat_inputs_path = workspace / "inputs.txt"
                transcript_json_path = workspace / "merged.json"
                raw_transcript_path = workspace / "transcript-raw.json"
                merged_transcript_path = workspace / "transcript-merged.json"
                transcript_markdown_path = workspace / "transcript.md"
                polished_path = workspace / "transcript.polished.json"
                speaker_mapping_path = workspace / "speaker-mapping.json"
                chapters_path = workspace / "chapters.json"
                minutes_path = workspace / "minutes.json"
                note_render_path = workspace / "note.generated.md"

                with manifest.step("audio.concat-recordings"):
                    concat_plan = build_concat_plan(
                        source_note.recordings, merged_audio_path
                    )
                    concatenate_recordings(concat_plan, concat_inputs_path)
                    manifest.artefact("audio.concat", concat_inputs_path)
                    manifest.artefact("audio.merged", merged_audio_path)

                with manifest.step("audio.scribe"):
                    scribe_report = run_scribe(merged_audio_path, transcript_json_path)
                    manifest.artefact("audio.transcript-json", transcript_json_path)
                    manifest.add_warnings(scribe_report.warnings)

                with manifest.step("parse.scribe"):
                    transcript = parse_scribe_transcript(transcript_json_path)
                    write_json(raw_transcript_path, transcript.model_dump(mode="json"))
                    manifest.artefact("transcript.raw", raw_transcript_path)

                with manifest.step("transform.normalise"):
                    transcript = normalise_transcript_artifact(transcript)

                with manifest.step("transform.merge-adjacent"):
                    transcript = merge_adjacent_turns(
                        transcript, max_gap_seconds=DEFAULT_MAX_GAP_SECONDS
                    )
                    write_json(
                        merged_transcript_path, transcript.model_dump(mode="json")
                    )
                    manifest.artefact("transcript.merged", merged_transcript_path)

                with manifest.step("stage.map-speakers"):
                    speaker_mapping, mapping_reply = await run_map_speakers_stage(
                        agent,
                        transcript,
                        attendees=source.attendees,
                        max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
                    )
                    manifest.record_reply("speaker_mapping", mapping_reply)
                    write_json(
                        speaker_mapping_path, speaker_mapping.model_dump(mode="json")
                    )
                    manifest.artefact("speaker-mapping", speaker_mapping_path)

                with manifest.step("stage.polish"):
                    polished, polish_reply = await run_polish_stage(
                        agent,
                        transcript,
                        max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
                    )
                    manifest.record_reply("transcript_polish", polish_reply)
                    write_json(polished_path, polished.model_dump(mode="json"))
                    manifest.artefact("transcript.polished", polished_path)

                with manifest.step("stage.title-chapters"):
                    chapter_plan, chapter_reply = await run_title_chapters_stage(
                        agent,
                        polished,
                        draft_plan=None,
                        max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
                    )
                    manifest.record_reply("chaptering", chapter_reply)
                    write_json(chapters_path, chapter_plan.model_dump(mode="json"))
                    manifest.artefact("chapters", chapters_path)

                with manifest.step("stage.minutes"):
                    minutes, minutes_reply = await run_minutes_stage(
                        agent,
                        polished,
                        chapters=chapter_plan,
                        max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
                    )
                    manifest.record_reply("meeting_minutes", minutes_reply)
                    write_json(minutes_path, minutes.model_dump(mode="json"))
                    manifest.artefact("minutes", minutes_path)

                with manifest.step("render.transcript"):
                    transcript_markdown = render_transcript_markdown(
                        polished, chapters=chapter_plan, speaker_mapping=speaker_mapping
                    )
                    transcript_markdown_path.write_text(
                        transcript_markdown, encoding="utf-8"
                    )
                    manifest.artefact("transcript.markdown", transcript_markdown_path)

                with manifest.step("render.meeting-note"):
                    generated_note, _sections = render_meeting_note_markdown(
                        source,
                        minutes,
                        transcript_markdown=transcript_markdown,
                        chapters=chapter_plan,
                    )
                    note_render_path.write_text(generated_note, encoding="utf-8")
                    manifest.artefact("note.generated", note_render_path)

                with manifest.step("note.merge"):
                    original_note = obsidian_note.read_text(encoding="utf-8")
                    merged_note = merge_generated_note(original_note, generated_note)

                with manifest.step("verify.note"):
                    note_report = verify_note(
                        merged_note,
                        expected_chapter_count=len(chapter_plan.chapters),
                        affected_path=obsidian_note,
                        original_body=original_note,
                    )
                    if note_report.failed_gate_ids:
                        raise RecipePrimitiveError(
                            "Obsidian recording note verification failed: "
                            + ", ".join(note_report.failed_gate_ids)
                        )

                if dry_run:
                    manifest.skip("note.write")
                    updated = False
                else:
                    with manifest.step("note.write"):
                        updated = write_note(obsidian_note, merged_note, dry_run=False)
                manifest.artefact("note.output", obsidian_note.resolve())

                return CoordinatorResult(
                    note_path=obsidian_note,
                    updated=updated,
                    chapter_summaries=_chapter_summaries(chapter_plan),
                    ai_stage_stats=manifest.ai_stage_stats,
                    ai_totals=build_ai_totals(manifest.ai_stage_stats),
                    speaker_message_counts=_speaker_message_counts(
                        polished, speaker_mapping
                    ),
                    warnings=manifest.warnings,
                )
            finally:
                if manifest_path is not None:
                    manifest.write(manifest_path)
    except TranscriptError as exc:
        raise RecipePrimitiveError(str(exc)) from exc
