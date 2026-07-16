from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextlib import contextmanager
from difflib import SequenceMatcher
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
    MeetingMinutes,
    RunManifest,
    RunStageStatus,
    SourceArtifact,
    SpeakerMapping,
    SpeakerMessageCount,
    TranscriptArtifact,
    TranscriptTurn,
    VerificationReport,
)
from .note_primitives import merge_generated_note, write_note
from .obsidian import load_source_note
from .parse_primitives import (
    ParsePrimitiveError,
    parse_scribe_transcript,
    parse_teams_vtt,
    parse_youtube_json3,
)
from .render_primitives import (
    RenderPrimitiveError,
    render_meeting_note_markdown,
    render_source_note_markdown,
    render_transcript_markdown,
)
from .source_primitives import (
    SourcePrimitiveError,
    source_from_obsidian_note,
    source_from_youtube,
)
from .stage_primitives import (
    StagePrimitiveError,
    StructuredHermes,
    run_map_speakers_stage,
    run_minutes_stage,
    run_polish_stage,
    run_source_note_plan_stage,
    run_title_chapters_stage,
)
from .teams_graph import (
    TeamsGraphError,
    TeamsMeetingSourceResult,
    source_from_teams_meeting,
)
from .transform_primitives import (
    draft_chapter_boundaries,
    merge_adjacent_turns,
    normalise_transcript_artifact,
)
from .verify import VerificationError, verify_note
from .verify_primitives import (
    verify_chapters,
    verify_turns,
)
from .verify_primitives import (
    verify_note as verify_note_profile,
)

DEFAULT_STAGE_MAX_ATTEMPTS = 2
DEFAULT_MAX_GAP_SECONDS = 3.0
DUMC_VAULT_DIR = Path("~/Documents/Vault/2 Areas/DUM-C").expanduser()


class RecipePrimitiveError(RuntimeError):
    pass


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
    run_id: str,
    stages: list[RunStageStatus],
    artifact_paths: dict[str, Path],
) -> None:
    manifest = RunManifest(
        run_id=run_id,
        stages=stages,
        artefact_paths=artifact_paths,
    )
    _write_json(manifest_path, manifest.model_dump(mode="json"))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "teams-meeting"


def _default_dumc_vault_note(title: str, source_date: str | None) -> Path:
    date_prefix = source_date or "undated"
    return DUMC_VAULT_DIR / f"{date_prefix}_{_slug(title)}.md"


def _deterministic_minutes(title: str, chapters: ChapterPlan) -> MeetingMinutes:
    key_points = [
        chapter.summary for chapter in chapters.chapters if chapter.summary.strip()
    ]
    if not key_points:
        key_points = ["Transcript imported and rendered for review."]
    return MeetingMinutes(
        summary=f"Teams transcript imported for {title}.",
        key_points=key_points,
    )


def _youtube_source_context(source: SourceArtifact) -> str:
    title = source.title or "Untitled YouTube video"
    channel = str(source.metadata.get("channel") or source.organisation or "Unknown")
    return "\n".join(
        [
            f"Video title: {title}",
            f"Channel: {channel}",
            f"Source URL: {source.source_url or ''}",
            "Caption provenance: "
            + str(source.metadata.get("subtitle_kind") or "unknown"),
        ]
    )


def _require_passing_report(report: VerificationReport, *, label: str) -> None:
    if report.failed_gate_ids:
        raise RecipePrimitiveError(
            f"{label} verification failed: " + ", ".join(report.failed_gate_ids)
        )


def _ordered_word_coverage(source: str, candidate: str) -> float:
    source_words = re.findall(r"[\w']+", source.casefold())
    if not source_words:
        return 1.0
    candidate_words = re.findall(r"[\w']+", candidate.casefold())
    retained = sum(
        block.size
        for block in SequenceMatcher(
            a=source_words, b=candidate_words, autojunk=False
        ).get_matching_blocks()
    )
    return retained / len(source_words)


def _polish_youtube_chunks(
    hermes: StructuredHermes,
    transcript: TranscriptArtifact,
    *,
    context: str,
    target_minutes: float = 5.0,
) -> tuple[TranscriptArtifact, list[tuple[str, Reply]]]:
    if target_minutes <= 0:
        raise RecipePrimitiveError("Polish chunk duration must be greater than zero.")
    if not transcript.turns:
        raise RecipePrimitiveError("Transcript has no turns to polish.")

    chunk_seconds = target_minutes * 60.0
    ranges: list[tuple[int, int]] = []
    chunk_start = 0
    window_start = transcript.turns[0].start
    for turn_index, turn in enumerate(transcript.turns):
        if turn_index > chunk_start and turn.start - window_start >= chunk_seconds:
            ranges.append((chunk_start, turn_index))
            chunk_start = turn_index
            window_start = turn.start
    ranges.append((chunk_start, len(transcript.turns)))

    polished_turns: list[TranscriptTurn] = []
    replies: list[tuple[str, Reply]] = []
    for chunk_number, (start_index, end_index) in enumerate(ranges, start=1):
        source_turns = transcript.turns[start_index:end_index]
        chunk = TranscriptArtifact(
            turns=source_turns,
            source_refs=[
                source_ref.model_copy(
                    update={"turn_index": source_ref.turn_index - start_index}
                )
                for source_ref in transcript.source_refs
                if start_index <= source_ref.turn_index < end_index
            ],
            speakers=transcript.speakers,
            warnings=transcript.warnings,
        )
        polished, reply = run_polish_stage(
            hermes,
            chunk,
            context=context,
            max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
        )
        for source_turn, polished_turn in zip(
            source_turns, polished.turns, strict=True
        ):
            same_boundary = (
                source_turn.start == polished_turn.start
                and source_turn.end == polished_turn.end
                and source_turn.speaker == polished_turn.speaker
            )
            faithful = (
                len(re.findall(r"[\w']+", source_turn.text)) < 5
                or _ordered_word_coverage(source_turn.text, polished_turn.text) >= 0.5
            )
            if not same_boundary or not faithful:
                raise RecipePrimitiveError(
                    f"Polish chunk {chunk_number} failed source fidelity."
                )
        polished_turns.extend(polished.turns)
        replies.append((f"transcript_polish.chunk-{chunk_number:03d}", reply))

    return transcript.model_copy(update={"turns": polished_turns}), replies


def run_youtube_source_notes_recipe(
    hermes: StructuredHermes,
    url: str,
    *,
    out_dir: Path,
    language: str = "en",
    vault_note: Path | None = None,
    dry_run: bool = False,
    source_fetcher: Callable[..., SourceArtifact] = source_from_youtube,
) -> dict[str, object]:
    artefacts: dict[str, Path] = {}
    replies: list[tuple[str, Reply | None]] = []
    try:
        out_dir = out_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        source = source_fetcher(url, out_dir=out_dir, language=language)
        if source.source_path is not None:
            artefacts["metadata"] = source.source_path
        if source.raw_text_path is not None:
            artefacts["captions"] = source.raw_text_path

        raw_transcript = parse_youtube_json3(source)
        raw_path = out_dir / "transcript-raw.json"
        _write_json(raw_path, raw_transcript.model_dump(mode="json"))
        artefacts["transcript.raw"] = raw_path

        normalised = normalise_transcript_artifact(raw_transcript)
        context = _youtube_source_context(source)
        polished, polish_replies = _polish_youtube_chunks(
            hermes,
            normalised,
            context=context,
        )
        polished_report = verify_turns(
            normalised,
            polished,
            affected_paths=[raw_path],
        )
        _require_passing_report(polished_report, label="Polished transcript")
        polished_path = out_dir / "transcript-polished.json"
        _write_json(polished_path, polished.model_dump(mode="json"))
        replies.extend(polish_replies)
        artefacts["transcript.polished"] = polished_path

        draft_chapters = draft_chapter_boundaries(polished, window_minutes=5.0)
        source_note_plan, plan_reply = run_source_note_plan_stage(
            hermes,
            source,
            polished,
            draft_plan=draft_chapters,
            max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
        )
        chapters = source_note_plan.chapters
        overview = source_note_plan.overview
        chapter_report = verify_chapters(
            chapters,
            transcript=polished,
            affected_paths=[polished_path],
        )
        _require_passing_report(chapter_report, label="Chapter plan")
        replies.append(("source-note-plan", plan_reply))

        note, _sections = render_source_note_markdown(
            source, overview, polished, chapters
        )
        note_path = out_dir / "source-note.md"
        note_path.write_text(note, encoding="utf-8")
        artefacts["note"] = note_path

        note_report = verify_note_profile(
            note,
            expected_chapter_count=len(chapters.chapters),
            affected_path=note_path,
            profile="source",
        )
        _require_passing_report(note_report, label="Source note")

        destination = vault_note.expanduser().resolve() if vault_note else None
        updated = False
        stages = [RunStageStatus(stage="source-note", status="pass")]
        if destination is not None:
            artefacts["destination"] = destination
            if dry_run:
                stages.append(RunStageStatus(stage="vault.write", status="skipped"))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(note, encoding="utf-8")
                stages.append(RunStageStatus(stage="vault.write", status="pass"))
                updated = True

        ai_stage_stats = [
            stats
            for stage, reply in replies
            if (stats := build_ai_stage_stats(stage, reply)) is not None
        ]
        manifest = RunManifest(
            run_id="youtube-source-notes",
            stages=stages,
            artefact_paths=artefacts,
            ai_totals=build_ai_totals(ai_stage_stats),
        )
        manifest_path = out_dir / "manifest.json"
        _write_json(manifest_path, manifest.model_dump(mode="json"))
        return {
            "title": source.title,
            "note": str(destination) if destination is not None else None,
            "updated": updated,
            "out_dir": str(out_dir),
            "rendered_note": str(note_path),
            "manifest": str(manifest_path),
            "verification_status": note_report.status,
            "subtitle_track": source.metadata.get("subtitle_track"),
            "subtitle_kind": source.metadata.get("subtitle_kind"),
        }
    except (
        ParsePrimitiveError,
        RenderPrimitiveError,
        SourcePrimitiveError,
        StagePrimitiveError,
        ValueError,
    ) as exc:
        raise RecipePrimitiveError(str(exc)) from exc


def run_teams_meeting_recipe(
    *,
    account: str,
    profile: str,
    out_dir: Path,
    token_file: Path | None = None,
    event_id: str | None = None,
    days_back: int = 14,
    query: str | None = None,
    vault_note: Path | None = None,
    write_vault: bool = False,
    dry_run: bool = False,
    source_fetcher: Callable[..., TeamsMeetingSourceResult] = source_from_teams_meeting,
) -> dict[str, object]:
    stage_names: list[str] = []
    artifact_paths: dict[str, Path] = {}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        source_result = source_fetcher(
            account=account,
            out_dir=out_dir,
            token_file=token_file,
            event_id=event_id,
            days_back=days_back,
            query=query,
        )
        source = source_result.source
        source_path = out_dir / "source.json"
        stage_names.append("source.teams-meeting")
        artifact_paths["source"] = source_path
        artifact_paths["raw-vtt"] = source_result.raw_vtt_path

        parsed = parse_teams_vtt(source)
        parsed_path = out_dir / "transcript-raw.json"
        _write_json(parsed_path, parsed.model_dump(mode="json"))
        stage_names.append("parse.teams-vtt")
        artifact_paths["transcript.raw"] = parsed_path

        normalised = normalise_transcript_artifact(parsed)
        merged = merge_adjacent_turns(
            normalised, max_gap_seconds=DEFAULT_MAX_GAP_SECONDS
        )
        merged_path = out_dir / "transcript-merged.json"
        _write_json(merged_path, merged.model_dump(mode="json"))
        stage_names.extend(["transform.normalise", "transform.merge-adjacent"])
        artifact_paths["transcript.merged"] = merged_path

        chapters = draft_chapter_boundaries(merged, window_minutes=10.0)
        chapters_path = out_dir / "chapters.json"
        _write_json(chapters_path, chapters.model_dump(mode="json"))
        stage_names.append("transform.chapter-boundaries")
        artifact_paths["chapters"] = chapters_path

        transcript_markdown = render_transcript_markdown(merged, chapters=chapters)
        transcript_markdown_path = out_dir / "transcript.md"
        transcript_markdown_path.write_text(transcript_markdown, encoding="utf-8")
        stage_names.append("render.transcript")
        artifact_paths["transcript.markdown"] = transcript_markdown_path

        title = source.title or "Teams meeting"
        minutes = _deterministic_minutes(title, chapters)
        minutes_path = out_dir / "minutes.json"
        _write_json(minutes_path, minutes.model_dump(mode="json"))
        artifact_paths["minutes"] = minutes_path

        selected_profile = "dumc" if profile == "dumc" else "default"
        rendered_note, _sections = render_meeting_note_markdown(
            source,
            minutes,
            transcript_markdown=transcript_markdown,
            chapters=chapters,
            profile=selected_profile,
        )
        rendered_note_path = out_dir / "rendered-note.md"
        rendered_note_path.write_text(rendered_note, encoding="utf-8")
        stage_names.append("render.meeting-note")
        artifact_paths["note.rendered"] = rendered_note_path

        report = verify_note_profile(
            rendered_note,
            expected_chapter_count=len(chapters.chapters),
            affected_path=rendered_note_path,
            profile=selected_profile,
        )
        report_path = out_dir / "verify-note.json"
        _write_json(report_path, report.model_dump(mode="json"))
        stage_names.append("verify.note")
        artifact_paths["verify.note"] = report_path
        if report.failed_gate_ids:
            raise RecipePrimitiveError(
                "DUM-C note verification failed: " + ", ".join(report.failed_gate_ids)
            )

        destination = vault_note
        if destination is None and write_vault and selected_profile == "dumc":
            destination = _default_dumc_vault_note(
                title, source.date.isoformat() if source.date else None
            )
        updated = False
        if destination is not None:
            destination = destination.expanduser()
            artifact_paths["note.destination"] = destination
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(rendered_note, encoding="utf-8")
                updated = True

        stages = [RunStageStatus(stage=name, status="pass") for name in stage_names]
        if destination is not None:
            stages.append(
                RunStageStatus(
                    stage="note.write", status="skipped" if dry_run else "pass"
                )
            )
        manifest_path = out_dir / "manifest.json"
        _write_manifest(
            manifest_path=manifest_path,
            run_id="teams-meeting",
            stages=stages,
            artifact_paths=artifact_paths,
        )
        return {
            "note": str(destination) if destination else None,
            "updated": updated,
            "out_dir": str(out_dir.resolve()),
            "rendered_note": str(rendered_note_path.resolve()),
            "raw_vtt": str(source_result.raw_vtt_path),
            "manifest": str(manifest_path.resolve()),
        }
    except (TeamsGraphError, ValueError) as exc:
        raise RecipePrimitiveError(str(exc)) from exc


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

            polished, polish_reply = run_polish_stage(
                hermes,
                transcript,
                max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
            )
            stage_replies.append(("transcript_polish", polish_reply))
            stage_names.append("stage.polish")
            _write_json(polished_path, polished.model_dump(mode="json"))
            artifact_paths["transcript.polished"] = polished_path

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
            artifact_paths["note.output"] = obsidian_note.resolve()

            if manifest_path is not None:
                stages = [
                    RunStageStatus(stage=name, status="pass") for name in stage_names
                ]
                stages.append(
                    RunStageStatus(
                        stage="note.write", status="skipped" if dry_run else "pass"
                    )
                )
                _write_manifest(
                    manifest_path=manifest_path,
                    run_id="obsidian-recording",
                    stages=stages,
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
