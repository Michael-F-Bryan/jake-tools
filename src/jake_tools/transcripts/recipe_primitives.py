from __future__ import annotations

import json
import re
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
    MeetingMinutes,
    RunManifest,
    RunStageStatus,
    SpeakerMapping,
    SpeakerMessageCount,
    TranscriptArtifact,
)
from .note_primitives import merge_generated_note, write_note
from .obsidian import load_source_note
from .parse_primitives import parse_scribe_transcript, parse_teams_vtt
from .render_primitives import render_meeting_note_markdown, render_transcript_markdown
from .source_primitives import source_from_obsidian_note
from .stage_primitives import (
    StagePrimitiveError,
    run_map_speakers_stage,
    run_minutes_stage,
    run_polish_stage,
    run_title_chapters_stage,
)
from .teams_graph import TeamsGraphError, source_from_teams_meeting
from .transform_primitives import (
    draft_chapter_boundaries,
    merge_adjacent_turns,
    normalise_transcript_artifact,
)
from .verify import VerificationError, verify_note
from .verify_primitives import verify_note as verify_note_profile

DEFAULT_STAGE_MAX_ATTEMPTS = 2
DEFAULT_MAX_GAP_SECONDS = 3.0
DUMC_VAULT_DIR = Path("~/Documents/Vault/2 Areas/DUM-C").expanduser()


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


def teams_meeting_recipe_plan() -> dict[str, Any]:
    return {
        "recipe": "teams-meeting",
        "description": "Import Teams transcript VTT into canonical transcript JSON and render a profiled note.",
        "supports": {
            "account": True,
            "event_id": True,
            "days_back": True,
            "profile": ["default", "dumc"],
            "out_dir": True,
            "vault_note": True,
            "dry_run": True,
        },
        "steps": [
            {"index": 1, "primitive": "source.teams-meeting"},
            {"index": 2, "primitive": "parse.teams-vtt"},
            {"index": 3, "primitive": "transform.normalise"},
            {"index": 4, "primitive": "transform.merge-adjacent"},
            {"index": 5, "primitive": "transform.chapter-boundaries"},
            {"index": 6, "primitive": "render.transcript"},
            {"index": 7, "primitive": "render.meeting-note"},
            {"index": 8, "primitive": "verify.note"},
            {
                "index": 9,
                "primitive": "note.write",
                "when": "--vault-note or --write-vault",
            },
        ],
    }


def render_teams_meeting_recipe_plan() -> str:
    plan = teams_meeting_recipe_plan()
    lines = [f"recipe: {plan['recipe']}", "primitive sequence:"]
    for step in plan["steps"]:
        suffix = f" ({step['when']})" if "when" in step else ""
        lines.append(f"{step['index']:02d}. {step['primitive']}{suffix}")
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
) -> dict[str, object]:
    stage_names: list[str] = []
    artifact_paths: dict[str, Path] = {}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        source_result = source_from_teams_meeting(
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
            stage_names.append("note.write")
            destination = destination.expanduser()
            artifact_paths["note.destination"] = destination
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(rendered_note, encoding="utf-8")
                updated = True

        manifest_path = out_dir / "manifest.json"
        _write_manifest(
            manifest_path=manifest_path,
            stage_names=stage_names,
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
