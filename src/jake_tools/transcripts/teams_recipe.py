from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from .errors import TranscriptError
from .manifest import ManifestBuilder, write_json
from .models import ChapterPlan, MeetingMinutes, TeamsMeetingResult
from .parse import parse_teams_vtt
from .render import (
    MeetingNoteProfile,
    render_meeting_note_markdown,
    render_transcript_markdown,
)
from .teams_graph import TeamsMeetingSourceResult, source_from_teams_meeting
from .transform import (
    draft_chapter_boundaries,
    merge_adjacent_turns,
    normalise_transcript_artifact,
)
from .verify import verify_note

DEFAULT_MAX_GAP_SECONDS = 3.0


class RecipePrimitiveError(TranscriptError):
    pass


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "teams-meeting"


def _default_vault_note(vault_dir: Path, title: str, source_date: str | None) -> Path:
    date_prefix = source_date or "undated"
    return vault_dir / f"{date_prefix}_{_slug(title)}.md"


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
    profile: MeetingNoteProfile,
    out_dir: Path,
    token_file: Path,
    event_id: str | None = None,
    days_back: int = 14,
    query: str | None = None,
    organisation: str | None = None,
    project: str | None = None,
    vault_note: Path | None = None,
    write_vault: bool = False,
    dumc_vault_dir: Path | None = None,
    dry_run: bool = False,
    source_fetcher: Callable[..., TeamsMeetingSourceResult] = source_from_teams_meeting,
) -> TeamsMeetingResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ManifestBuilder(run_id="teams-meeting")
    manifest_path = out_dir / "manifest.json"

    try:
        with manifest.step("source.teams-meeting"):
            source_result = source_fetcher(
                account=account,
                out_dir=out_dir,
                token_file=token_file,
                event_id=event_id,
                days_back=days_back,
                query=query,
                organisation=organisation,
                project=project,
            )
            source = source_result.source
            manifest.artefact("source", out_dir / "source.json")
            manifest.artefact("raw-vtt", source_result.raw_vtt_path)

        with manifest.step("parse.teams-vtt"):
            parsed = parse_teams_vtt(source)
            parsed_path = out_dir / "transcript-raw.json"
            write_json(parsed_path, parsed.model_dump(mode="json"))
            manifest.artefact("transcript.raw", parsed_path)

        with manifest.step("transform.normalise"):
            normalised = normalise_transcript_artifact(parsed)

        with manifest.step("transform.merge-adjacent"):
            merged = merge_adjacent_turns(
                normalised, max_gap_seconds=DEFAULT_MAX_GAP_SECONDS
            )
            merged_path = out_dir / "transcript-merged.json"
            write_json(merged_path, merged.model_dump(mode="json"))
            manifest.artefact("transcript.merged", merged_path)

        with manifest.step("transform.chapter-boundaries"):
            chapters = draft_chapter_boundaries(merged, window_minutes=10.0)
            chapters_path = out_dir / "chapters.json"
            write_json(chapters_path, chapters.model_dump(mode="json"))
            manifest.artefact("chapters", chapters_path)

        with manifest.step("render.transcript"):
            transcript_markdown = render_transcript_markdown(merged, chapters=chapters)
            transcript_markdown_path = out_dir / "transcript.md"
            transcript_markdown_path.write_text(transcript_markdown, encoding="utf-8")
            manifest.artefact("transcript.markdown", transcript_markdown_path)

        title = source.title or "Teams meeting"
        minutes = _deterministic_minutes(title, chapters)
        minutes_path = out_dir / "minutes.json"
        write_json(minutes_path, minutes.model_dump(mode="json"))
        manifest.artefact("minutes", minutes_path)

        with manifest.step("render.meeting-note"):
            rendered_note, _sections = render_meeting_note_markdown(
                source,
                minutes,
                transcript_markdown=transcript_markdown,
                chapters=chapters,
                profile=profile,
            )
            rendered_note_path = out_dir / "rendered-note.md"
            rendered_note_path.write_text(rendered_note, encoding="utf-8")
            manifest.artefact("note.rendered", rendered_note_path)

        with manifest.step("verify.note"):
            report = verify_note(
                rendered_note,
                expected_chapter_count=len(chapters.chapters),
                affected_path=rendered_note_path,
                profile=profile,
            )
            report_path = out_dir / "verify-note.json"
            write_json(report_path, report.model_dump(mode="json"))
            manifest.artefact("verify.note", report_path)
            if report.failed_gate_ids:
                raise RecipePrimitiveError(
                    f"{profile} meeting note verification failed: "
                    + ", ".join(report.failed_gate_ids)
                )

        destination = vault_note
        if destination is None and write_vault and profile == "dumc":
            if dumc_vault_dir is None:
                raise RecipePrimitiveError(
                    "--write-vault requires a DUM-C vault directory."
                )
            destination = _default_vault_note(
                dumc_vault_dir, title, source.date.isoformat() if source.date else None
            )
        updated = False
        if destination is not None:
            destination = destination.expanduser()
            manifest.artefact("note.destination", destination)
            if dry_run:
                manifest.skip("note.write")
            else:
                with manifest.step("note.write"):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(rendered_note, encoding="utf-8")
                updated = True

        return TeamsMeetingResult(
            note=destination,
            updated=updated,
            out_dir=out_dir.resolve(),
            rendered_note=rendered_note_path.resolve(),
            raw_vtt=source_result.raw_vtt_path,
            manifest=manifest_path.resolve(),
        )
    except TranscriptError as exc:
        raise RecipePrimitiveError(str(exc)) from exc
    finally:
        manifest.write(manifest_path)
