from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..claude import Reply
from .errors import TranscriptError
from .manifest import ManifestBuilder, write_json
from .models import (
    SourceArtifact,
    VerificationReport,
    YoutubeCaptureMetadata,
    YoutubeSourceNoteResult,
)
from .parse import parse_youtube_json3
from .render import render_source_note_markdown
from .sources import source_from_youtube
from .stages import (
    StructuredAgent,
    run_source_note_plan_stage,
    run_youtube_polish_stage,
)
from .transform import draft_chapter_boundaries, normalise_transcript_artifact
from .verify import verify_chapters, verify_note, verify_turns

DEFAULT_STAGE_MAX_ATTEMPTS = 2


class RecipePrimitiveError(TranscriptError):
    pass


def _require_passing_report(report: VerificationReport, *, label: str) -> None:
    if report.failed_gate_ids:
        raise RecipePrimitiveError(
            f"{label} verification failed: " + ", ".join(report.failed_gate_ids)
        )


def _youtube_source_context(
    source: SourceArtifact, capture: YoutubeCaptureMetadata
) -> str:
    title = source.title or "Untitled YouTube video"
    return "\n".join(
        [
            f"Video title: {title}",
            f"Channel: {capture.channel}",
            f"Source URL: {source.source_url or ''}",
            f"Caption provenance: {capture.subtitle_kind}",
        ]
    )


async def run_youtube_source_notes_recipe(
    agent: StructuredAgent,
    url: str,
    *,
    out_dir: Path,
    language: str = "en",
    vault_note: Path | None = None,
    dry_run: bool = False,
    source_fetcher: Callable[..., SourceArtifact] = source_from_youtube,
) -> YoutubeSourceNoteResult:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ManifestBuilder(run_id="youtube-source-notes")
    manifest_path = out_dir / "manifest.json"
    replies: list[tuple[str, Reply | None]] = []

    try:
        with manifest.step("source.youtube"):
            source = source_fetcher(url, out_dir=out_dir, language=language)
            capture = YoutubeCaptureMetadata.model_validate(source.metadata)
            if source.source_path is not None:
                manifest.artefact("metadata", source.source_path)
            if source.raw_text_path is not None:
                manifest.artefact("captions", source.raw_text_path)

        with manifest.step("parse.youtube-captions"):
            raw_transcript = parse_youtube_json3(source)
            raw_path = out_dir / "transcript-raw.json"
            write_json(raw_path, raw_transcript.model_dump(mode="json"))
            manifest.artefact("transcript.raw", raw_path)

        with manifest.step("transform.normalise"):
            normalised = normalise_transcript_artifact(raw_transcript)

        with manifest.step("stage.polish"):
            context = _youtube_source_context(source, capture)
            polished, polish_replies = await run_youtube_polish_stage(
                agent,
                normalised,
                context=context,
                max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
            )
            polished_report = verify_turns(
                normalised, polished, affected_paths=[raw_path]
            )
            _require_passing_report(polished_report, label="Polished transcript")
            polished_path = out_dir / "transcript-polished.json"
            write_json(polished_path, polished.model_dump(mode="json"))
            replies.extend(polish_replies)
            manifest.artefact("transcript.polished", polished_path)

        with manifest.step("transform.chapter-boundaries"):
            draft_chapters = draft_chapter_boundaries(polished, window_minutes=5.0)

        with manifest.step("stage.source-note-plan"):
            source_note_plan, plan_reply = await run_source_note_plan_stage(
                agent,
                source,
                polished,
                draft_plan=draft_chapters,
                max_attempts=DEFAULT_STAGE_MAX_ATTEMPTS,
            )
            chapters = source_note_plan.chapters
            overview = source_note_plan.overview
            chapter_report = verify_chapters(
                chapters, transcript=polished, affected_paths=[polished_path]
            )
            _require_passing_report(chapter_report, label="Chapter plan")
            replies.append(("source-note-plan", plan_reply))

        with manifest.step("render.source-note"):
            note, _sections = render_source_note_markdown(
                source, overview, polished, chapters
            )
            note_path = out_dir / "source-note.md"
            note_path.write_text(note, encoding="utf-8")
            manifest.artefact("note", note_path)

        with manifest.step("verify.note"):
            note_report = verify_note(
                note,
                expected_chapter_count=len(chapters.chapters),
                affected_path=note_path,
                profile="source",
            )
            _require_passing_report(note_report, label="Source note")

        destination = vault_note.expanduser().resolve() if vault_note else None
        updated = False
        if destination is not None:
            manifest.artefact("destination", destination)
            if dry_run:
                manifest.skip("note.write")
            else:
                with manifest.step("note.write"):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(note, encoding="utf-8")
                updated = True

        for stage, reply in replies:
            manifest.record_reply(stage, reply)

        return YoutubeSourceNoteResult(
            title=source.title,
            note=destination,
            updated=updated,
            out_dir=out_dir,
            rendered_note=note_path,
            manifest=manifest_path,
            verification_status=note_report.status,
            subtitle_track=capture.subtitle_track,
            subtitle_kind=capture.subtitle_kind,
        )
    except TranscriptError as exc:
        raise RecipePrimitiveError(str(exc)) from exc
    finally:
        manifest.write(manifest_path)
