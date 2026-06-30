from __future__ import annotations

from pathlib import Path

from .merge import (
    render_chaptered_transcript,
    render_chapters_index,
    render_minutes,
    render_transcript,
)
from .models import (
    Chapter,
    ChapterPlan,
    MeetingMinutes,
    RenderedNote,
    SourceArtifact,
    SpeakerMapping,
    TranscriptArtifact,
)


class RenderPrimitiveError(RuntimeError):
    pass


def _plan_to_chapters(plan: ChapterPlan) -> list[Chapter]:
    return [
        Chapter(
            title=chapter.title,
            start=chapter.start,
            end=chapter.end,
            summary=chapter.summary,
        )
        for chapter in plan.chapters
    ]


def render_transcript_markdown(
    transcript: TranscriptArtifact,
    *,
    chapters: ChapterPlan | None = None,
    speaker_mapping: SpeakerMapping | None = None,
) -> str:
    if not transcript.turns:
        raise RenderPrimitiveError("TranscriptArtifact has no turns to render.")
    if chapters is not None and chapters.chapters:
        return render_chaptered_transcript(
            transcript.turns,
            _plan_to_chapters(chapters),
            speaker_mapping,
        )
    return render_transcript(transcript.turns, speaker_mapping)


def render_chapters_markdown(chapters: ChapterPlan) -> str:
    if not chapters.chapters:
        raise RenderPrimitiveError("ChapterPlan has no chapters to render.")
    return render_chapters_index(_plan_to_chapters(chapters))


def render_meeting_note_markdown(
    source: SourceArtifact,
    minutes: MeetingMinutes,
    *,
    transcript_markdown: str | None = None,
    chapters: ChapterPlan | None = None,
) -> tuple[str, list[str]]:
    title = source.title
    if not title and source.source_path is not None:
        title = source.source_path.stem
    title = title or "Meeting Note"

    sections_included = ["meeting-notes"]
    lines = [f"# {title}", "", "## Meeting Notes", "", render_minutes(minutes)]

    if chapters is not None and chapters.chapters:
        sections_included.append("chapters")
        lines.extend(
            [
                "",
                "## Chapters",
                "",
                render_chapters_markdown(chapters),
            ]
        )

    if transcript_markdown is not None:
        sections_included.append("transcript")
        lines.extend(["", "## Transcript", "", transcript_markdown.strip()])

    return "\n".join(lines).rstrip() + "\n", sections_included


def build_rendered_note_metadata(
    *,
    rendered_markdown_path: Path,
    sections_included: list[str],
    destination_path: Path | None = None,
    source: SourceArtifact | None = None,
    warnings: list[str] | None = None,
) -> RenderedNote:
    links: list[str] = []
    if source is not None:
        if source.source_path is not None:
            links.append(str(source.source_path))
        links.extend(str(path) for path in source.attachments)

    return RenderedNote(
        rendered_markdown_path=rendered_markdown_path.resolve(),
        destination_path=destination_path.resolve() if destination_path else None,
        sections_included=sections_included,
        attachments_or_source_links=links,
        warnings=warnings or [],
    )
