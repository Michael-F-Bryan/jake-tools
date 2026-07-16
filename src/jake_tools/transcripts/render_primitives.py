from __future__ import annotations

import json
import re
from typing import Literal

from .merge import (
    format_timestamp,
    render_chaptered_transcript,
    render_chapters_index,
    render_minutes,
    render_transcript,
)
from .models import (
    Chapter,
    ChapterPlan,
    MeetingMinutes,
    SourceArtifact,
    SourceOverview,
    SpeakerMapping,
    TranscriptArtifact,
    TranscriptTurn,
)


class RenderPrimitiveError(RuntimeError):
    pass


MeetingNoteProfile = Literal["default", "dumc"]


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
    profile: MeetingNoteProfile = "default",
) -> tuple[str, list[str]]:
    if profile == "dumc":
        return _render_dumc_meeting_note_markdown(
            source,
            minutes,
            transcript_markdown=transcript_markdown,
            chapters=chapters,
        )
    if profile != "default":
        raise RenderPrimitiveError(f"unknown meeting-note profile: {profile}")

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


def _render_dumc_meeting_note_markdown(
    source: SourceArtifact,
    minutes: MeetingMinutes,
    *,
    transcript_markdown: str | None,
    chapters: ChapterPlan | None,
) -> tuple[str, list[str]]:
    title = source.title
    if not title and source.source_path is not None:
        title = source.source_path.stem
    title = title or "DUM-C meeting"

    frontmatter = ["---", f'title: "{title}"']
    if source.date is not None:
        frontmatter.append(f"date: {source.date.isoformat()}")
    if source.message_id:
        frontmatter.append(f'message-id: "{source.message_id}"')
    frontmatter.extend(['timezone: "Australia/Perth"', "---"])

    sections_included = ["summary", "discussion-notes"]
    lines = [
        *frontmatter,
        "",
        "> [!summary]",
        f"> {minutes.summary.strip() or 'Meeting transcript processed.'}",
        "",
        f"# {title}",
        "",
        "## Discussion Notes",
        "",
        render_minutes(minutes),
    ]

    if chapters is not None and chapters.chapters:
        sections_included.append("chapters")
        lines.extend(["", "## Chapters", "", render_chapters_markdown(chapters)])

    if transcript_markdown is not None:
        sections_included.append("transcript")
        lines.extend(["", "## Transcript", "", transcript_markdown.strip()])

    return "\n".join(lines).rstrip() + "\n", sections_included


def _yaml_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _source_timestamp_url(source: SourceArtifact, start: float) -> str:
    seconds = max(0, int(start))
    video_id = str(source.metadata.get("video_id") or "").strip()
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"
    separator = "&" if source.source_url and "?" in source.source_url else "?"
    return f"{source.source_url or ''}{separator}t={seconds}s"


def _source_chapter_paragraphs(
    turns: list[TranscriptTurn], *, target_chars: int = 700
) -> str:
    text = " ".join(turn.text.strip() for turn in turns if turn.text.strip())
    if not text:
        return ""
    sentences = re.split(
        r'(?<=[.!?])\s+(?=(?:["\'“”‘’(\[])?[A-Z0-9>])',
        text,
    )
    paragraphs: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and current_length + len(sentence) + 1 > target_chars:
            paragraphs.append(" ".join(current))
            current = []
            current_length = 0
        current.append(sentence)
        current_length += len(sentence) + 1
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def render_source_note_markdown(
    source: SourceArtifact,
    overview: SourceOverview,
    transcript: TranscriptArtifact,
    chapters: ChapterPlan,
) -> tuple[str, list[str]]:
    if not source.source_url:
        raise RenderPrimitiveError("Source note rendering requires source_url.")
    if not chapters.chapters:
        raise RenderPrimitiveError("Source note rendering requires chapters.")
    if not transcript.turns:
        raise RenderPrimitiveError("Source note rendering requires transcript turns.")

    title = source.title or "YouTube source note"
    metadata = source.metadata
    frontmatter = [
        "---",
        f"title: {_yaml_string(title)}",
        f"source: {_yaml_string(source.source_url)}",
    ]
    channel = metadata.get("channel") or source.organisation
    if channel:
        frontmatter.append(f"channel: {_yaml_string(channel)}")
    if source.date is not None:
        frontmatter.append(f"published: {source.date.isoformat()}")
    if source.message_id:
        frontmatter.append(f"message-id: {_yaml_string(source.message_id)}")
    duration = metadata.get("duration_seconds")
    if isinstance(duration, int | float):
        frontmatter.append(f"duration: {int(duration)}")
    for yaml_key, metadata_key in (
        ("video-id", "video_id"),
        ("subtitle-track", "subtitle_track"),
        ("subtitle-kind", "subtitle_kind"),
        ("capture-method", "capture_method"),
    ):
        value = metadata.get(metadata_key)
        if value:
            frontmatter.append(f"{yaml_key}: {_yaml_string(value)}")
    frontmatter.extend(["tags:", "  - source-note", "  - youtube", "---"])

    summary = " ".join(overview.summary.split())
    lines = [*frontmatter, "", "> [!summary]", f"> {summary}"]
    lines.extend(["", "## Key points", ""])
    lines.extend(f"- {point.strip()}" for point in overview.key_points if point.strip())

    lines.extend(["", "## Chapters", ""])
    for chapter in chapters.chapters:
        timestamp = format_timestamp(chapter.start)
        link = _source_timestamp_url(source, chapter.start)
        lines.append(f"- [{timestamp} — {chapter.title}]({link})")
        if chapter.summary.strip():
            lines.append(f"  - {chapter.summary.strip()}")

    lines.extend(["", "## Transcript", ""])
    for index, chapter in enumerate(chapters.chapters):
        next_start = (
            chapters.chapters[index + 1].start
            if index + 1 < len(chapters.chapters)
            else None
        )
        chapter_turns = [
            turn
            for turn in transcript.turns
            if turn.start >= chapter.start
            and (next_start is None or turn.start < next_start)
        ]
        timestamp = format_timestamp(chapter.start)
        link = _source_timestamp_url(source, chapter.start)
        lines.extend(
            [
                f"### [{timestamp} — {chapter.title}]({link})",
                "",
                f"> {chapter.summary.strip()}",
                "",
                _source_chapter_paragraphs(chapter_turns),
                "",
            ]
        )

    sections = ["summary", "key-points", "chapters", "transcript"]
    return "\n".join(lines).rstrip() + "\n", sections
