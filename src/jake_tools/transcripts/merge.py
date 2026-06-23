from __future__ import annotations

import re

from .models import Chapter, MeetingMinutes, MergeReport, SpeakerMapping, TranscriptTurn

GENERATED_HEADINGS = ("## Meeting Notes", "## Chapters", "## Transcript")
RECORDING_EMBED_RE = re.compile(
    r"!\[\[[^\]]+\.(?:mp3|m4a|wav|mp4|webm|ogg|flac)(?:\|[^\]]+)?\]\]", re.IGNORECASE
)


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{secs:02}"
    return f"{minutes:02}:{secs:02}"


def transcript_heading(title: str, start: float) -> str:
    return f"### {format_timestamp(start)} — {title}"


def speaker_name(label: str, mapping: SpeakerMapping | None) -> str:
    if mapping and label in mapping.mapping:
        return mapping.mapping[label].name
    return label


def render_transcript(
    turns: list[TranscriptTurn], mapping: SpeakerMapping | None = None
) -> str:
    lines: list[str] = []
    for turn in turns:
        name = speaker_name(turn.speaker, mapping)
        lines.append(f"**{name}** {turn.text}")
    return "\n\n".join(lines)


def render_chaptered_transcript(
    turns: list[TranscriptTurn],
    chapters: list[Chapter],
    mapping: SpeakerMapping | None = None,
) -> str:
    sections: list[str] = []
    for index, chapter in enumerate(chapters):
        next_start = chapters[index + 1].start if index + 1 < len(chapters) else None
        chapter_turns = [
            turn
            for turn in turns
            if turn.start >= chapter.start
            and (next_start is None or turn.start < next_start)
        ]
        sections.append(
            "\n\n".join(
                [
                    transcript_heading(chapter.title, chapter.start),
                    render_transcript(chapter_turns, mapping),
                ]
            ).strip()
        )
    return "\n\n".join(section for section in sections if section)


def render_chapters_index(chapters: list[Chapter]) -> str:
    return "\n".join(
        f"- {format_timestamp(chapter.start)} — {chapter.title}" for chapter in chapters
    )


def render_minutes(minutes: MeetingMinutes) -> str:
    notes = [item.strip() for item in minutes.key_points if item.strip()]
    if not notes and minutes.summary.strip():
        notes = [minutes.summary.strip()]
    if not notes:
        notes = ["No high-level meeting notes recorded"]
    return "\n".join(f"- {note}" for note in notes)


def _strip_existing_generated_sections(note_body: str) -> str:
    starts = [
        note_body.find(heading)
        for heading in GENERATED_HEADINGS
        if heading in note_body
    ]
    starts = [index for index in starts if index >= 0]
    if not starts:
        return note_body.rstrip()
    return note_body[: min(starts)].rstrip()


def merge_note(
    note_body: str,
    *,
    transcript_body: str,
    chapters: list[Chapter],
    minutes: MeetingMinutes,
) -> str:
    prefix = _strip_existing_generated_sections(note_body)
    sections: list[str] = [prefix]

    sections.append("## Meeting Notes\n\n" + render_minutes(minutes))
    sections.append("## Chapters\n\n" + render_chapters_index(chapters))
    sections.append("## Transcript\n\n" + transcript_body)
    return (
        "\n\n".join(section.strip() for section in sections if section.strip()) + "\n"
    )


def build_merge_report(
    original_body: str,
    merged_body: str,
    chapter_count: int,
) -> MergeReport:
    transcript_heading_count = merged_body.count("### ")
    original_content_preserved = (
        _strip_existing_generated_sections(original_body) in merged_body
    )

    original_embeds = set(RECORDING_EMBED_RE.findall(original_body))
    recording_embed_preserved = all(embed in merged_body for embed in original_embeds)

    return MergeReport(
        status="pass"
        if original_content_preserved and recording_embed_preserved
        else "fail",
        transcript_heading_count=transcript_heading_count,
        chapter_count=chapter_count,
        original_content_preserved=original_content_preserved,
        recording_embed_preserved=recording_embed_preserved,
    )
