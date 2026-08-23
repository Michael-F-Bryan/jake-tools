"""Parse an Obsidian meeting/ops-log note into a section-aware model.

The note is the pipeline's single input (attendees, diarisation hints,
recording embeds, context tags) and, eventually, its output surface: plan
010's integrate step writes tier-owned content back into specific sections.
That later surgical write depends on one property this module guarantees —
:func:`render_body` applied to :attr:`ParsedNote.sections` reproduces the
original body byte-for-byte. Keep that property intact if you touch the
section splitter.

Only ATX headings (``## Title``) occur in these notes — no setext headings —
so the splitter only needs to recognise the ``#`` prefix form.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .models import NoteContext

_HEADING_RE = re.compile(r"^(#{1,6}) (.*)$")
_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
_BULLET_RE = re.compile(r"^([ \t]*)-\s*(.*)$")
_DIARISATION_HINTS_LABEL = "diarisation hints"


class NoteSection(BaseModel):
    """One heading-delimited slice of a note body.

    ``body`` is the raw markdown strictly between this heading's line and
    the next heading's line (of any level — these notes never nest a
    heading deeper than its neighbours skip past, so "next heading" and
    "next same-or-higher heading" coincide here). The preamble before the
    first heading is its own section with ``heading=None`` and ``level=0``.
    """

    heading: str | None
    level: int
    body: str


class ParsedNote(BaseModel):
    """A parsed view of one Obsidian note, ready for the pipeline to consume."""

    path: str
    frontmatter: dict[str, Any]
    context: NoteContext
    attendees: list[str]
    diarisation_hints: list[str]
    embeds: list[str]
    sections: list[NoteSection]


def render_body(sections: list[NoteSection]) -> str:
    """Reconstruct a note body from its sections, byte-for-byte.

    The inverse of the split performed in :func:`parse_note`. Plan 010's
    surgical writes replace one section's ``body`` and call this to
    regenerate the whole document unchanged elsewhere.
    """
    parts: list[str] = []
    for section in sections:
        if section.heading is not None:
            parts.append(f"{'#' * section.level} {section.heading}\n")
        parts.append(section.body)
    return "".join(parts)


def parse_note(path: Path) -> ParsedNote:
    """Parse the note at ``path`` into attendees, hints, embeds, and sections."""
    text = path.read_text()
    frontmatter, body = _split_frontmatter(text)
    sections = _split_sections(body)
    return ParsedNote(
        path=str(path),
        frontmatter=frontmatter,
        context=_detect_context(frontmatter, path),
        attendees=_extract_attendees(frontmatter),
        diarisation_hints=_extract_diarisation_hints(body),
        embeds=_EMBED_RE.findall(body),
        sections=sections,
    )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.split("\n")
    if not lines or lines[0] != "---":
        return {}, text
    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        return {}, text
    yaml_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])
    loaded = yaml.safe_load(yaml_text)
    frontmatter = loaded if isinstance(loaded, dict) else {}
    return frontmatter, body


def _split_sections(body: str) -> list[NoteSection]:
    sections: list[NoteSection] = []
    heading: str | None = None
    level = 0
    buffer: list[str] = []

    def flush() -> None:
        sections.append(NoteSection(heading=heading, level=level, body="".join(buffer)))

    for line in body.splitlines(keepends=True):
        core = line[:-1] if line.endswith("\n") else line
        match = _HEADING_RE.match(core)
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2)
            buffer = []
        else:
            buffer.append(line)
    flush()
    return sections


def _detect_context(frontmatter: dict[str, Any], path: Path) -> NoteContext:
    tags = _as_list(frontmatter.get("tags"))
    for tag in tags:
        if isinstance(tag, str) and (tag == "ops-log" or tag.startswith("ops-log")):
            return "ops-log"
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("note/meeting"):
            return "meeting"
    if "DUM-C" in path.parts:
        return "ops-log"
    return "meeting"


def _extract_attendees(frontmatter: dict[str, Any]) -> list[str]:
    raw = frontmatter.get("Attendees")
    return [_strip_wikilink(item) for item in _as_list(raw) if isinstance(item, str)]


def _strip_wikilink(text: str) -> str:
    text = text.strip()
    if text.startswith("[[") and text.endswith("]]"):
        return text[2:-2]
    return text


def _extract_diarisation_hints(body: str) -> list[str]:
    lines = body.split("\n")
    hints: list[str] = []
    index = 0
    while index < len(lines):
        match = _BULLET_RE.match(lines[index])
        if (
            match
            and match.group(2).strip().lower().rstrip(":") == _DIARISATION_HINTS_LABEL
        ):
            parent_indent = len(match.group(1).expandtabs())
            index += 1
            while index < len(lines):
                line = lines[index]
                if not line.strip():
                    break
                child = _BULLET_RE.match(line)
                if child is None:
                    break
                child_indent = len(child.group(1).expandtabs())
                if child_indent <= parent_indent:
                    break
                hints.append(child.group(2).strip())
                index += 1
            continue
        index += 1
    return hints


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
