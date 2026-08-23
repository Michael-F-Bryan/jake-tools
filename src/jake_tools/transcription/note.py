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
from collections.abc import Sequence
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


_MEETING_PREP_HEADING = "meeting prep"


def append_diarisation_hints(path: Path, lines: Sequence[str]) -> bool:
    """Append ``lines`` as bullets under Meeting Prep's ``Diarisation hints:`` list.

    This is the **only** sanctioned write into a human-owned note section
    (plan 006's speaker resolution). It is scoped structurally, not just by
    convention: the note is split into ``NoteSection``s exactly as
    :func:`parse_note` would, only the ``Meeting Prep`` section's body is
    ever touched, and the whole document is reassembled with
    :func:`render_body` — the same byte-for-byte reconstruction property
    plan 010's surgical writes will depend on. Frontmatter is preserved as
    raw text (never round-tripped through ``yaml.dump``, which could
    reorder keys or reformat values) so nothing outside Meeting Prep can
    change, not even incidentally.

    The ``Diarisation hints:`` bullet is created at the end of Meeting Prep
    when absent (and nothing else is created — no Meeting Prep section
    means this is a silent no-op). A line already present as a hint bullet
    (compared stripped, verbatim) is never duplicated, so calling this
    repeatedly with overlapping input is safe.

    Returns ``True`` if the file changed, ``False`` if every line was
    already present or there was no Meeting Prep section to append into.
    """
    text = path.read_text()
    prefix, body = _split_raw_frontmatter(text)
    sections = _split_sections(body)

    target_index = next(
        (
            index
            for index, section in enumerate(sections)
            if section.heading is not None
            and section.heading.strip().lower() == _MEETING_PREP_HEADING
        ),
        None,
    )
    if target_index is None:
        return False  # no Meeting Prep section: this helper creates nothing

    new_body, changed = _append_hints_to_section_body(
        sections[target_index].body, lines
    )
    if not changed:
        return False

    sections[target_index] = sections[target_index].model_copy(
        update={"body": new_body}
    )
    path.write_text(prefix + render_body(sections))
    return True


def _split_raw_frontmatter(text: str) -> tuple[str, str]:
    """Like :func:`_split_frontmatter`, but keeps the frontmatter as raw text.

    :func:`append_diarisation_hints` needs byte-fidelity outside Meeting
    Prep, and re-serialising frontmatter through ``yaml.dump`` risks
    reordering keys or reformatting values it merely parsed. This returns
    the frontmatter block's exact original text (delimiters included) so
    mutation only ever touches the body.
    """
    lines = text.split("\n")
    if not lines or lines[0] != "---":
        return "", text
    try:
        closing_index = lines.index("---", 1)
    except ValueError:
        return "", text
    prefix = "\n".join(lines[: closing_index + 1]) + "\n"
    body = "\n".join(lines[closing_index + 1 :])
    return prefix, body


def _append_hints_to_section_body(body: str, lines: Sequence[str]) -> tuple[str, bool]:
    wanted = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    if not wanted:
        return body, False

    split = body.split("\n")
    parent_index, children_start, children_end, child_indent = _locate_hints_bullet(
        split
    )

    if parent_index is None:
        block = "- Diarisation hints:\n" + "".join(f"\t- {line}\n" for line in wanted)
        prefix = "" if body == "" else (body if body.endswith("\n") else body + "\n")
        return prefix + block, True

    existing = {_bullet_text(split[i]) for i in range(children_start, children_end)}
    to_add = [line for line in wanted if line not in existing]
    if not to_add:
        return body, False

    indent = child_indent if child_indent is not None else "\t"
    inserted = [f"{indent}- {line}" for line in to_add]
    new_split = split[:children_end] + inserted + split[children_end:]
    return "\n".join(new_split), True


def _bullet_text(line: str) -> str:
    match = _BULLET_RE.match(line)
    assert match is not None
    return match.group(2).strip()


def _locate_hints_bullet(
    lines: list[str],
) -> tuple[int | None, int, int, str | None]:
    """Find the ``Diarisation hints:`` bullet's children within ``lines``.

    Mirrors :func:`_extract_diarisation_hints`'s scan so the two stay
    consistent about what counts as a child bullet. Returns
    ``(parent_index, children_start, children_end, child_indent)``, or
    ``(None, 0, 0, None)`` when no such bullet exists. ``child_indent`` is
    the exact leading whitespace of the existing children (``None`` when
    the bullet has none yet), so newly appended lines match their style.
    """
    for index, line in enumerate(lines):
        match = _BULLET_RE.match(line)
        if match is None:
            continue
        if match.group(2).strip().lower().rstrip(":") != _DIARISATION_HINTS_LABEL:
            continue
        parent_indent = len(match.group(1).expandtabs())
        cursor = index + 1
        child_indent: str | None = None
        while cursor < len(lines):
            candidate = lines[cursor]
            if not candidate.strip():
                break
            child = _BULLET_RE.match(candidate)
            if child is None:
                break
            if len(child.group(1).expandtabs()) <= parent_indent:
                break
            if child_indent is None:
                child_indent = child.group(1)
            cursor += 1
        return index, index + 1, cursor, child_indent
    return None, 0, 0, None
