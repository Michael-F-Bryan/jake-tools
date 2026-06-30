from __future__ import annotations

import shutil
from pathlib import Path

from .merge import GENERATED_HEADINGS, _strip_existing_generated_sections


class NotePrimitiveError(RuntimeError):
    pass


def merge_generated_note(existing_note: str, generated_note: str) -> str:
    prefix = _strip_existing_generated_sections(existing_note)

    starts = [
        generated_note.find(heading)
        for heading in GENERATED_HEADINGS
        if heading in generated_note
    ]
    starts = [index for index in starts if index >= 0]
    if not starts:
        raise NotePrimitiveError(
            "Generated note is missing all expected sections: "
            + ", ".join(GENERATED_HEADINGS)
        )

    generated_sections = generated_note[min(starts) :].strip()
    if not generated_sections:
        raise NotePrimitiveError("Generated note sections are empty.")

    if prefix.strip():
        return prefix.rstrip() + "\n\n" + generated_sections + "\n"
    return generated_sections + "\n"


def write_note(path: Path, merged_note: str, *, dry_run: bool) -> bool:
    if dry_run:
        return False
    path.write_text(merged_note, encoding="utf-8")
    return True


def _source_link_markdown(
    note_path: Path,
    attached_path: Path,
    *,
    display_name: str,
) -> str:
    try:
        target = attached_path.relative_to(note_path.parent).as_posix()
    except ValueError:
        target = attached_path.as_posix()
    return f"[{display_name}]({target})"


def _append_source_link(note_body: str, link_markdown: str) -> str:
    if link_markdown in note_body:
        return note_body if note_body.endswith("\n") else note_body + "\n"

    if "## Sources" not in note_body:
        base = note_body.rstrip()
        if base:
            return base + "\n\n## Sources\n\n- " + link_markdown + "\n"
        return "## Sources\n\n- " + link_markdown + "\n"

    base = note_body.rstrip()
    return base + "\n- " + link_markdown + "\n"


def attach_source_file(
    note_path: Path,
    source_path: Path,
    *,
    display_name: str | None,
    attachments_dir: str,
    dry_run: bool,
) -> tuple[Path, str, bool]:
    if not source_path.exists():
        raise NotePrimitiveError(f"Source attachment does not exist: {source_path}")

    destination_dir = (note_path.parent / attachments_dir).resolve()
    destination_path = destination_dir / (display_name or source_path.name)
    attachment_display_name = display_name or destination_path.name

    if not dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

        note_body = note_path.read_text(encoding="utf-8")
        link_markdown = _source_link_markdown(
            note_path, destination_path, display_name=attachment_display_name
        )
        updated_note = _append_source_link(note_body, link_markdown)
        note_path.write_text(updated_note, encoding="utf-8")

    return destination_path, attachment_display_name, not dry_run
