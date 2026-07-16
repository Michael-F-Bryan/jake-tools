from __future__ import annotations

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
