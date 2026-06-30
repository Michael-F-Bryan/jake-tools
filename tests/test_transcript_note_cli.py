from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main


def test_note_merge_writes_merged_markdown(tmp_path: Path) -> None:
    note_path = tmp_path / "meeting.md"
    generated_path = tmp_path / "generated.md"
    merged_path = tmp_path / "merged.md"
    note_path.write_text(
        "# Existing Note\n\n![[recording.m4a]]\n\n## Meeting Notes\n\n- old\n",
        encoding="utf-8",
    )
    generated_path.write_text(
        "## Meeting Notes\n\n- new notes\n\n## Chapters\n\n- 00:00 — Intro\n\n## Transcript\n\n### 00:00 — Intro\n\n**Speaker 1** Hello.\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "note",
            "merge",
            str(note_path),
            str(generated_path),
            "--out",
            str(merged_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["destination_path"] == str(note_path.resolve())
    merged = merged_path.read_text(encoding="utf-8")
    assert "![[recording.m4a]]" in merged
    assert "- new notes" in merged


def test_note_write_honors_dry_run(tmp_path: Path) -> None:
    note_path = tmp_path / "meeting.md"
    merged_path = tmp_path / "merged.md"
    note_path.write_text("# Existing\n", encoding="utf-8")
    merged_path.write_text("# Updated\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "note",
            "write",
            str(note_path),
            str(merged_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "dry-run:" in result.output
    assert note_path.read_text(encoding="utf-8") == "# Existing\n"


def test_note_attach_dry_run_and_mutating_modes(tmp_path: Path) -> None:
    note_path = tmp_path / "meeting.md"
    source_path = tmp_path / "source.pdf"
    note_path.write_text("# Meeting\n", encoding="utf-8")
    source_path.write_text("binary-ish", encoding="utf-8")

    dry_run = CliRunner().invoke(
        main,
        [
            "transcript",
            "note",
            "attach",
            str(note_path),
            str(source_path),
            "--name",
            "Source.pdf",
            "--dry-run",
        ],
    )
    assert dry_run.exit_code == 0
    assert not (tmp_path / "Attachments" / "Source.pdf").exists()
    assert "## Sources" not in note_path.read_text(encoding="utf-8")

    mutate = CliRunner().invoke(
        main,
        [
            "transcript",
            "note",
            "attach",
            str(note_path),
            str(source_path),
            "--name",
            "Source.pdf",
            "--json",
        ],
    )
    assert mutate.exit_code == 0
    payload = json.loads(mutate.output)
    assert payload["updated"] is True
    copied_path = tmp_path / "Attachments" / "Source.pdf"
    assert copied_path.exists()
    note_body = note_path.read_text(encoding="utf-8")
    assert "## Sources" in note_body
    assert "[Source.pdf](Attachments/Source.pdf)" in note_body
