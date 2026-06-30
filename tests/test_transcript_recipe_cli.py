from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main


def _write_note_and_recording(tmp_path: Path) -> Path:
    recording_path = tmp_path / "recording.m4a"
    recording_path.write_bytes(b"fake-audio")
    note_path = tmp_path / "meeting.md"
    note_path.write_text(
        "\n".join(
            [
                "---",
                "Attendees:",
                "  - [[Michael]]",
                "  - Alex",
                "---",
                "",
                "# Meeting",
                "![Recording](recording.m4a)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return note_path


def test_transcript_recipe_obsidian_recording_show_plan(tmp_path: Path) -> None:
    note_path = _write_note_and_recording(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "recipe",
            "obsidian-recording",
            str(note_path),
            "--show-plan",
        ],
    )

    assert result.exit_code == 0
    assert "recipe: obsidian-recording" in result.output
    assert "source.obsidian-note" in result.output
    assert "note.write" in result.output


def test_transcript_recipe_obsidian_recording_show_plan_json_shape(
    tmp_path: Path,
) -> None:
    note_path = _write_note_and_recording(tmp_path)
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "recipe",
            "obsidian-recording",
            str(note_path),
            "--show-plan",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["recipe"] == "obsidian-recording"
    assert payload["supports"]["show_plan"] is True
    assert isinstance(payload["steps"], list)
    assert payload["steps"][0]["primitive"] == "source.obsidian-note"
    assert payload["steps"][-1]["primitive"] == "note.write"
