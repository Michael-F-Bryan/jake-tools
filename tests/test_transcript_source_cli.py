from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcript"


def test_transcript_source_gemini_text_writes_source_artifact(tmp_path: Path) -> None:
    source_path = FIXTURES_DIR / "gemini-text-sample.txt"
    out_path = tmp_path / "source.json"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "source",
            "gemini-text",
            str(source_path),
            "--out",
            str(out_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kind"] == "gemini-text"
    assert payload["raw_text_path"] == str(source_path.resolve())

    written_payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert written_payload["source_path"] == str(source_path.resolve())


def test_transcript_source_obsidian_note_writes_recording_refs(tmp_path: Path) -> None:
    note_path = tmp_path / "meeting.md"
    recording_path = tmp_path / "recording.m4a"
    recording_path.write_bytes(b"fake-audio")
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
    out_path = tmp_path / "source.json"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "source",
            "obsidian-note",
            str(note_path),
            "--out",
            str(out_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kind"] == "obsidian-note"
    assert payload["title"] == "meeting"
    assert payload["attendees"] == ["Michael", "Alex"]
    assert payload["attachments"] == [str(recording_path.resolve())]
