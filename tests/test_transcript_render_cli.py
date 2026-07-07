from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_render_transcript_with_chapters_and_json_metadata(tmp_path: Path) -> None:
    turns_path = tmp_path / "turns.polished.json"
    chapters_path = tmp_path / "chapters.json"
    out_path = tmp_path / "transcript.md"

    _write_json(
        turns_path,
        {
            "turns": [
                {"start": 0.0, "end": 6.0, "speaker": "Speaker 1", "text": "Welcome."},
                {"start": 7.0, "end": 14.0, "speaker": "Speaker 2", "text": "Thanks."},
            ],
            "source_refs": [],
            "speakers": {},
            "warnings": [],
        },
    )
    _write_json(
        chapters_path,
        {
            "chapters": [
                {
                    "start": 0.0,
                    "end": 14.0,
                    "title": "Introductions",
                    "summary": "Meeting opened and attendees greeted each other.",
                }
            ],
            "boundary_source": "llm",
        },
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "render",
            "transcript",
            str(turns_path),
            "--chapters",
            str(chapters_path),
            "--out",
            str(out_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    metadata = json.loads(result.output)
    assert metadata["sections_included"] == ["chapters", "transcript"]
    rendered = out_path.read_text(encoding="utf-8")
    assert "### 00:00 — Introductions" in rendered
    assert "**Speaker 1** Welcome." in rendered


def test_render_meeting_note_with_optional_transcript(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    minutes_path = tmp_path / "minutes.json"
    transcript_path = tmp_path / "transcript.md"
    out_path = tmp_path / "note.md"

    _write_json(
        source_path,
        {
            "kind": "gemini-text",
            "source_path": str(tmp_path / "source.txt"),
            "title": "Weekly Sync",
            "attendees": [],
            "attachments": [],
            "warnings": [],
        },
    )
    _write_json(
        minutes_path,
        {
            "summary": "Shared updates across projects.",
            "key_points": ["Confirmed sprint goals.", "Assigned next actions."],
        },
    )
    transcript_path.write_text("**Speaker 1** Hello.\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "render",
            "meeting-note",
            str(source_path),
            str(minutes_path),
            "--transcript",
            str(transcript_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    note = out_path.read_text(encoding="utf-8")
    assert "# Weekly Sync" in note
    assert "## Meeting Notes" in note
    assert "## Transcript" in note


def test_render_dumc_meeting_note_profile_uses_discussion_notes(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    minutes_path = tmp_path / "minutes.json"
    chapters_path = tmp_path / "chapters.json"
    transcript_path = tmp_path / "transcript.md"
    out_path = tmp_path / "dumc-note.md"

    _write_json(
        source_path,
        {
            "kind": "msgraph-teams",
            "title": "Marine Rescue Comms Support presentation for SES",
            "date": "2026-07-06",
            "message_id": "msgraph-teams:meeting-123:transcript-abc",
            "raw_text_path": str(tmp_path / "transcript.vtt"),
        },
    )
    _write_json(
        minutes_path,
        {
            "summary": "Comms support briefing for DUM-C.",
            "key_points": [
                "Confirmed comms support roster.",
                "Check volunteer availability before Sunday.",
            ],
        },
    )
    _write_json(
        chapters_path,
        {
            "chapters": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "title": "Briefing",
                    "summary": "Comms support discussed.",
                }
            ],
            "boundary_source": "deterministic",
        },
    )
    transcript_path.write_text(
        "### 00:00 — Briefing\n\n**Joanne Olsen** So much.\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "render",
            "meeting-note",
            str(source_path),
            str(minutes_path),
            "--profile",
            "dumc",
            "--chapters",
            str(chapters_path),
            "--transcript",
            str(transcript_path),
            "--out",
            str(out_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    metadata = json.loads(result.output)
    assert metadata["sections_included"] == [
        "summary",
        "discussion-notes",
        "chapters",
        "transcript",
    ]
    assert metadata["attachments_or_source_links"] == []
    note = out_path.read_text(encoding="utf-8")
    assert note.startswith("---\n")
    assert note.index("> [!summary]") < note.index("## Discussion Notes")
    assert 'message-id: "msgraph-teams:meeting-123:transcript-abc"' in note
    assert 'timezone: "Australia/Perth"' in note
    assert "## Discussion Notes" in note
    assert "## Chapters" in note
    assert "## Transcript" in note
    assert "## Meeting Notes" not in note
    assert ".vtt" not in note


def test_render_chapters_helper(tmp_path: Path) -> None:
    chapters_path = tmp_path / "chapters.json"
    out_path = tmp_path / "chapters.md"
    _write_json(
        chapters_path,
        {
            "chapters": [
                {
                    "start": 0.0,
                    "end": 30.0,
                    "title": "Opening",
                    "summary": "Opened meeting.",
                }
            ],
            "boundary_source": "deterministic",
        },
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "render",
            "chapters",
            str(chapters_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0
    assert "- 00:00 — Opening" in out_path.read_text(encoding="utf-8")
