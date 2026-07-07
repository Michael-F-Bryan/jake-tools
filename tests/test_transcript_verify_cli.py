from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcript"


def _build_normalised_turns(tmp_path: Path) -> tuple[Path, Path]:
    source_out = tmp_path / "source.json"
    turns_out = tmp_path / "turns.json"
    normalised_out = tmp_path / "turns.normalised.json"

    source_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "source",
            "gemini-text",
            str(FIXTURES_DIR / "gemini-text-sample.txt"),
            "--out",
            str(source_out),
        ],
    )
    assert source_result.exit_code == 0

    parse_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "parse",
            "gemini",
            str(source_out),
            "--out",
            str(turns_out),
        ],
    )
    assert parse_result.exit_code == 0

    normalise_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "normalise",
            str(turns_out),
            "--out",
            str(normalised_out),
        ],
    )
    assert normalise_result.exit_code == 0
    return turns_out, normalised_out


def test_transcript_verify_turns_passes_for_phase2_pipeline(tmp_path: Path) -> None:
    turns_out, normalised_out = _build_normalised_turns(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "turns",
            str(turns_out),
            str(normalised_out),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
    assert payload["failed_gate_ids"] == []


def test_transcript_verify_turns_fails_and_returns_non_zero(tmp_path: Path) -> None:
    turns_out, _ = _build_normalised_turns(tmp_path)
    after_path = tmp_path / "turns.empty.json"
    before_payload = json.loads(turns_out.read_text(encoding="utf-8"))
    before_payload["turns"] = []
    after_path.write_text(json.dumps(before_payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "turns",
            str(turns_out),
            str(after_path),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "fail"
    assert "turns.non-empty" in payload["failed_gate_ids"]


def test_transcript_verify_turns_flags_adjacent_duplicate_turns(
    tmp_path: Path,
) -> None:
    turns_out, _ = _build_normalised_turns(tmp_path)
    after_path = tmp_path / "turns.duplicated.json"
    payload = json.loads(turns_out.read_text(encoding="utf-8"))
    payload["turns"].insert(1, dict(payload["turns"][0]))
    payload["turns"][1]["start"] = payload["turns"][0]["end"]
    payload["turns"][1]["end"] = payload["turns"][1]["start"] + 1
    after_path.write_text(json.dumps(payload), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "turns",
            str(turns_out),
            str(after_path),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "turns.no-adjacent-duplicates" in payload["failed_gate_ids"]


def test_transcript_verify_boilerplate_detects_operational_chatter(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        "Loading the transcript-polisher skill.\nHello world.\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "boilerplate",
            str(transcript_path),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "boilerplate.no-operational-chatter" in payload["failed_gate_ids"]


def test_transcript_verify_chapters_and_note_pass(tmp_path: Path) -> None:
    turns_out, normalised_out = _build_normalised_turns(tmp_path)
    chapters_out = tmp_path / "chapters.json"
    note_path = tmp_path / "note.md"

    chapters_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "chapter-boundaries",
            str(normalised_out),
            "--out",
            str(chapters_out),
        ],
    )
    assert chapters_result.exit_code == 0
    chapter_count = len(
        json.loads(chapters_out.read_text(encoding="utf-8"))["chapters"]
    )

    chapters_verify_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "chapters",
            str(chapters_out),
            "--turns",
            str(turns_out),
            "--json",
        ],
    )
    assert chapters_verify_result.exit_code == 0
    chapters_verify_payload = json.loads(chapters_verify_result.output)
    assert chapters_verify_payload["status"] == "pass"

    note_headings = "\n".join(
        f"### Chapter {index + 1}" for index in range(chapter_count)
    )
    note_path.write_text(
        "\n".join(
            [
                "# Meeting",
                "",
                "## Meeting Notes",
                "- Captured outcomes",
                "",
                "## Chapters",
                "- 00:00 — Chapter 1",
                "",
                "## Transcript",
                "",
                note_headings,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    note_verify_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "note",
            str(note_path),
            "--chapters",
            str(chapters_out),
            "--json",
        ],
    )
    assert note_verify_result.exit_code == 0
    note_verify_payload = json.loads(note_verify_result.output)
    assert note_verify_payload["status"] == "pass"


def test_transcript_verify_note_missing_file_fails() -> None:
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "note",
            "/tmp/missing-note.md",
        ],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_transcript_verify_dumc_note_rejects_generic_meeting_shape(
    tmp_path: Path,
) -> None:
    note_path = tmp_path / "generic-note.md"
    note_path.write_text(
        "\n".join(
            [
                "# Meeting",
                "",
                "## Meeting Notes",
                "- Action: Follow up.",
                "- [ ] Assigned task",
                "",
                "## Chapters",
                "",
                "## Transcript",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "note",
            str(note_path),
            "--profile",
            "dumc",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "note.has-discussion-notes" in payload["failed_gate_ids"]
    assert "note.no-meeting-notes" in payload["failed_gate_ids"]
    assert "note.no-task-checkboxes" in payload["failed_gate_ids"]
    assert "note.no-action-labels" in payload["failed_gate_ids"]


def test_transcript_verify_dumc_note_accepts_profile_shape(tmp_path: Path) -> None:
    note_path = tmp_path / "dumc-note.md"
    note_path.write_text(
        "\n".join(
            [
                "---",
                'message-id: "msgraph-teams:meeting-123:transcript-abc"',
                'timezone: "Australia/Perth"',
                "---",
                "",
                "> [!summary]",
                "> Comms support briefing.",
                "",
                "## Discussion Notes",
                "",
                "- Confirmed support roster.",
                "\t- Joanne to check availability outside the transcript note.",
                "",
                "## Chapters",
                "",
                "- 00:00 — Briefing",
                "",
                "## Transcript",
                "",
                "### 00:00 — Briefing",
                "",
                "**Joanne Olsen** So much.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "verify",
            "note",
            str(note_path),
            "--profile",
            "dumc",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "pass"
