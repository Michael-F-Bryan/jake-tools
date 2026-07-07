from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcript"


def test_transcript_parse_gemini_reads_source_artifact(tmp_path: Path) -> None:
    source_out = tmp_path / "source.json"
    turns_out = tmp_path / "turns.json"
    source_input = FIXTURES_DIR / "gemini-text-sample.txt"

    source_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "source",
            "gemini-text",
            str(source_input),
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
            "--json",
        ],
    )

    assert parse_result.exit_code == 0
    payload = json.loads(parse_result.output)
    assert [turn["speaker"] for turn in payload["turns"]] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 1",
    ]
    assert payload["turns"][0]["start"] == 0.0
    assert payload["turns"][0]["end"] == 7.0
    assert (
        payload["turns"][0]["text"]
        == "Welcome everyone. We are kicking off the handoff."
    )
    assert payload["source_refs"][0]["source_ref"].endswith(":lines 5-6")


def test_transcript_parse_gemini_accepts_standalone_timestamp_blocks(
    tmp_path: Path,
) -> None:
    source_text = tmp_path / "gemini-standalone.txt"
    source_text.write_text(
        """Meeting notes

📖 Transcript

Jun 18, 2026

Meeting Jun 18, 2026 at 10:52
IST - Transcript
00:00:01

Akshay Sharma: Hey, Michael. How are you?
Michael Bryan: Yeah, I'm not too bad. What about you?

00:01:10

Akshay Sharma: I know that it's about medical insurance and to work in the
pipeline to enforce policy.
Michael Bryan: Yep.

Transcription ended after 00:40:18
""",
        encoding="utf-8",
    )
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "kind": "gemini-text",
                "source_path": str(source_text),
                "raw_text_path": str(source_text),
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "turns.json"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "parse",
            "gemini",
            str(source_path),
            "--out",
            str(out_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [turn["speaker"] for turn in payload["turns"]] == [
        "Akshay Sharma",
        "Michael Bryan",
        "Akshay Sharma",
        "Michael Bryan",
    ]
    assert payload["turns"][0]["start"] == 1.0
    assert payload["turns"][1]["end"] == 70.0
    assert payload["turns"][2]["text"] == (
        "I know that it's about medical insurance and to work in the "
        "pipeline to enforce policy."
    )
    assert payload["warnings"] == [
        "Parsed Gemini transcript from standalone timestamp blocks; turn times inside each block are interpolated."
    ]


def test_transcript_parse_scribe_converts_segments(tmp_path: Path) -> None:
    turns_out = tmp_path / "scribe-turns.json"
    scribe_input = FIXTURES_DIR / "scribe-sample.json"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "parse",
            "scribe",
            str(scribe_input),
            "--out",
            str(turns_out),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["turns"]) == 3
    assert payload["turns"][1]["speaker"] == "SPEAKER_02"
    assert payload["turns"][2]["speaker"] == "Michael"
    assert payload["source_refs"][2]["source_ref"].endswith("segments[2]")


def test_transcript_parse_teams_vtt_preserves_speaker_timestamps_and_refs(
    tmp_path: Path,
) -> None:
    vtt_path = FIXTURES_DIR / "teams-sample.vtt"
    source_path = tmp_path / "source.json"
    out_path = tmp_path / "teams-turns.json"
    source_path.write_text(
        json.dumps(
            {
                "kind": "msgraph-teams",
                "title": "Marine Rescue Comms Support presentation for SES",
                "message_id": "msgraph-teams:meeting-123:transcript-abc",
                "raw_text_path": str(vtt_path),
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "parse",
            "teams-vtt",
            str(source_path),
            "--out",
            str(out_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [turn["speaker"] for turn in payload["turns"]] == [
        "Joanne Olsen",
        "Michael Bryan",
        "Joanne Olsen",
    ]
    assert payload["turns"][0] == {
        "start": 3.277,
        "end": 3.917,
        "speaker": "Joanne Olsen",
        "text": "So much.",
    }
    assert payload["turns"][2]["start"] == 7.0
    assert payload["source_refs"][0]["turn_index"] == 0
    assert "cue line" in payload["source_refs"][0]["source_ref"]
