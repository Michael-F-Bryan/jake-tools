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
