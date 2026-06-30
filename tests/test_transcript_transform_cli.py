from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcript"


def _build_turns_artifact(tmp_path: Path) -> Path:
    source_out = tmp_path / "source.json"
    turns_out = tmp_path / "turns.json"
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
    return turns_out


def test_transcript_transform_strip_boilerplate_writes_cleaned_source(
    tmp_path: Path,
) -> None:
    text_path = tmp_path / "gemini.txt"
    text_path.write_text(
        "Loading the transcript-polisher skill.\n## Transcript\nHello.\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "kind": "gemini-text",
                "source_path": str(text_path),
                "raw_text_path": str(text_path),
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "source.clean.json"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "strip-boilerplate",
            str(source_path),
            "--out",
            str(out_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["warnings"] == ["Removed 1 boilerplate line(s)."]
    cleaned_path = Path(payload["raw_text_path"])
    assert cleaned_path.exists()
    assert "Loading the transcript-polisher skill." not in cleaned_path.read_text(
        encoding="utf-8"
    )


def test_transcript_transform_normalise_merge_split_and_chapter_boundaries(
    tmp_path: Path,
) -> None:
    turns_out = _build_turns_artifact(tmp_path)
    normalised_out = tmp_path / "turns.normalised.json"
    merged_out = tmp_path / "turns.merged.json"
    split_out = tmp_path / "chunks.json"
    chapters_out = tmp_path / "chapters.json"

    normalise_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "normalise",
            str(turns_out),
            "--out",
            str(normalised_out),
            "--json",
        ],
    )
    assert normalise_result.exit_code == 0
    normalised_payload = json.loads(normalise_result.output)
    assert len(normalised_payload["turns"]) == 3
    assert normalised_payload["turns"][0]["text"].startswith("Welcome everyone.")

    merge_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "merge-adjacent",
            str(normalised_out),
            "--max-gap",
            "6",
            "--out",
            str(merged_out),
            "--json",
        ],
    )
    assert merge_result.exit_code == 0
    merged_payload = json.loads(merge_result.output)
    assert len(merged_payload["turns"]) == 3

    split_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "split",
            str(merged_out),
            "--target-minutes",
            "0.1",
            "--out",
            str(split_out),
            "--json",
        ],
    )
    assert split_result.exit_code == 0
    split_payload = json.loads(split_result.output)
    assert split_payload["run_id"] == "split-manifest"
    assert split_payload["stages"]

    chapters_result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "chapter-boundaries",
            str(merged_out),
            "--window-minutes",
            "0.1",
            "--out",
            str(chapters_out),
            "--json",
        ],
    )
    assert chapters_result.exit_code == 0
    chapters_payload = json.loads(chapters_result.output)
    assert chapters_payload["boundary_source"] == "deterministic"
    assert chapters_payload["chapters"][0]["title"] == "Chapter 1"


def test_transcript_transform_merge_adjacent_merges_same_speaker(
    tmp_path: Path,
) -> None:
    turns_path = tmp_path / "turns.json"
    merged_out = tmp_path / "turns.merged.json"
    turns_path.write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "speaker": "Alex",
                        "text": "First update",
                    },
                    {
                        "start": 2.1,
                        "end": 4.0,
                        "speaker": "Alex",
                        "text": "Second update",
                    },
                    {
                        "start": 5.0,
                        "end": 6.0,
                        "speaker": "Blair",
                        "text": "New speaker",
                    },
                ],
                "source_refs": [],
                "speakers": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "merge-adjacent",
            str(turns_path),
            "--max-gap",
            "1",
            "--out",
            str(merged_out),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["turns"]) == 2
    assert payload["turns"][0]["speaker"] == "Alex"


def test_transcript_transform_merge_adjacent_rejects_negative_gap(
    tmp_path: Path,
) -> None:
    turns_out = _build_turns_artifact(tmp_path)
    out_path = tmp_path / "turns.merged.json"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "merge-adjacent",
            str(turns_out),
            "--max-gap",
            "-1",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code != 0
    assert "--max-gap must be zero or greater." in result.output


def test_transcript_transform_normalise_missing_input_fails() -> None:
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "transform",
            "normalise",
            "/tmp/does-not-exist-turns.json",
            "--out",
            "/tmp/out.json",
        ],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output
