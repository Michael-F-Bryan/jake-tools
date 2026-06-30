from __future__ import annotations

import json

from click.testing import CliRunner

from jake_tools.cli import main


def test_transcript_help_lists_schema_and_placeholder_groups() -> None:
    result = CliRunner().invoke(main, ["transcript", "--help"])

    assert result.exit_code == 0
    assert "schema" in result.output
    assert "source" in result.output
    assert "parse" in result.output
    assert "transform" in result.output
    assert "stage" in result.output
    assert "render" in result.output
    assert "note" in result.output
    assert "verify" in result.output
    assert "recipe" in result.output


def test_transcript_schema_list_lists_all_public_artifacts() -> None:
    result = CliRunner().invoke(main, ["transcript", "schema", "list"])

    assert result.exit_code == 0
    names = {line.strip() for line in result.output.splitlines() if line.strip()}
    assert names == {
        "SourceArtifact",
        "AudioArtifact",
        "TranscriptArtifact",
        "ChapterPlan",
        "RenderedNote",
        "VerificationReport",
        "RunManifest",
    }


def test_transcript_schema_show_outputs_json_schema() -> None:
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "schema",
            "show",
            "TranscriptArtifact",
            "--format",
            "json-schema",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["title"] == "TranscriptArtifact"
    assert payload["type"] == "object"
    assert "turns" in payload["properties"]


def test_transcript_schema_example_outputs_parseable_json() -> None:
    result = CliRunner().invoke(
        main,
        ["transcript", "schema", "example", "TranscriptArtifact"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    assert "turns" in payload
