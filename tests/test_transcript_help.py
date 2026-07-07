from __future__ import annotations

import re

from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.transcripts.models import list_public_transcript_artifact_models

GROUP_HELP_CASES: tuple[tuple[list[str], tuple[str, ...]], ...] = (
    (
        ["transcript", "--help"],
        (
            "schema",
            "source",
            "parse",
            "transform",
            "stage",
            "render",
            "note",
            "verify",
            "recipe",
        ),
    ),
    (["transcript", "schema", "--help"], ("schema", "artefact", "show", "example")),
    (
        ["transcript", "source", "--help"],
        (
            "SourceArtifact",
            "obsidian-note",
            "gemini-pdf",
            "gemini-text",
            "teams-meeting",
        ),
    ),
    (
        ["transcript", "parse", "--help"],
        ("TranscriptArtifact", "gemini", "scribe", "teams-vtt"),
    ),
    (
        ["transcript", "transform", "--help"],
        ("source", "transcript", "strip-boilerplate", "chapter-boundaries"),
    ),
    (
        ["transcript", "stage", "--help"],
        ("LLM", "structured", "minutes", "map-speakers"),
    ),
    (
        ["transcript", "render", "--help"],
        ("markdown", "meeting-note", "transcript", "chapters"),
    ),
    (["transcript", "note", "--help"], ("merge", "write", "attach", "mutation")),
    (["transcript", "verify", "--help"], ("stable", "check", "boilerplate", "note")),
    (
        ["transcript", "recipe", "--help"],
        ("workflow", "obsidian-recording", "teams-meeting"),
    ),
)

LEAF_COMMANDS: tuple[list[str], ...] = (
    ["transcript", "schema", "list", "--help"],
    ["transcript", "schema", "show", "--help"],
    ["transcript", "schema", "example", "--help"],
    ["transcript", "source", "obsidian-note", "--help"],
    ["transcript", "source", "gemini-pdf", "--help"],
    ["transcript", "source", "gemini-text", "--help"],
    ["transcript", "source", "teams-meeting", "--help"],
    ["transcript", "parse", "gemini", "--help"],
    ["transcript", "parse", "scribe", "--help"],
    ["transcript", "parse", "teams-vtt", "--help"],
    ["transcript", "transform", "strip-boilerplate", "--help"],
    ["transcript", "transform", "normalise", "--help"],
    ["transcript", "transform", "merge-adjacent", "--help"],
    ["transcript", "transform", "split", "--help"],
    ["transcript", "transform", "chapter-boundaries", "--help"],
    ["transcript", "stage", "polish", "--help"],
    ["transcript", "stage", "map-speakers", "--help"],
    ["transcript", "stage", "title-chapters", "--help"],
    ["transcript", "stage", "minutes", "--help"],
    ["transcript", "render", "transcript", "--help"],
    ["transcript", "render", "chapters", "--help"],
    ["transcript", "render", "meeting-note", "--help"],
    ["transcript", "note", "merge", "--help"],
    ["transcript", "note", "write", "--help"],
    ["transcript", "note", "attach", "--help"],
    ["transcript", "verify", "boilerplate", "--help"],
    ["transcript", "verify", "turns", "--help"],
    ["transcript", "verify", "chapters", "--help"],
    ["transcript", "verify", "note", "--help"],
    ["transcript", "recipe", "obsidian-recording", "--help"],
    ["transcript", "recipe", "teams-meeting", "--help"],
)


def _invoke(args: list[str]):
    return CliRunner().invoke(main, args)


def test_transcript_group_help_surfaces_progressive_disclosure() -> None:
    for args, expected_tokens in GROUP_HELP_CASES:
        result = _invoke(args)
        assert result.exit_code == 0
        output = result.output.lower()
        for token in expected_tokens:
            assert token.lower() in output, (args, token, result.output)


def test_transcript_leaf_command_help_includes_interface_sections() -> None:
    for args in LEAF_COMMANDS:
        result = _invoke(args)
        assert result.exit_code == 0
        assert "Input:" in result.output, (args, result.output)
        assert "Output:" in result.output, (args, result.output)
        assert re.search(r"side\s+effects:", result.output, re.IGNORECASE), (
            args,
            result.output,
        )
        assert re.search(r"next\s+steps:", result.output, re.IGNORECASE), (
            args,
            result.output,
        )


def test_recipe_help_mentions_show_plan() -> None:
    result = _invoke(["transcript", "recipe", "obsidian-recording", "--help"])
    assert result.exit_code == 0
    assert "--show-plan" in result.output


def test_stage_help_mentions_provider_and_model_flags() -> None:
    for leaf in ("polish", "map-speakers", "title-chapters", "minutes"):
        result = _invoke(["transcript", "stage", leaf, "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--default-model" in result.output


def test_mutating_note_help_mentions_dry_run() -> None:
    for leaf in ("write", "attach"):
        result = _invoke(["transcript", "note", leaf, "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output


def test_schema_list_includes_every_public_artifact_model() -> None:
    result = _invoke(["transcript", "schema", "list"])
    assert result.exit_code == 0
    listed_names = {line.strip() for line in result.output.splitlines() if line.strip()}
    expected_names = {
        model.__name__ for model in list_public_transcript_artifact_models()
    }
    assert listed_names == expected_names
