from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main


def _invoke(*args: str):
    return CliRunner().invoke(main, list(args))


def test_transcript_help_exposes_only_operator_tasks() -> None:
    result = _invoke("transcript", "--help")

    assert result.exit_code == 0
    assert "teams-meeting" in result.output
    assert "youtube" in result.output
    for internal_group in (
        "schema",
        "source",
        "parse",
        "transform",
        "stage",
        "render",
        "note",
        "verify",
        "recipe",
    ):
        assert f"\n  {internal_group}" not in result.output


def test_transcript_youtube_help_is_task_shaped() -> None:
    result = _invoke("transcript", "youtube", "--help")

    assert result.exit_code == 0
    assert "URL" in result.output
    for option in (
        "--out-dir",
        "--language",
        "--vault-note",
        "--dry-run",
        "--json",
        "--model",
        "--effort",
    ):
        assert option in result.output
    assert "--show-plan" not in result.output
    assert "--chapter-window-minutes" not in result.output


def test_transcript_teams_meeting_help_is_task_shaped() -> None:
    result = _invoke("transcript", "teams-meeting", "--help")

    assert result.exit_code == 0
    for option in (
        "--account",
        "--profile",
        "--event-id",
        "--out-dir",
        "--vault-note",
        "--dry-run",
        "--json",
    ):
        assert option in result.output
    assert "--show-plan" not in result.output


def test_teams_meeting_default_vault_requires_dumc_profile(tmp_path: Path) -> None:
    result = _invoke(
        "transcript",
        "teams-meeting",
        "--out-dir",
        str(tmp_path),
        "--profile",
        "default",
        "--write-vault",
    )

    assert result.exit_code == 2
    assert "--write-vault requires --profile dumc" in result.output


def test_teams_meeting_rejects_two_vault_destinations(tmp_path: Path) -> None:
    result = _invoke(
        "transcript",
        "teams-meeting",
        "--out-dir",
        str(tmp_path),
        "--vault-note",
        str(tmp_path / "meeting.md"),
        "--write-vault",
    )

    assert result.exit_code == 2
    assert "Use either --vault-note or --write-vault" in result.output
