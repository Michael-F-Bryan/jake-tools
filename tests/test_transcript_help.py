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
    assert "obsidian-recording" in result.output
    assert "polish" in result.output
    # "source" was reserved here for a since-superseded internal design; the
    # bundle-store control plane (Phase 2) makes `transcript source ingest`
    # a genuinely public command (ADVERSARIAL-REVIEW §4.1), so it is no
    # longer in this "must stay hidden" list -- "transform" is the same
    # story for Phase 3A's `transform timeline`/`transform transcribe`
    # (see test_transcript_bundle_cli.py for both groups' own --help
    # coverage).
    for internal_group in (
        "schema",
        "parse",
        "stage",
        "render",
        "note",
        "verify",
        "recipe",
    ):
        assert f"\n  {internal_group}" not in result.output


def test_transcript_obsidian_recording_help_states_the_invariants() -> None:
    result = _invoke("transcript", "obsidian-recording", "--help")

    assert result.exit_code == 0
    for option in ("--work-dir", "--dry-run", "--json", "--model", "--effort"):
        assert option in result.output
    assert "ffmpeg" in result.output
    assert "scribe" in result.output
    assert "verification passes" in result.output


def test_main_help_does_not_list_the_deprecated_transcribe_alias() -> None:
    result = _invoke("--help")

    assert result.exit_code == 0
    assert "transcript" in result.output
    assert "\n  transcribe" not in result.output


def test_transcribe_alias_group_is_still_invokable_and_marked_deprecated() -> None:
    result = _invoke("transcribe", "--help")

    assert result.exit_code == 0
    assert "Deprecated" in result.output
    assert "transcript obsidian-recording" in result.output
    assert "obsidian-recording" in result.output
    assert "polish" in result.output


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
