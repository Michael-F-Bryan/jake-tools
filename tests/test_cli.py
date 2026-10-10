from click.testing import CliRunner

from jake_tools.cli import main


def test_ai_watch_command_is_not_registered() -> None:
    result = CliRunner().invoke(main, ["ai-watch"])

    assert result.exit_code == 2
    assert "No such command 'ai-watch'" in result.output


def test_codex_usage_alert_command_is_registered() -> None:
    result = CliRunner().invoke(main, ["codex-usage-alert", "--help"])

    assert result.exit_code == 0
    assert "20%, 10%, or 5%" in result.output


def test_transcription_commands_are_not_registered() -> None:
    runner = CliRunner()
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0
    assert "transcri" not in help_result.output
    for command in ("transcribe", "transcript"):
        result = runner.invoke(main, [command])
        assert result.exit_code == 2
        assert f"No such command {command!r}" in result.output
