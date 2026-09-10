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
