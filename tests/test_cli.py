from click.testing import CliRunner

from jake_tools.cli import main


def test_ai_watch_command_is_not_registered() -> None:
    result = CliRunner().invoke(main, ["ai-watch"])

    assert result.exit_code == 2
    assert "No such command 'ai-watch'" in result.output
