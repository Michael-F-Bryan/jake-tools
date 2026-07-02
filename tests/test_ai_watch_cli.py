from __future__ import annotations

from click.testing import CliRunner

from jake_tools.cli import main


def test_ai_watch_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["ai-watch", "--help"])
    assert result.exit_code == 0
    for name in (
        "collect",
        "fetch",
        "scout",
        "curate",
        "obsidian-sync",
        "digest",
        "deliver",
        "run",
        "audit",
    ):
        assert name in result.output
