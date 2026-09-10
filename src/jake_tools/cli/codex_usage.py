from __future__ import annotations

from pathlib import Path

import click

from jake_tools.codex_usage import (
    CodexUsageClient,
    CodexUsageError,
    run_codex_usage_alert,
)


@click.command("codex-usage-alert")
@click.option(
    "--auth-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path.home() / ".hermes" / "auth.json",
    show_default=True,
    help="Hermes auth store containing the OpenAI Codex OAuth token.",
)
@click.option(
    "--state-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path.home() / ".hermes" / "cron" / "state" / "codex_usage_watch.json",
    show_default=True,
    help="Threshold-crossing state used to suppress duplicate alerts.",
)
def codex_usage_alert(auth_file: Path, state_file: Path) -> None:
    """Print a warning only when a Codex quota window crosses 20%, 10%, or 5%."""
    try:
        lines = run_codex_usage_alert(
            client=CodexUsageClient(auth_path=auth_file),
            state_path=state_file,
        )
    except CodexUsageError as exc:
        raise click.ClickException(str(exc)) from exc

    for line in lines:
        click.echo(line)
