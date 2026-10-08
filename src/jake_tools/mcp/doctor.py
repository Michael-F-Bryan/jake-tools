"""``python -m jake_tools.mcp doctor``: deployment checks, one line each.

Prints every setting with its source (secrets only as set/unset), the parse
status of each candidate config file, then one line per check. Exits non-zero
if any required check fails. The report goes to stdout: unlike ``serve``,
this command is not the MCP transport.
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from ..config import ConfigError, load_config
from .checks import Check, inspect_config_files, run_checks


@click.command()
@click.option(
    "--runs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the runs directory for this check only.",
)
def doctor(runs_dir: Path | None) -> None:
    """Check this deployment: settings, credentials, claude CLI, directories."""
    environ = os.environ
    files = inspect_config_files(environ)
    overrides = {"claude.runs_dir": runs_dir} if runs_dir is not None else None
    checks: list[Check]
    try:
        config = load_config(environ, overrides=overrides)
    except ConfigError as exc:
        config = None
        checks = [Check(name="config", status="fail", detail=str(exc), required=True)]
    else:
        checks = run_checks(config, files, environ)

    click.echo("Settings")
    if config is not None:
        for setting in config.settings():
            click.echo(
                f"  {setting.name} = {setting.display_value()}"
                f"  [{setting.display_source()}]"
            )
    click.echo("Config files")
    for file in files:
        suffix = f": {file.detail}" if file.detail else ""
        click.echo(f"  {file.status:<7} {file.path}{suffix}")
    click.echo("Checks")
    for check in checks:
        click.echo(f"  {check.status.upper():<5} {check.name}  {check.detail}")

    if any(check.failed_required for check in checks):
        raise SystemExit(1)
