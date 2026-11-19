"""``python -m jake_tools.mcp doctor``: deployment checks, one line each.

Prints every setting with its source (secrets only as set/unset), then one
line per check. Settings come from flags and the process environment only. Exits non-zero
if any required check fails. The report goes to stdout: unlike ``serve``,
this command is not the MCP transport.
"""

from __future__ import annotations

import os
from typing import Any

import click

from ..config import ConfigError, XdgPaths, load_config
from .checks import Check, run_checks
from .options import config_options


@click.command()
@config_options
def doctor(overrides: dict[str, Any]) -> None:
    """Check this deployment: settings, credentials, claude CLI, directories."""
    environ = os.environ
    checks: list[Check]
    try:
        config = load_config(environ, overrides=overrides)
    except ConfigError as exc:
        config = None
        checks = [Check(name="config", status="fail", detail=str(exc), required=True)]
    else:
        checks = run_checks(config, environ)

    click.echo("Environment")
    click.echo(f"  HOME = {environ.get('HOME', 'unset')}")
    click.echo(f"  HERMES_HOME = {environ.get('HERMES_HOME') or 'unset'}")
    xdg = XdgPaths.from_environ(environ)
    click.echo(f"  XDG state home = {xdg.state_home}")
    click.echo(f"  XDG cache home = {xdg.cache_home}")
    click.echo("Settings")
    if config is not None:
        for setting in config.settings():
            click.echo(
                f"  {setting.name} = {setting.display_value()}"
                f"  [{setting.display_source()}]"
            )
    click.echo("Checks")
    for check in checks:
        click.echo(f"  {check.status.upper():<5} {check.name}  {check.detail}")

    if any(check.failed_required for check in checks):
        raise SystemExit(1)
