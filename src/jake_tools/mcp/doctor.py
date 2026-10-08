"""``python -m jake_tools.mcp doctor``: deployment checks, one line each.

Phase 0 ships the command's shape only; the checks are implemented in
phase 1. The contract: print every setting with its source and expanded
path, never print a secret value, and exit non-zero if any required check
fails.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option(
    "--runs-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the runs directory for this check only.",
)
def doctor(runs_dir: Path | None) -> None:
    """Check this deployment: settings, credentials, claude CLI, directories."""
    raise click.ClickException("doctor is not implemented yet")
