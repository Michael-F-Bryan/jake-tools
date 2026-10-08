"""``python -m jake_tools.mcp worker RUN_DIR``: the delegated-task worker.

Hidden sub-command; the only place a delegated task actually runs. Phase 0
ships the command's shape only. The contract is in
:mod:`jake_tools.claude_runs` and the design's "Claude worker" section.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command(hidden=True)
@click.argument(
    "run_dir", type=click.Path(path_type=Path, file_okay=False, exists=True)
)
def worker(run_dir: Path) -> None:
    """Run the delegated task described by RUN_DIR (internal)."""
    raise click.ClickException("worker is not implemented yet")
