"""``python -m jake_tools.mcp worker RUN_DIR``: the delegated-task worker.

Hidden sub-command; the only place a delegated task actually runs. The body
is :func:`jake_tools.claude_runs.worker.main`; the contract is in
:mod:`jake_tools.claude_runs` and the design's "Claude worker" section.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.command(hidden=True)
@click.argument(
    "run_dir", type=click.Path(path_type=Path, file_okay=False, exists=True)
)
def worker(run_dir: Path) -> None:
    """Run the delegated task described by RUN_DIR (internal)."""
    from ..claude_runs.worker import main

    sys.exit(main(run_dir.resolve()))
