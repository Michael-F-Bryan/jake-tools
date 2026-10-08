"""``python -m jake_tools.mcp install-skills``: copy packaged skills into Hermes.

Phase 0 ships the command's shape only. The contract: default target is
``$HERMES_HOME/skills/``; ``--into`` for ``skills.external_dirs`` setups;
copy by default, ``--link`` symlinks for development; existing files are
only overwritten with ``--force``.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command("install-skills")
@click.option(
    "--into",
    type=click.Path(path_type=Path),
    default=None,
    help="Target skills directory. Defaults to $HERMES_HOME/skills.",
)
@click.option("--link", is_flag=True, help="Symlink instead of copying.")
@click.option("--force", is_flag=True, help="Overwrite existing skills.")
def install_skills(into: Path | None, link: bool, force: bool) -> None:
    """Install the packaged skills into a Hermes skills directory."""
    raise click.ClickException("install-skills is not implemented yet")
