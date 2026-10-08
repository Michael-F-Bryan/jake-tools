"""``python -m jake_tools.mcp install-skills``: copy packaged skills into Hermes.

Default target is ``$HERMES_HOME/skills/``; ``--into`` serves
``skills.external_dirs`` setups. Each ``skills/<name>/`` directory is
installed whole, copied by default or symlinked with ``--link``. An existing
``<target>/<name>`` is only replaced with ``--force``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

import click
from pydantic import BaseModel, ConfigDict

from ..config import ConfigError, load_config

SkillOutcome = Literal["installed", "linked", "replaced", "skipped"]


class SkillResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    outcome: SkillOutcome
    path: Path


def _remove(path: Path) -> None:
    # A symlink (even to a directory) is unlinked, never followed.
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def install_skill_tree(
    source_root: Path, target: Path, *, link: bool, force: bool
) -> list[SkillResult]:
    """Install every ``<source_root>/<name>/`` directory into ``target``."""
    if not source_root.is_dir():
        return []
    skills = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not skills:
        return []
    target.mkdir(parents=True, exist_ok=True)
    results: list[SkillResult] = []
    for skill in skills:
        destination = target / skill.name
        exists = destination.is_symlink() or destination.exists()
        if exists and not force:
            results.append(
                SkillResult(name=skill.name, outcome="skipped", path=destination)
            )
            continue
        if exists:
            _remove(destination)
        if link:
            destination.symlink_to(skill.resolve(), target_is_directory=True)
        else:
            shutil.copytree(skill, destination, symlinks=True)
        outcome: SkillOutcome = (
            "replaced" if exists else ("linked" if link else "installed")
        )
        results.append(SkillResult(name=skill.name, outcome=outcome, path=destination))
    return results


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
    from .server import SKILLS_DIR

    if into is None:
        try:
            hermes_home = load_config(os.environ).hermes_home
        except ConfigError as exc:
            raise click.ClickException(str(exc)) from exc
        if hermes_home is None:
            raise click.ClickException("no target: pass --into DIR or set HERMES_HOME")
        into = hermes_home / "skills"
    results = install_skill_tree(SKILLS_DIR, into, link=link, force=force)
    if not results:
        click.echo("No packaged skills found; nothing installed.")
        return
    for result in results:
        click.echo(f"{result.outcome}: {result.name} -> {result.path}")
    if any(r.outcome == "skipped" for r in results):
        raise click.ClickException("some skills already exist; use --force to replace")
