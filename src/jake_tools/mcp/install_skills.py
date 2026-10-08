"""``python -m jake_tools.mcp install-skills``: copy packaged skills into Hermes.

Default target is ``$HERMES_HOME/skills/``; ``--into`` serves
``skills.external_dirs`` setups. Each ``skills/<name>/`` directory (one with a
``SKILL.md``) is installed whole, copied by default or symlinked with
``--link``. An existing ``<target>/<name>`` is only replaced with ``--force``,
and never when the target aliases the packaged sources.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import click
from pydantic import BaseModel, ConfigDict

from ..config import ConfigError, load_config

SkillOutcome = Literal["installed", "linked", "replaced", "skipped"]


class SkillInstallError(RuntimeError):
    """The install was refused before anything was touched."""


class SkillResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    outcome: SkillOutcome
    path: Path


def skill_dirs(source_root: Path) -> list[Path]:
    """Every ``<source_root>/<name>/`` containing a ``SKILL.md``."""
    if not source_root.is_dir():
        return []
    return sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    )


def _remove(path: Path) -> None:
    # A symlink (even to a directory) is unlinked, never followed.
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def install_skill_tree(
    source_root: Path, target: Path, *, link: bool, force: bool
) -> list[SkillResult]:
    """Install every skill under ``source_root`` into ``target``.

    Raises :class:`SkillInstallError`, with nothing touched, if ``target``
    or any destination resolves to or into the sources.
    """
    skills = skill_dirs(source_root)
    if not skills:
        return []
    real_root = source_root.resolve()
    if _overlaps(target.resolve(), real_root):
        raise SkillInstallError(
            f"target {target} resolves to or into the packaged skills {real_root}"
        )
    for skill in skills:
        destination = target / skill.name
        if not destination.exists():
            continue
        relink = (
            link
            and destination.is_symlink()
            and destination.resolve() == skill.resolve()
        )
        if not relink and _overlaps(destination.resolve(), skill.resolve()):
            raise SkillInstallError(
                f"{destination} resolves to or into its source {skill.resolve()}"
            )
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
        _place(skill, destination, link=link, replace=exists)
        outcome: SkillOutcome = (
            "replaced" if exists else ("linked" if link else "installed")
        )
        results.append(SkillResult(name=skill.name, outcome=outcome, path=destination))
    return results


def _place(skill: Path, destination: Path, *, link: bool, replace: bool) -> None:
    """Build the new entry beside ``destination``, then swap it into place."""
    staging = Path(tempfile.mkdtemp(prefix=f".{skill.name}-", dir=destination.parent))
    staged = staging / skill.name
    try:
        if link:
            staged.symlink_to(skill.resolve(), target_is_directory=True)
        else:
            shutil.copytree(skill, staged, symlinks=True)
        if replace:
            # Only now, with the new copy complete, is the old entry removed.
            _remove(destination)
        os.replace(staged, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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
    try:
        results = install_skill_tree(SKILLS_DIR, into, link=link, force=force)
    except SkillInstallError as exc:
        raise click.ClickException(str(exc)) from exc
    if not results:
        click.echo("No packaged skills found; nothing installed.")
        return
    for result in results:
        click.echo(f"{result.outcome}: {result.name} -> {result.path}")
    if any(r.outcome == "skipped" for r in results):
        raise click.ClickException("some skills already exist; use --force to replace")
