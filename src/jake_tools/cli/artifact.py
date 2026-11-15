"""``jake-tools artifact``: the artifact operations from a shell.

Thin: the same functions the ``artifact_*`` tools call. ``read`` is read-only
and is the independent way to check an exact URL. Both commands use an
undocumented Claude Code mechanism (see ``docs/claude-tools.md``).
"""

from __future__ import annotations

import anyio
import click
from pydantic import BaseModel

from jake_tools.artifacts import (
    ArtifactReadResult,
    SourceError,
    publish,
    read,
    update,
)
from jake_tools.operations.errors import ToolError


def _emit(result: BaseModel, as_json: bool) -> None:
    if as_json:
        click.echo(result.model_dump_json(indent=2, exclude_none=True))
        return
    data = result.model_dump(exclude_none=True)
    for key, value in data.items():
        if isinstance(value, dict):
            click.echo(f"{key}:")
            for inner_key, inner in value.items():
                click.echo(f"  {inner_key}: {inner}")
        else:
            click.echo(f"{key}: {value}")


def _fail(exc: SourceError | ToolError, as_json: bool) -> click.ClickException:
    """The error, with everything a caller needs to act on an uncertain write.

    A ``ToolError`` carries structured detail (``publish_attempted``,
    ``retry_safe``, ``urls_seen``, denials): dropping it would hide that an
    artifact may exist. ``--json`` prints the full payload to stderr."""
    if isinstance(exc, SourceError):
        return click.ClickException(str(exc))
    payload = exc.payload
    if as_json:
        return click.ClickException(
            payload.model_dump_json(indent=2, exclude_none=True)
        )
    lines = [f"{exc.code}: {exc.message}"]
    for key, value in (exc.detail or {}).items():
        lines.append(f"  {key}: {value}")
    return click.ClickException("\n".join(lines))


@click.group()
def artifact() -> None:
    """Publish an HTML file as a private Claude Artifact, or read one back."""


@artifact.command("publish")
@click.argument("file_path", type=click.Path(dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
def publish_command(file_path: str, as_json: bool) -> None:
    """Publish FILE_PATH (an absolute .html path) as a NEW private artifact, then verify it."""

    try:
        result = anyio.run(publish, file_path)
    except (SourceError, ToolError) as exc:
        raise _fail(exc, as_json) from exc
    _emit(result, as_json)
    if not result.verified:
        raise click.ClickException("published but NOT verified; see problems")


@artifact.command("read")
@click.argument("url")
@click.option(
    "--file",
    "file_path",
    help="Absolute .html path; exit nonzero unless the hosted file equals it byte for byte.",
)
@click.option(
    "--expected-version-id", help="Exit nonzero unless this is the live version id."
)
@click.option(
    "--output",
    "output_path",
    help="Save the hosted file (shell removed) to a NEW path.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
def read_command(
    url: str,
    file_path: str | None,
    expected_version_id: str | None,
    output_path: str | None,
    as_json: bool,
) -> None:
    """Read artifact URL back (read-only): visibility, version and content hash."""

    async def run() -> ArtifactReadResult:
        return await read(
            url,
            file_path=file_path,
            expected_version_id=expected_version_id,
            output_path=output_path,
        )

    try:
        result = anyio.run(run)
    except (SourceError, ToolError) as exc:
        raise _fail(exc, as_json) from exc
    _emit(result, as_json)
    # An explicit comparison that fails, or cannot be made, is a failure: the
    # result above is printed first so the evidence is not lost.
    failed: list[str] = []
    if file_path is not None and result.matches_file is not True:
        failed.append(
            "the hosted file does not match --file (or could not be isolated)"
        )
    if expected_version_id is not None and result.version_matches is not True:
        failed.append(
            f"the live version is {result.version_id!r}, not {expected_version_id!r}"
        )
    if failed:
        raise click.ClickException("; ".join(failed))


@artifact.command("update")
@click.argument("url")
@click.argument("file_path", type=click.Path(dir_okay=False))
@click.option(
    "--expected-version-id",
    required=True,
    help="The live version you believe it is at; a stale one writes nothing.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
def update_command(
    url: str, file_path: str, expected_version_id: str, as_json: bool
) -> None:
    """Publish FILE_PATH as the next version of private artifact URL, then verify it."""

    async def run() -> BaseModel:
        return await update(url, file_path, expected_version_id)

    try:
        result = anyio.run(run)
    except (SourceError, ToolError) as exc:
        raise _fail(exc, as_json) from exc
    _emit(result, as_json)
    if not getattr(result, "verified", False):
        raise click.ClickException("updated but NOT verified; see problems")
