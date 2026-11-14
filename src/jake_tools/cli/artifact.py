"""``jake-tools artifact``: the artifact operations from a shell.

Thin: the same functions the ``artifact_*`` tools call. ``read`` is read-only
and is the independent way to check an exact URL. Both commands use an
undocumented Claude Code mechanism (see ``docs/claude-tools.md``).
"""

from __future__ import annotations

import anyio
import click
from pydantic import BaseModel

from jake_tools.artifacts import SourceError, publish, read
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


def _fail(exc: SourceError | ToolError) -> click.ClickException:
    if isinstance(exc, ToolError):
        return click.ClickException(f"{exc.code}: {exc.message}")
    return click.ClickException(str(exc))


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
        raise _fail(exc) from exc
    _emit(result, as_json)
    if not result.verified:
        raise click.ClickException("published but NOT verified; see problems")


@artifact.command("read")
@click.argument("url")
@click.option(
    "--file", "file_path", help="Absolute .html path to compare byte for byte."
)
@click.option("--expected-version-id", help="Fail unless this is the live version id.")
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

    async def run() -> BaseModel:
        return await read(
            url,
            file_path=file_path,
            expected_version_id=expected_version_id,
            output_path=output_path,
        )

    try:
        result = anyio.run(run)
    except (SourceError, ToolError) as exc:
        raise _fail(exc) from exc
    _emit(result, as_json)
