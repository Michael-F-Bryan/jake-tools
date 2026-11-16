"""``jake-tools files``: the ``sharepoint_*`` operations from a shell.

Thin, like ``artifact``: each command builds the same operation handler the MCP
server registers (``OPERATION.build(config)``) and runs it, so the arguments
are validated, the Microsoft identity is the configured one
(``docs/microsoft-auth.md``; ``.env`` is honoured like every other command), and
the result and the failure detail are the tool's own. There is no second
code path and no ``az login`` identity.

Exit status is nonzero on any error, including an uncertain write, whose
``outcome``/``next_step`` are printed.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any

import anyio
import click
from pydantic import BaseModel

from jake_tools.config import ConfigError, load_config
from jake_tools.operations import (
    Operation,
    sharepoint_download,
    sharepoint_get,
    sharepoint_list,
    sharepoint_pdf,
    sharepoint_replace,
    sharepoint_resolve,
    sharepoint_upload,
)
from jake_tools.operations.errors import ToolError

from ._output import emit, tool_failure

json_option = click.option(
    "--json", "as_json", is_flag=True, help="Print the result as JSON."
)


def _run(operation: Operation, as_json: bool, **arguments: Any) -> None:
    try:
        config = load_config(os.environ)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    handler: Callable[..., Any] = operation.build(config)
    try:
        result: BaseModel = anyio.run(functools.partial(handler, **arguments))
    except ToolError as exc:
        raise tool_failure(exc, as_json) from exc
    emit(result, as_json)


@click.group()
def files() -> None:
    """Resolve, list, download, upload and replace SharePoint/Teams files.

    Acts as the one Microsoft account configured for jake-tools. Items are
    addressed by DRIVE_ID and ITEM_ID, as `resolve` and `list` print them.
    """


@files.command("resolve")
@click.option("--url", help="A SharePoint link on an allowed host.")
@click.option("--team", "team_id", help="A Teams team ID (with --channel).")
@click.option("--channel", "channel_id", help="A Teams channel ID (with --team).")
@json_option
def resolve_command(
    url: str | None, team_id: str | None, channel_id: str | None, as_json: bool
) -> None:
    """Resolve a link, or a team and channel, to a drive item."""
    _run(
        sharepoint_resolve.OPERATION,
        as_json,
        url=url,
        team_id=team_id,
        channel_id=channel_id,
    )


@files.command("get")
@click.argument("drive_id")
@click.argument("item_id")
@json_option
def get_command(drive_id: str, item_id: str, as_json: bool) -> None:
    """Show one item and its current version (etag)."""
    _run(sharepoint_get.OPERATION, as_json, drive_id=drive_id, item_id=item_id)


@files.command("list")
@click.argument("drive_id")
@click.argument("item_id")
@click.option("--limit", default=50, show_default=True, type=click.IntRange(1, 200))
@click.option("--cursor", help="The next_cursor of the previous page.")
@json_option
def list_command(
    drive_id: str, item_id: str, limit: int, cursor: str | None, as_json: bool
) -> None:
    """List one page of a folder; `complete: True` means it was the last."""
    _run(
        sharepoint_list.OPERATION,
        as_json,
        drive_id=drive_id,
        item_id=item_id,
        limit=limit,
        cursor=cursor,
    )


@files.command("download")
@click.argument("drive_id")
@click.argument("item_id")
@click.argument("destination")
@json_option
def download_command(
    drive_id: str, item_id: str, destination: str, as_json: bool
) -> None:
    """Save a file to DESTINATION, an absolute path that must not exist."""
    _run(
        sharepoint_download.OPERATION,
        as_json,
        drive_id=drive_id,
        item_id=item_id,
        destination=destination,
    )


@files.command("pdf")
@click.argument("drive_id")
@click.argument("item_id")
@click.argument("destination")
@json_option
def pdf_command(drive_id: str, item_id: str, destination: str, as_json: bool) -> None:
    """Save Microsoft's PDF rendering to DESTINATION (a new absolute path)."""
    _run(
        sharepoint_pdf.OPERATION,
        as_json,
        drive_id=drive_id,
        item_id=item_id,
        destination=destination,
    )


@files.command("upload")
@click.argument("drive_id")
@click.argument("parent_id")
@click.argument("source_path")
@click.option("--name", help="Name in SharePoint (default: the source file's name).")
@click.option(
    "--on-conflict",
    type=click.Choice(["fail", "rename"]),
    default="fail",
    show_default=True,
    help="What to do if the name exists. A file is never overwritten.",
)
@json_option
def upload_command(
    drive_id: str,
    parent_id: str,
    source_path: str,
    name: str | None,
    on_conflict: str,
    as_json: bool,
) -> None:
    """Upload SOURCE_PATH as a NEW file in folder PARENT_ID, then verify it."""
    _run(
        sharepoint_upload.OPERATION,
        as_json,
        drive_id=drive_id,
        parent_id=parent_id,
        source_path=source_path,
        name=name,
        on_conflict=on_conflict,
    )


@files.command("replace")
@click.argument("drive_id")
@click.argument("item_id")
@click.argument("source_path")
@click.option(
    "--expected-etag",
    required=True,
    help="The etag you last saw, exactly; a different current version writes nothing.",
)
@json_option
def replace_command(
    drive_id: str, item_id: str, source_path: str, expected_etag: str, as_json: bool
) -> None:
    """Replace file ITEM_ID with SOURCE_PATH only if it is still at the expected etag."""
    _run(
        sharepoint_replace.OPERATION,
        as_json,
        drive_id=drive_id,
        item_id=item_id,
        source_path=source_path,
        expected_etag=expected_etag,
    )
