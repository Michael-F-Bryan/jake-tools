"""Printing a tool result or failure from a shell, shared by the thin commands
that call an operation (``artifact``, ``files``)."""

from __future__ import annotations

import click
from pydantic import BaseModel

from jake_tools.operations.errors import ToolError


def emit(result: BaseModel, as_json: bool) -> None:
    if as_json:
        click.echo(result.model_dump_json(indent=2, exclude_none=True))
        return
    data = result.model_dump(mode="json", exclude_none=True)
    for key, value in data.items():
        _line(key, value, 0)


def _line(key: str, value: object, depth: int) -> None:
    pad = "  " * depth
    if isinstance(value, dict):
        click.echo(f"{pad}{key}:")
        for inner_key, inner in value.items():
            _line(inner_key, inner, depth + 1)
    elif isinstance(value, list) and any(isinstance(v, dict) for v in value):
        click.echo(f"{pad}{key}:")
        for index, inner in enumerate(value):
            _line(f"[{index}]", inner, depth + 1)
    else:
        click.echo(f"{pad}{key}: {value}")


def tool_failure(exc: ToolError, as_json: bool) -> click.ClickException:
    """The error with everything a caller needs to act on an uncertain write.

    A ``ToolError`` carries structured detail (``outcome``, ``retry_safe``,
    ``next_step``, the item seen): dropping it would hide that a write may have
    happened. ``--json`` prints the full payload to stderr."""
    payload = exc.payload
    if as_json:
        return click.ClickException(
            payload.model_dump_json(indent=2, exclude_none=True)
        )
    lines = [f"{exc.code}: {exc.message}"]
    for key, value in (exc.detail or {}).items():
        lines.append(f"  {key}: {value}")
    return click.ClickException("\n".join(lines))
