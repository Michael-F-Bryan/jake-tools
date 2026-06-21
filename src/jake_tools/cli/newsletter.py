from __future__ import annotations

import json
from pathlib import Path
import sys

import click

from ..newsletters import NewsletterAttachment, NewsletterClient, NewsletterError, NewsletterItem


@click.group()
def newsletter():
    """
    Read and update the CSU Weekly Newsletter list.
    """
    pass


@newsletter.command("list")
@click.option("--limit", default=10, show_default=True, type=click.IntRange(1, 100))
@click.option("--body", is_flag=True, help="Include the plaintext body for each item.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def list_items(limit: int, body: bool, as_json: bool):
    """
    Show recent CSU Weekly Newsletter items.
    """
    client = NewsletterClient()
    try:
        items = client.list_items(limit=limit)
    except NewsletterError as exc:
        raise click.ClickException(str(exc)) from exc

    _emit_items(items, include_body=body, as_json=as_json)


@newsletter.command()
@click.argument("title")
@click.option(
    "--attach",
    "attachments",
    multiple=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Attach a file to the newsletter item. Can be supplied multiple times.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def add(title: str, attachments: tuple[Path, ...], as_json: bool):
    """
    Create a CSU Weekly Newsletter item.

    The body is read from stdin. Example:

        jake-tools newsletter add "Training night update" < /tmp/newsletter-item.txt
    """
    body = _read_stdin_body(required=True)
    if body is None:
        raise click.UsageError("newsletter body is required on stdin")

    client = NewsletterClient()
    try:
        item = client.create_item(
            title=title,
            body=body,
            attachments=_attachments(attachments),
        )
    except NewsletterError as exc:
        raise click.ClickException(str(exc)) from exc

    _emit_item(item, include_body=True, as_json=as_json)


@newsletter.command()
@click.argument("item_id")
@click.option("--title", help="Replace the newsletter item title.")
@click.option(
    "--attach",
    "attachments",
    multiple=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    help="Attach a file to the newsletter item. Can be supplied multiple times.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def edit(
    item_id: str,
    title: str | None,
    attachments: tuple[Path, ...],
    as_json: bool,
):
    """
    Update an existing CSU Weekly Newsletter item.

    The new body is read from stdin when provided. Use --title to update the title,
    and --attach to add attachments without replacing existing attachments.
    """
    body = _read_stdin_body(required=False)
    if title is None and body is None and not attachments:
        raise click.UsageError("nothing to update: provide --title, stdin body, or --attach")

    client = NewsletterClient()
    try:
        item = client.update_item(
            item_id,
            title=title,
            body=body,
            attachments=_attachments(attachments),
        )
    except NewsletterError as exc:
        raise click.ClickException(str(exc)) from exc

    _emit_item(item, include_body=True, as_json=as_json)


def _attachments(paths: tuple[Path, ...]) -> list[NewsletterAttachment]:
    return [NewsletterAttachment(path) for path in paths]


def _read_stdin_body(*, required: bool) -> str | None:
    body = sys.stdin.read()
    if body.strip():
        return body
    if required:
        raise click.UsageError("newsletter body is required on stdin")
    return None


def _emit_items(items: list[NewsletterItem], *, include_body: bool, as_json: bool) -> None:
    if as_json:
        payload = [item.as_dict() if include_body else _item_summary(item) for item in items]
        click.echo(json.dumps(payload, indent=2))
        return

    for item in items:
        _emit_item(item, include_body=include_body, as_json=False)
        click.echo("---")


def _emit_item(item: NewsletterItem, *, include_body: bool, as_json: bool) -> None:
    if as_json:
        payload = item.as_dict() if include_body else _item_summary(item)
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"ID: {item.id}")
    click.echo(f"Title: {item.title}")
    click.echo(f"Created: {item.created}")
    click.echo(f"Modified: {item.modified}")
    click.echo(f"URL: {item.url}")
    if include_body:
        click.echo()
        click.echo(item.body)


def _item_summary(item: NewsletterItem) -> dict[str, str]:
    return {
        "id": item.id,
        "title": item.title,
        "created": item.created,
        "modified": item.modified,
        "url": item.url,
    }
