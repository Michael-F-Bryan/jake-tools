from __future__ import annotations

import json

import click

from ..clockify import (
    CLOCKIFY_API_ROOT,
    ClockifyClient,
    ClockifyError,
    ClockifyUser,
    clockify_api_key_from_env,
    clockify_base_url_from_env,
)


@click.group()
@click.option(
    "--api-key",
    envvar="CLOCKIFY_API_KEY",
    help="Clockify API key. Defaults to CLOCKIFY_API_KEY.",
)
@click.option(
    "--api-base-url",
    default=None,
    help=f"Clockify API base URL. Defaults to CLOCKIFY_API_BASE_URL or {CLOCKIFY_API_ROOT}.",
)
@click.pass_context
def clockify(ctx: click.Context, api_key: str | None, api_base_url: str | None) -> None:
    """Work with Clockify time-tracking data."""
    ctx.obj = {
        "api_key": api_key,
        "api_base_url": api_base_url,
    }


@clockify.command()
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def whoami(ctx: click.Context, as_json: bool) -> None:
    """Show the Clockify user for the configured API key."""
    try:
        client = _client_from_context(ctx)
        user = client.get_user()
    except ClockifyError as exc:
        raise click.ClickException(str(exc)) from exc

    _emit_user(user, as_json=as_json)


def _client_from_context(ctx: click.Context) -> ClockifyClient:
    config = ctx.obj or {}
    api_key = config.get("api_key") or clockify_api_key_from_env()
    base_url = config.get("api_base_url") or clockify_base_url_from_env()
    return ClockifyClient(api_key=api_key, base_url=base_url)


def _emit_user(user: ClockifyUser, *, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(user.model_dump(mode="json"), indent=2))
        return

    click.echo(f"ID: {user.id}")
    click.echo(f"Name: {user.name}")
    click.echo(f"Email: {user.email}")
    click.echo(f"Active workspace: {user.active_workspace}")
    click.echo(f"Default workspace: {user.default_workspace}")
