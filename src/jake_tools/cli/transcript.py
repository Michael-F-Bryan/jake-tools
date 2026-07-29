from __future__ import annotations

import json
from pathlib import Path

import click

from ..claude import ClaudeAgent
from ..transcripts.recipe_primitives import (
    RecipePrimitiveError,
    run_teams_meeting_recipe,
    run_youtube_source_notes_recipe,
)
from .options import agent, coro


@click.group(help="Turn recorded sources into verified notes.")
def transcript() -> None:
    pass


@transcript.command("teams-meeting")
@click.option(
    "--account",
    default="csu-teams",
    show_default=True,
    help="Configured Teams/Graph account to use.",
)
@click.option(
    "--profile",
    type=click.Choice(["default", "dumc"], case_sensitive=True),
    default="dumc",
    show_default=True,
    help="Note shape and verification profile.",
)
@click.option(
    "--token-file",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="OAuth token JSON file. Defaults from --account.",
)
@click.option("--event-id", help="Exact Outlook calendar event ID to import.")
@click.option(
    "--days-back",
    default=14,
    show_default=True,
    type=click.IntRange(min=1),
    help="Calendar search window when --event-id is omitted.",
)
@click.option(
    "--query",
    help="Case-insensitive event subject filter for recent calendar search.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for source data, transcript artefacts, and the rendered note.",
)
@click.option(
    "--vault-note",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Obsidian note to write after verification.",
)
@click.option(
    "--write-vault",
    is_flag=True,
    help="Write to the default DUM-C vault destination.",
)
@click.option("--dry-run", is_flag=True, help="Do not write the vault note.")
@click.option("--json", "as_json", is_flag=True, help="Emit a JSON result.")
def teams_meeting(
    account: str,
    profile: str,
    token_file: Path | None,
    event_id: str | None,
    days_back: int,
    query: str | None,
    out_dir: Path,
    vault_note: Path | None,
    write_vault: bool,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Import a Teams transcript and produce a verified meeting note."""
    if write_vault and vault_note is not None:
        raise click.UsageError("Use either --vault-note or --write-vault, not both.")
    if write_vault and profile != "dumc":
        raise click.UsageError("--write-vault requires --profile dumc.")

    try:
        result = run_teams_meeting_recipe(
            account=account,
            profile=profile,
            out_dir=out_dir,
            token_file=token_file,
            event_id=event_id,
            days_back=days_back,
            query=query,
            vault_note=vault_note,
            write_vault=write_vault,
            dry_run=dry_run,
        )
    except RecipePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result, sort_keys=True))
        return
    click.echo(f"rendered_note: {result['rendered_note']}")
    click.echo(f"note: {result['note']}")
    click.echo(f"updated: {result['updated']}")


@transcript.command("youtube")
@agent
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory for captions, transcript artefacts, and the rendered note.",
)
@click.option(
    "--language",
    default="en",
    show_default=True,
    help="Preferred caption language code.",
)
@click.option(
    "--vault-note",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Obsidian note to write after verification.",
)
@click.option("--dry-run", is_flag=True, help="Do not write the vault note.")
@click.option("--json", "as_json", is_flag=True, help="Emit a JSON result.")
@click.argument("url")
@coro
async def youtube(
    agent: ClaudeAgent,
    out_dir: Path,
    language: str,
    vault_note: Path | None,
    dry_run: bool,
    as_json: bool,
    url: str,
) -> None:
    """Fetch YouTube captions and produce a verified, chapterised source note."""
    try:
        result = await run_youtube_source_notes_recipe(
            agent,
            url,
            out_dir=out_dir,
            language=language,
            vault_note=vault_note,
            dry_run=dry_run,
        )
    except RecipePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result, sort_keys=True))
        return
    click.echo(f"rendered_note: {result['rendered_note']}")
    click.echo(f"subtitle: {result['subtitle_track']} ({result['subtitle_kind']})")
    click.echo(f"note: {result['note']}")
    click.echo(f"updated: {result['updated']}")
