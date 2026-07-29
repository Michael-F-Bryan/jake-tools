from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import click

from ..claude import ClaudeAgent
from ..transcripts.errors import TranscriptError
from ..transcripts.models import CoordinatorResult
from ..transcripts.obsidian_recipe import run_obsidian_recording_recipe
from ..transcripts.polish import polish_transcript
from ..transcripts.render import MeetingNoteProfile
from ..transcripts.teams_recipe import run_teams_meeting_recipe
from ..transcripts.youtube_recipe import run_youtube_source_notes_recipe
from .options import agent, coro

# Personal policy for the CSU/DUM-C Teams account: which token file backs it,
# and what provenance to stamp onto the resulting SourceArtifact. The library
# layer (teams_graph.py, teams_recipe.py) takes these as plain parameters; this
# is the one place that knows what "csu-teams" means.
_TEAMS_ACCOUNT_TOKEN_FILES: dict[str, Path] = {
    "csu-teams": Path("~/.hermes/csu-teams-graph-token.json").expanduser(),
}
_TEAMS_ACCOUNT_PROVENANCE: dict[str, tuple[str, str]] = {
    "csu-teams": ("CSU", "DUM-C"),
}
_DEFAULT_DUMC_VAULT_DIR = Path("~/Documents/Vault/2 Areas/DUM-C").expanduser()


def _resolve_teams_account(
    account: str, token_file: Path | None
) -> tuple[Path, str | None, str | None]:
    default_token_file = _TEAMS_ACCOUNT_TOKEN_FILES.get(account)
    if token_file is None and default_token_file is None:
        raise click.UsageError(
            f"unknown --account {account!r}; pass --token-file explicitly or use "
            f"one of: {', '.join(sorted(_TEAMS_ACCOUNT_TOKEN_FILES))}"
        )
    organisation, project = _TEAMS_ACCOUNT_PROVENANCE.get(account, (None, None))
    return token_file or cast(Path, default_token_file), organisation, project


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
    """Import a Teams transcript and produce a verified meeting note.

    The rendered note is written to --vault-note (or the default DUM-C vault
    destination with --write-vault) only after note verification passes.
    --dry-run runs the full pipeline and writes every intermediate artefact
    under --out-dir, but skips the final vault write.
    """
    if write_vault and vault_note is not None:
        raise click.UsageError("Use either --vault-note or --write-vault, not both.")
    if write_vault and profile != "dumc":
        raise click.UsageError("--write-vault requires --profile dumc.")

    resolved_token_file, organisation, project = _resolve_teams_account(
        account, token_file
    )

    try:
        result = run_teams_meeting_recipe(
            account=account,
            profile=cast(MeetingNoteProfile, profile),
            out_dir=out_dir,
            token_file=resolved_token_file,
            event_id=event_id,
            days_back=days_back,
            query=query,
            organisation=organisation,
            project=project,
            vault_note=vault_note,
            write_vault=write_vault,
            dumc_vault_dir=_DEFAULT_DUMC_VAULT_DIR,
            dry_run=dry_run,
        )
    except TranscriptError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        return
    click.echo(f"rendered_note: {result.rendered_note}")
    click.echo(f"note: {result.note}")
    click.echo(f"updated: {result.updated}")


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
    """Fetch YouTube captions and produce a verified, chapterised source note.

    Prefers authored captions in --language, then falls back to automatic
    captions. --vault-note is written only after note verification passes;
    --dry-run runs the full pipeline (captions, transcript, rendered note,
    manifest under --out-dir) but skips that final write.
    """
    try:
        result = await run_youtube_source_notes_recipe(
            agent,
            url,
            out_dir=out_dir,
            language=language,
            vault_note=vault_note,
            dry_run=dry_run,
        )
    except TranscriptError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result.model_dump(mode="json"), sort_keys=True))
        return
    click.echo(f"rendered_note: {result.rendered_note}")
    click.echo(f"subtitle: {result.subtitle_track} ({result.subtitle_kind})")
    click.echo(f"note: {result.note}")
    click.echo(f"updated: {result.updated}")


def _emit_obsidian_recording_result(
    result: CoordinatorResult, *, as_json: bool
) -> None:
    if as_json:
        click.echo(json.dumps(result.json_summary(), indent=2))
        return

    click.echo(f"note: {result.note_path}")
    click.echo(f"updated: {result.updated}")


@transcript.command("obsidian-recording")
@agent
@click.option(
    "--work-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help=(
        "Directory to keep intermediate artefacts (audio, transcripts, "
        "chapters, a manifest) in. Without it, artefacts live in a temporary "
        "directory that is gone once the command returns."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run the pipeline without writing the updated note back to disk.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit a machine-readable JSON summary.",
)
@click.argument(
    "obsidian_note",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@coro
async def obsidian_recording(
    agent: ClaudeAgent,
    work_dir: Path | None,
    dry_run: bool,
    as_json: bool,
    obsidian_note: Path,
) -> None:
    """Process an Obsidian recording into a polished, chapterised note.

    Requires local `ffmpeg` and `scribe` executables. Rewrites OBSIDIAN_NOTE
    in place only after note verification passes; --dry-run runs the full
    pipeline (transcription, speaker mapping, polish, chaptering, minutes)
    without that final write.
    """
    try:
        result = await run_obsidian_recording_recipe(
            agent,
            obsidian_note,
            dry_run=dry_run,
            workdir=work_dir,
        )
    except TranscriptError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_obsidian_recording_result(result, as_json=as_json)


@transcript.command("polish")
@agent
@click.argument(
    "transcript_file", required=True, type=click.File("r", encoding="utf-8")
)
@coro
async def polish(agent: ClaudeAgent, transcript_file) -> None:
    """Polish a raw transcript file and print the result to stdout."""
    raw = transcript_file.read()
    polished = await polish_transcript(agent, raw)
    click.echo(polished)
