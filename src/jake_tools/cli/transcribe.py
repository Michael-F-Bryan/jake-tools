import json
from pathlib import Path

import click

from ..claude import ClaudeAgent
from ..transcripts.models import CoordinatorResult
from ..transcripts.polish import polish_transcript
from ..transcripts.recipe_primitives import (
    RecipePrimitiveError,
    run_obsidian_recording_recipe,
)
from .options import agent, coro


@click.group
def transcribe():
    """
    Tools for transcribing audio files.
    """
    pass


@transcribe.command
@agent
@click.argument("transcript", required=True, type=click.File("r", encoding="utf-8"))
@coro
async def polish(agent: ClaudeAgent, transcript):
    """
    Polish a transcript.
    """
    raw = transcript.read()
    polished = await polish_transcript(agent, raw)
    click.echo(polished)


def _emit_obsidian_recording_result(
    result: CoordinatorResult, *, as_json: bool
) -> None:
    if as_json:
        click.echo(json.dumps(result.json_summary(), indent=2))
        return

    click.echo(f"note: {result.note_path}")
    click.echo(f"updated: {result.updated}")


@transcribe.command()
@agent
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
    dry_run: bool,
    as_json: bool,
    obsidian_note: Path,
):
    """
    Process an Obsidian recording into a polished, chapterised note.
    """
    try:
        result = await run_obsidian_recording_recipe(
            agent,
            obsidian_note,
            dry_run=dry_run,
        )
    except RecipePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_obsidian_recording_result(result, as_json=as_json)
