from pathlib import Path
import json

import click

from ..hermes import Hermes
from ..transcripts.coordinator import Mode, process_obsidian_recording
from ..transcripts.polish import polish_transcript
from .options import hermes


@click.group
def transcribe():
    """
    Tools for transcribing audio files.
    """
    pass


@transcribe.command
@hermes
@click.argument("transcript", required=True, type=click.File("r", encoding="utf-8"))
def polish(hermes: Hermes, transcript):
    """
    Polish a transcript.
    """
    raw = transcript.read()
    polished = polish_transcript(hermes, raw)
    click.echo(polished)


@transcribe.command()
@hermes
@click.option(
    "--mode",
    type=click.Choice(["transcript", "chaptered-transcript", "minutes"], case_sensitive=False),
    default="minutes",
    show_default=True,
    help="Output mode for the note update.",
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
def obsidian_recording(
    hermes: Hermes,
    mode: str,
    dry_run: bool,
    as_json: bool,
    obsidian_note: Path,
):
    result = process_obsidian_recording(
        hermes,
        obsidian_note,
        mode=mode,  # type: ignore[arg-type]
        dry_run=dry_run,
    )

    if as_json:
        click.echo(json.dumps(result.model_dump(mode="json"), indent=2))
        return

    click.echo(f"mode: {result.mode}")
    click.echo(f"note: {result.note_path}")
    click.echo(f"updated: {result.updated}")
