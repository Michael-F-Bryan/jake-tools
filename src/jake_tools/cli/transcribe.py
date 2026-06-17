from pathlib import Path
from typing import IO

import click

from ..hermes import Hermes
from ..transcripts import process_obsidian_recording
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
def polish(hermes: Hermes, transcript: IO[str]):
    """
    Polish a transcript.
    """
    raw = transcript.read()
    polished = polish_transcript(hermes, raw)
    click.echo(polished)


@transcribe.command()
@hermes
@click.argument(
    "obsidian-note",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def obsidian_recording(hermes: Hermes, obsidian_note: Path):
    process_obsidian_recording(hermes, obsidian_note)
