"""The ``jake-tools transcript`` command group.

Plumbing sub-commands for the meeting-transcription pipeline: each one is a
thin Click wrapper that resolves inputs through :class:`~.context.AppContext`
and prints a JSON document to stdout for the next stage — or an agent — to
chain. Orchestration logic lives in ``transcription/audio.py``, not here.
"""

from __future__ import annotations

from pathlib import Path

import click

from ..transcription.audio import AudioToolError, NoAudioEmbedsError, merge_note_audio
from ..transcription.note import parse_note
from ..transcription.obsidian import ObsidianCliError
from .context import app_context


@click.group()
def transcript() -> None:
    """Plumbing sub-commands for the meeting-transcription pipeline."""


@transcript.command("merge-audio")
@click.argument(
    "note_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.pass_context
def merge_audio(ctx: click.Context, note_path: Path) -> None:
    """Merge a meeting note's audio embeds into one recording.

    Resolves every audio embed in NOTE_PATH via the vault client, orders the
    clips chronologically, merges them with ffmpeg, and prints a JSON
    document: run id, merged audio path, audio hash, and per-clip offsets.
    """
    context = app_context(ctx)
    note = parse_note(note_path)
    vault = context.vault_client_factory()
    audio_tool = context.audio_tool_factory()
    cache = context.run_cache_factory()

    try:
        result = merge_note_audio(note, vault=vault, audio_tool=audio_tool, cache=cache)
    except (NoAudioEmbedsError, ObsidianCliError, AudioToolError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(result.model_dump_json(indent=2))
