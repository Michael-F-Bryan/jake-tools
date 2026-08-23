"""The ``jake-tools transcript`` command group.

Plumbing sub-commands for the meeting-transcription pipeline: each one is a
thin Click wrapper that builds its dependencies from injected options
objects (`transcript_options.py`) and prints a JSON document to stdout for
the next stage — or an agent — to chain. Orchestration logic lives in
``transcription/audio.py``, not here. Per the CLI-options memo (E19), these
commands build their dependencies from injected options objects rather than
shared context state.
"""

from __future__ import annotations

from pathlib import Path

import click

from ..transcription.adapt import AdaptError, adapt_transcript
from ..transcription.asr import TranscriberError, transcribe_merged_audio
from ..transcription.audio import (
    AudioEmbedResolutionError,
    AudioToolError,
    NoAudioEmbedsError,
    merge_note_audio,
)
from ..transcription.note import parse_note
from ..transcription.obsidian import ObsidianCliError
from .options import AgentOptions, agent_options, coro
from .transcript_options import (
    AudioOptions,
    CacheOptions,
    ObsidianOptions,
    TranscriberOptions,
    audio_options,
    cache_options,
    obsidian_options,
    transcriber_options,
)

# The cache name both `asr` and `adapt` store `RawTranscript` under: whichever
# stage produced it, downstream stages read the same key from the run cache.
_RAW_TRANSCRIPT_CACHE_NAME = "raw_transcript"


@click.group()
def transcript() -> None:
    """Plumbing sub-commands for the meeting-transcription pipeline."""


@transcript.command("merge-audio")
@obsidian_options
@audio_options
@cache_options
@click.argument(
    "note_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
def merge_audio(
    note_path: Path,
    obsidian_options: ObsidianOptions,
    audio_options: AudioOptions,
    cache_options: CacheOptions,
) -> None:
    """Merge a meeting note's audio embeds into one recording.

    Resolves every audio embed in NOTE_PATH via the vault client, orders the
    clips chronologically, merges them with ffmpeg, and prints a JSON
    document: run id, merged audio path, audio hash, and per-clip offsets.
    """
    note = parse_note(note_path)
    vault = obsidian_options.vault_client()
    audio_tool = audio_options.audio_tool()
    cache = cache_options.run_cache()

    try:
        result = merge_note_audio(note, vault=vault, audio_tool=audio_tool, cache=cache)
    except (
        NoAudioEmbedsError,
        AudioEmbedResolutionError,
        ObsidianCliError,
        AudioToolError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(result.model_dump_json(indent=2))


@transcript.command("asr")
@transcriber_options
@cache_options
@click.argument(
    "audio_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--run-id",
    "run_id",
    required=True,
    help="Run id from `transcript merge-audio` (the cache key for this run).",
)
def asr(
    audio_path: Path,
    run_id: str,
    transcriber_options: TranscriberOptions,
    cache_options: CacheOptions,
) -> None:
    """Run local ASR + diarisation over AUDIO_PATH, producing the raw transcript.

    Consults the run cache first (idempotent re-runs): a cached
    `raw_transcript.json` for RUN_ID is printed as-is, with no model
    download or HF_TOKEN needed. On a cache miss, runs parakeet-mlx ASR and
    pyannote-audio diarisation over AUDIO_PATH, aligns them into
    utterances, caches the result, and prints it. Both models run locally —
    audio never leaves this machine.
    """
    transcriber = transcriber_options.transcriber()
    cache = cache_options.run_cache()

    try:
        result = transcribe_merged_audio(
            audio_path, run_id=run_id, transcriber=transcriber, cache=cache
        )
    except TranscriberError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(result.model_dump_json(indent=2))


@transcript.command("adapt")
@agent_options
@cache_options
@click.argument(
    "transcript_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--run-id",
    "run_id",
    default=None,
    help=(
        "Run id to cache the result under (the same cache key `transcript "
        "asr` uses). Omit to skip caching and just print the JSON."
    ),
)
@coro
async def adapt(
    transcript_path: Path,
    run_id: str | None,
    agent_options: AgentOptions,
    cache_options: CacheOptions,
) -> None:
    """Adapt a pre-diarised text transcript (Gemini/Teams export) into the raw transcript.

    Some meetings arrive as text rather than audio — a Gemini-produced
    transcript document from Google Meet, or a Teams `.vtt`/plain-text
    export. TRANSCRIPT_PATH is that file. Tries deterministic parsers first
    (WebVTT cue tags, then `Name: text` / `Name (00:12:34): text` plain
    text); if neither matches, falls back to an LLM call that restructures
    the document into utterances without rewriting its words. Prints the
    resulting `RawTranscript` JSON to stdout, and — like `transcript asr` —
    stores it as `raw_transcript.json` in the run cache when `--run-id` is
    given, so downstream stages find it under the same key regardless of
    whether it came from audio or text.
    """
    agent = agent_options.agent()
    cache = cache_options.run_cache()

    try:
        result = await adapt_transcript(transcript_path, agent=agent)
    except AdaptError as exc:
        raise click.ClickException(str(exc)) from exc

    if run_id is not None:
        cache.store(run_id, _RAW_TRANSCRIPT_CACHE_NAME, result)

    click.echo(result.model_dump_json(indent=2))
