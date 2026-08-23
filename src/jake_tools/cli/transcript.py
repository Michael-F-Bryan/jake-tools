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

from ..claude import ClaudeAgentError
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
from ..transcription.speakers import SpeakersError, run_speaker_resolution
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

# `transcript speakers` exits with this code (never 0 or the generic
# ClickException code 1) when unresolved clusters need a human answer.
# Stable contract: plan 011's porcelain and any coordinating agent branch on
# it to know a Discord relay is needed before re-running with `--assign`.
NEEDS_INPUT_EXIT_CODE = 3


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


@transcript.command("speakers")
@agent_options
@audio_options
@cache_options
@click.argument(
    "note_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--run-id",
    "run_id",
    required=True,
    help="Run id from `transcript merge-audio` (the cache key for this run).",
)
@click.option(
    "--assign",
    "assign",
    multiple=True,
    metavar="CLUSTER=NAME",
    help=(
        "Confirm one cluster's speaker, e.g. --assign \"SPEAKER_03=Nikki "
        'Staltari" (repeatable). Assignments are ground truth: merged into '
        "the run cache's assignments.json and never re-litigated by the "
        'resolver, in this run or a later one. --assign "SPEAKER_04=Unknown" '
        "records that a cluster could not be identified."
    ),
)
@click.option(
    "--finalise",
    "finalise",
    is_flag=True,
    help=(
        "Map any clusters still unresolved after --assign and the LLM's "
        'proposals to "Unknown", instead of asking for more input.'
    ),
)
@coro
async def speakers(
    note_path: Path,
    run_id: str,
    assign: tuple[str, ...],
    finalise: bool,
    agent_options: AgentOptions,
    audio_options: AudioOptions,
    cache_options: CacheOptions,
) -> None:
    """Resolve diarisation clusters (SPEAKER_NN) to attendee names.

    Reads the cached raw transcript for RUN_ID (from `transcript asr` or
    `transcript adapt --run-id`), applies any --assign overrides, and asks
    an LLM to propose names for every other cluster from NOTE_PATH's
    Attendees list and Meeting Prep section (hints in Michael's own
    phrasing, wherever they appear in that section). Only high-confidence
    proposals are accepted, plus medium-confidence ones a hint corroborates
    - never a guess.

    A cluster that stays unresolved gets 2-3 short audio snippets cut from
    the merged recording. If any remain and --finalise was not given, this
    prints `{"status": "needs_input", "requests": [...]}` and exits with
    code 3 (a stable contract: relay the requests - e.g. over Discord - and
    re-run with more --assign flags, or with --finalise to give up and
    label the rest "Unknown").

    Once every cluster is resolved, the result is cached as
    resolved_transcript.json, newly confirmed names are appended as hint
    bullets under Meeting Prep's "Diarisation hints:" list - the only
    sanctioned write into that human-owned section - and this prints
    `{"status": "resolved", ...}` and exits 0.
    """
    agent = agent_options.agent()
    audio_tool = audio_options.audio_tool()
    cache = cache_options.run_cache()

    try:
        result = await run_speaker_resolution(
            note_path,
            run_id,
            assign=assign,
            finalise=finalise,
            agent=agent,
            audio_tool=audio_tool,
            cache=cache,
        )
    except (SpeakersError, ClaudeAgentError, AudioToolError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(result.model_dump_json(indent=2))
    if result.status == "needs_input":
        raise click.exceptions.Exit(NEEDS_INPUT_EXIT_CODE)
