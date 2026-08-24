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

from ..ai_usage import AITelemetry, AITotals
from ..claude import ClaudeAgent, ClaudeAgentError
from ..transcription.adapt import (
    AdaptedTranscript,
    AdaptError,
    AdaptTranscriptPrompt,
    adapt_transcript,
)
from ..transcription.asr import TranscriberError, transcribe_merged_audio
from ..transcription.audio import (
    AudioEmbedResolutionError,
    AudioToolError,
    NoAudioEmbedsError,
    merge_note_audio,
)
from ..transcription.cache import stable_hash
from ..transcription.chapters import (
    CHAPTERS_ADAPTER,
    ChaptersError,
    run_chapterisation,
)
from ..transcription.integrate import IntegrateError, load_products, run_integrate
from ..transcription.minutes import MinutesError, run_minutes
from ..transcription.models import RawTranscript
from ..transcription.note import NoteParseError, parse_note
from ..transcription.obsidian import ObsidianCliError
from ..transcription.polish import PolishError, run_polish
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


@transcript.command("telemetry")
@cache_options
@click.option(
    "--run-id",
    "run_id",
    required=True,
    help="Run id whose durable AI telemetry should be inspected.",
)
def telemetry(run_id: str, cache_options: CacheOptions) -> None:
    """Print durable per-stage LLM usage and API-rate-equivalent totals."""
    cache = cache_options.run_cache()
    value = cache.load(run_id, "ai_telemetry", AITelemetry) or AITelemetry(
        calls=[], stages=[], totals=AITotals()
    )
    click.echo(value.model_dump_json(indent=2))


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
    vault = obsidian_options.vault_client()
    audio_tool = audio_options.audio_tool()
    cache = cache_options.run_cache()

    try:
        note = parse_note(note_path)
        result = merge_note_audio(note, vault=vault, audio_tool=audio_tool, cache=cache)
    except (
        NoAudioEmbedsError,
        AudioEmbedResolutionError,
        ObsidianCliError,
        AudioToolError,
        NoteParseError,
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
        if transcriber_options.num_speakers is None:
            result = transcribe_merged_audio(
                audio_path, run_id=run_id, transcriber=transcriber, cache=cache
            )
        else:
            result = transcribe_merged_audio(
                audio_path,
                run_id=run_id,
                transcriber=transcriber,
                cache=cache,
                num_speakers=transcriber_options.num_speakers,
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
        if run_id is None:
            result = await adapt_transcript(transcript_path, agent=agent)
        else:
            document = transcript_path.read_text(encoding="utf-8")
            stage_agent = agent.for_stage("adapt").with_telemetry(
                cache.telemetry_sink(run_id)
            )
            manifest = cache.stage_manifest(
                "adapt",
                inputs={"source_text": document},
                input_hashes={"source_text": stable_hash(document)},
                config={
                    "agent": stage_agent.defaults.model_dump(mode="json"),
                    "prompt": AdaptTranscriptPrompt.template,
                    "response_schema": AdaptedTranscript.model_json_schema(),
                },
            )
            cached = cache.load_resumable(
                run_id, _RAW_TRANSCRIPT_CACHE_NAME, RawTranscript
            )
            current_manifest = cache.load_manifest(run_id, "adapt")
            if (
                cached is not None
                and cache.manifest_matches(current_manifest, manifest)
                and current_manifest is not None
                and current_manifest.output_hash
                == stable_hash(cached.model_dump(mode="json"))
            ):
                stage_agent.record_cache_hit()
                result = cached
            else:
                if cached is not None or current_manifest is not None:
                    cache.invalidate_artefacts(
                        run_id, {"raw_transcript", "adapt.manifest"}
                    )
                    cache.invalidate_downstream(
                        run_id, reason="adapt input or configuration changed"
                    )
                result = await adapt_transcript(transcript_path, agent=stage_agent)
                cache.store(run_id, _RAW_TRANSCRIPT_CACHE_NAME, result)
                cache.store_manifest(
                    run_id,
                    manifest.model_copy(
                        update={
                            "output_hash": stable_hash(result.model_dump(mode="json"))
                        }
                    ),
                )
    except (AdaptError, ClaudeAgentError) as exc:
        raise click.ClickException(str(exc)) from exc

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
    "--correct",
    "correct",
    multiple=True,
    metavar="CLUSTER=START-END=NAME",
    help=(
        "Relabel only utterances fully contained in a half-open time range; "
        "repeatable, e.g. --correct SPEAKER_00=12.0-18.5=Ada Lovelace."
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
    correct: tuple[str, ...],
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
        if correct:
            result = await run_speaker_resolution(
                note_path,
                run_id,
                assign=assign,
                correct=correct,
                finalise=finalise,
                agent=agent,
                audio_tool=audio_tool,
                cache=cache,
            )
        else:
            result = await run_speaker_resolution(
                note_path,
                run_id,
                assign=assign,
                finalise=finalise,
                agent=agent,
                audio_tool=audio_tool,
                cache=cache,
            )
    except (SpeakersError, ClaudeAgentError, AudioToolError, NoteParseError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(result.model_dump_json(indent=2))
    if result.status == "needs_input":
        raise click.exceptions.Exit(NEEDS_INPUT_EXIT_CODE)


@transcript.command("chapterise")
@agent_options
@cache_options
@click.option(
    "--run-id",
    "run_id",
    required=True,
    help="Run id from `transcript merge-audio` (the cache key for this run).",
)
@coro
async def chapterise(
    run_id: str,
    agent_options: AgentOptions,
    cache_options: CacheOptions,
) -> None:
    """Chapterise the resolved transcript into topic-based spans.

    Reads the cached resolved transcript for RUN_ID (from `transcript
    speakers`; errors naming that prerequisite if it hasn't run yet), asks
    an LLM for chapter boundaries over a compact index/speaker/text
    rendering of the raw dialogue (chapter boundaries need topic flow, not
    clean prose, so the un-polished text is sufficient), and
    deterministically repairs the boundaries in code - sorted, clamped,
    deduplicated - so every utterance ends up in exactly one chapter,
    never trusting that arithmetic to the model. Stores the result as
    `chapters.json` and prints it to stdout as a JSON array.

    This is a cheap-model stage: unless you pass `--effort` explicitly,
    this command layers `effort="low"` onto the agent spec built from
    `--model`/`--effort` (`AgentOptions.spec()`), rather than changing the
    `agent_options` decorator's own default.
    """
    spec = agent_options.spec()
    if agent_options.effort is None:
        # `--effort` was not passed, so `AgentOptions.effort` is still its
        # unset default (None) - the one case this stage's own "low"
        # default should apply. An explicit `--effort` (including a
        # user-chosen "low") is captured above and left alone.
        spec = spec.model_copy(update={"effort": "low"})
    agent = ClaudeAgent(defaults=spec)
    cache = cache_options.run_cache()

    try:
        chapters = await run_chapterisation(run_id, agent=agent, cache=cache)
    except (ChaptersError, ClaudeAgentError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(CHAPTERS_ADAPTER.dump_json(chapters, indent=2).decode())


@transcript.command("polish")
@agent_options
@obsidian_options
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
    "--chapter",
    "chapter",
    type=int,
    default=None,
    help=(
        "Re-polish a single chapter by its 0-based index into chapters.json, "
        "merging the result into the existing polished.json/polish_issues.json "
        "(for an agent-driven retry of one bad chapter after a full run). "
        "Omit to polish every chapter."
    ),
)
@click.option(
    "--max-concurrency",
    "max_concurrency",
    type=int,
    default=4,
    show_default=True,
    help="Maximum number of chapters polished concurrently.",
)
@coro
async def polish(
    note_path: Path,
    run_id: str,
    chapter: int | None,
    max_concurrency: int,
    agent_options: AgentOptions,
    obsidian_options: ObsidianOptions,
    cache_options: CacheOptions,
) -> None:
    """Polish each chapter's raw dialogue, then check it with a fresh adversarial pass.

    Reads the cached resolved transcript and chapters for RUN_ID (from
    `transcript speakers` and `transcript chapterise`; errors naming
    whichever prerequisite hasn't run yet), builds a lexicon of
    meeting-specific vocabulary from NOTE_PATH's attendees, wikilinks, and
    vault note titles, and polishes every chapter concurrently (bounded by
    --max-concurrency).

    Each chapter gets two separate LLM calls: one that polishes the raw
    dialogue (removing filler, merging turns, untangling crosstalk,
    correcting mishearings against the lexicon), and a second, independent
    call that reviews that polish adversarially against the raw utterances
    and returns a corrected chapter - never a "check your work" turn on the
    same conversation. Meaning preservation is the invariant: form is
    rewritten, content never is.

    Stores polished.json (list[PolishedChapter]) and polish_issues.json
    (every fixer issue, prefixed with its chapter's title) in the run
    cache, and prints a JSON summary (chapter count, issue count, issues)
    to stdout. One chapter's failure fails the whole run, naming the
    chapter, rather than silently producing an incomplete result.

    --chapter N re-polishes just that one chapter and merges it into an
    existing polished.json - for retrying a single bad chapter without
    re-running (and re-paying for) the whole meeting.
    """
    agent = agent_options.agent()
    vault = obsidian_options.vault_client()
    cache = cache_options.run_cache()

    try:
        result = await run_polish(
            note_path,
            run_id,
            chapter=chapter,
            max_concurrency=max_concurrency,
            agent=agent,
            vault=vault,
            cache=cache,
        )
    except (PolishError, ClaudeAgentError, NoteParseError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(result.model_dump_json(indent=2))


@transcript.command("minutes")
@agent_options
@obsidian_options
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
@coro
async def minutes(
    note_path: Path,
    run_id: str,
    agent_options: AgentOptions,
    obsidian_options: ObsidianOptions,
    cache_options: CacheOptions,
) -> None:
    """Generate the meeting summary and Discussion Notes minutes.

    Reads the cached polished chapters for RUN_ID (from `transcript polish`;
    errors naming that prerequisite if it hasn't run yet), builds the same
    vault lexicon `transcript polish` builds from NOTE_PATH (attendees,
    wikilinks already in the note, vault note titles) as the wikilink
    candidate set, and asks an LLM for the meeting summary and Discussion
    Notes in one call over every polished chapter.

    The minutes report what was said - facts, positions, decisions, and
    open questions - and never prescribe what should happen next: an
    action item only appears when someone actually took it on in the
    meeting, attributed as they said it. Stores the result as
    minutes.json in the run cache and prints it as JSON to stdout.
    """
    agent = agent_options.agent()
    vault = obsidian_options.vault_client()
    cache = cache_options.run_cache()

    try:
        result = await run_minutes(
            note_path, run_id, agent=agent, vault=vault, cache=cache
        )
    except (MinutesError, ClaudeAgentError, NoteParseError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(result.model_dump_json(indent=2))


@transcript.command("integrate")
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
def integrate(
    note_path: Path,
    run_id: str,
    cache_options: CacheOptions,
) -> None:
    """Write this run's products into NOTE_PATH's tier-owned sections.

    Reads the cached polished chapters and minutes for RUN_ID (from
    `transcript polish` and `transcript minutes`; errors naming whichever
    prerequisite hasn't run yet), then writes them into the note: `##
    Chapters` and `## Transcript` are replaced wholesale (pipeline-owned),
    while the `> [!summary]` callout and `## Discussion Notes` go through a
    three-way textual merge against this run id's cached baseline - never
    an LLM - so a human edit is preserved verbatim rather than clobbered.
    Frontmatter and `## Meeting Prep` are never touched here.

    Makes no LLM calls. Prints the resulting `IntegrationReport` (sections
    created/replaced/merged/appended, human-preserved unit counts) as
    JSON.
    """
    cache = cache_options.run_cache()

    try:
        products = load_products(note_path, run_id, cache)
        report = run_integrate(note_path, products, cache=cache, run_id=run_id)
    except (IntegrateError, NoteParseError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(report.model_dump_json(indent=2))
