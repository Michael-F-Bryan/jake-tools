"""``jake-tools transcribe`` - the composed transcription porcelain.

One command, the full product set: merges/transcribes (or adapts), resolves
speakers, chapterises, polishes, generates minutes, and writes the note - or
pauses non-interactively with the same `needs_input` contract `transcript
speakers` uses, for a coordinating agent to relay (e.g. over Discord) and
resume. All composition logic lives in `transcription/pipeline.py`; this
module only parses flags into options objects and delegates, per the
CLI-options memo (E19) - the same shape as every command in
``cli/transcript.py``.
"""

from __future__ import annotations

from pathlib import Path

import click

from ..claude import ClaudeAgentError
from ..transcription.adapt import AdaptError
from ..transcription.asr import TranscriberError
from ..transcription.audio import (
    AudioEmbedResolutionError,
    AudioToolError,
    NoAudioEmbedsError,
)
from ..transcription.chapters import ChaptersError
from ..transcription.integrate import IntegrateError
from ..transcription.minutes import MinutesError
from ..transcription.note import NoteParseError
from ..transcription.obsidian import ObsidianCliError
from ..transcription.pipeline import PipelineError, PipelineFactories, run_pipeline
from ..transcription.polish import PolishError
from ..transcription.product_review import ProductReviewError
from ..transcription.speakers import SpeakersError, SpeakersResponse
from .options import AgentOptions, agent_options, coro
from .transcript import NEEDS_INPUT_EXIT_CODE, REVIEW_REQUIRED_EXIT_CODE
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

# Domain errors mapped to a clean `ClickException` - the same precedent
# every `transcript` sub-command follows (domain errors + `ClaudeAgentError`
# -> `ClickException`). This is a manually maintained union of every
# `except (...)` tuple across `cli/transcript.py`'s own sub-commands (each
# stage raises its own error type(s); some, like `audio.py`'s
# `NoAudioEmbedsError`/`AudioEmbedResolutionError`, deliberately do NOT
# share a base with that module's other error, `AudioToolError`, so no
# single `issubclass` check can stand in for this list). Keep it in sync
# with `cli/transcript.py` when a stage adds or changes an error type; the
# CLI-layer tests below exercise at least one case per family so an
# omission fails loudly rather than surfacing as a raw traceback.
_STAGE_ERRORS: tuple[type[Exception], ...] = (
    PipelineError,
    SpeakersError,
    ChaptersError,
    PolishError,
    MinutesError,
    IntegrateError,
    ProductReviewError,
    AdaptError,
    TranscriberError,
    AudioToolError,
    NoAudioEmbedsError,
    AudioEmbedResolutionError,
    ObsidianCliError,
    ClaudeAgentError,
    NoteParseError,
)


@click.command("transcribe")
@agent_options
@obsidian_options
@audio_options
@transcriber_options
@cache_options
@click.argument(
    "note_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--assign",
    "assign",
    multiple=True,
    metavar="CLUSTER=NAME",
    help=(
        'Confirm one speaker cluster, e.g. --assign "SPEAKER_03=Nikki '
        'Staltari" (repeatable) - the same contract `transcript speakers '
        "--assign` uses. Use this to resume a run that stopped for "
        "needs_input."
    ),
)
@click.option(
    "--correct",
    "correct",
    multiple=True,
    metavar="CLUSTER=START-END=NAME",
    help="Relabel utterances fully contained in a half-open range (repeatable).",
)
@click.option(
    "--finalise",
    "finalise",
    is_flag=True,
    help=(
        "Map any speaker clusters still unresolved after --assign and the "
        'LLM\'s own proposals to "Unknown", instead of asking for more '
        "input."
    ),
)
@click.option(
    "--max-concurrency",
    "max_concurrency",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Maximum number of chapters polished concurrently.",
)
@coro
async def transcribe(
    note_path: Path,
    assign: tuple[str, ...],
    correct: tuple[str, ...],
    finalise: bool,
    max_concurrency: int,
    agent_options: AgentOptions,
    obsidian_options: ObsidianOptions,
    audio_options: AudioOptions,
    transcriber_options: TranscriberOptions,
    cache_options: CacheOptions,
) -> None:
    """Run the full meeting-transcription pipeline over NOTE_PATH.

    One flow, full product set, no partial modes: merges NOTE_PATH's audio
    embeds and transcribes them (or adapts a pre-diarised transcript embed
    when there's no audio), resolves speakers, chapterises, polishes,
    generates the meeting summary and Discussion Notes, and writes every
    product into the note - the same eight stages `transcript merge-audio` /
    `asr` / `adapt` / `speakers` / `chapterise` / `polish` / `minutes` /
    `integrate` run individually, composed into one call. ASR/diarisation
    (or adapt, on the text ramp) checks the run cache first and model-backed
    chapterise/polish/minutes products are reused only when their durable
    content/config manifests match. A mismatch invalidates dependent
    products before regeneration; telemetry remains durable across the
    needs_input -> `--assign` resume loop.

    If speaker resolution can't confidently name every cluster, this prints
    `{"status": "needs_input", "run_id": ..., "requests": [...]}` and exits
    with code 3 - the same status/run-id/requests/exit-code contract as
    `transcript speakers`, with additive timing telemetry. This is the same
    contract a coordinating agent already knows how to relay (e.g. over
    Discord) and answer. Re-run with more `--assign "SPEAKER_03=Name"` flags
    to supply an answer, or `--assign "...=Unknown" --finalise` to give up on
    the rest and proceed anyway.

    On success, prints a `RunReport` as JSON and exits 0: the run id, chapter
    count, the ratio of "Unknown"-speaker turns to resolved source turns
    when provenance is available (otherwise polished turns), the fixer's
    issues (from `polish_issues.json`), and the `IntegrationReport` describing
    what changed in the note. This report is informational, never gating -
    Michael's own skim of the note is the acceptance test; the report just
    says where to aim it.

    `--model`/`--effort` apply to every LLM call this run makes (chapterise
    still defaults to `--effort low` when `--effort` isn't given, the same
    default `transcript chapterise` applies, on the grounds that boundary
    proposal is a cheap-model job).

    A coordinating agent with no direct access to Michael's judgement can ask
    Jake (his Hermes agent) for supporting context - which prep note to run
    against, how to answer a needs_input request - via
    `hermes chat --quiet --query '...'` (`--resume <session_id>` to continue
    a session).
    """
    factories = PipelineFactories(
        vault=obsidian_options.vault_client,
        audio_tool=audio_options.audio_tool,
        transcriber=transcriber_options.transcriber,
        agent=agent_options.agent,
        cache=cache_options.run_cache,
    )

    try:
        if correct:
            outcome = await run_pipeline(
                note_path,
                factories,
                agent_options.spec(),
                assignments=assign,
                corrections=correct,
                finalise=finalise,
                max_concurrency=max_concurrency,
            )
        else:
            outcome = await run_pipeline(
                note_path,
                factories,
                agent_options.spec(),
                assignments=assign,
                finalise=finalise,
                max_concurrency=max_concurrency,
            )
    except _STAGE_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc

    if outcome.status == "needs_input":
        # Preserve the needs_input status/requests/exit semantics while
        # allowing additive timing telemetry in the shared payload.
        payload = SpeakersResponse(
            status="needs_input",
            run_id=outcome.run_id or "",
            requests=outcome.requests,
            timings=outcome.timings,
            telemetry=outcome.telemetry,
        )
        click.echo(payload.model_dump_json(indent=2))
        raise click.exceptions.Exit(NEEDS_INPUT_EXIT_CODE)

    if outcome.status == "review_required":
        click.echo(outcome.model_dump_json(indent=2))
        raise click.exceptions.Exit(REVIEW_REQUIRED_EXIT_CODE)

    assert outcome.report is not None  # "complete" always carries a report
    click.echo(outcome.report.model_dump_json(indent=2))
