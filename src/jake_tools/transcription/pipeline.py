"""The composed transcription pipeline: `jake-tools transcribe`'s library seam.

This is the porcelain the operator actually runs, one command over one note,
producing the full product set. It is deliberately thin composition over the
plumbing stages plans 003-010 already built as library functions — every bit
of actual transcription/polishing/writing logic stays in its owning stage
module; this module only decides *which* stages to call, *in what order*,
and *what to do with a stage that needs human input*.

**Ramp selection.** A note enters the pipeline one of two ways: an audio
embed (`![[Recording ....m4a]]`) merges via `merge_note_audio` (003) then
transcribes via `transcribe_merged_audio` (004); a pre-diarised transcript
embed (`.vtt`/`.txt` - a Gemini/Teams export the coordinating agent already
placed in the vault) adapts via `adapt_transcript` (005). A note with neither
raises `NoEntryRampError`. Audio wins when a note somehow has both, matching
the plan's stated priority order.

**Resume.** `run_speaker_resolution` (006) returns `needs_input` when
clusters remain unresolved; this function stops there and returns a
`needs_input` `PipelineOutcome` *before* chapterising/polishing/writing
anything. Local checkpoints and every model-backed product use durable,
content/config-bound manifests: a valid hit replays the product without a
model call, while a mismatch invalidates dependent products and regenerates
them. The `ai_telemetry.json` audit document is append-only from the
operator's perspective and survives needs_input, failures, retries, and
later inspection.

**Clips provenance.** `transcribe_merged_audio`'s `RawTranscript.clips`
carries a single fabricated pseudo-clip for the whole merged file (asr.py's
own docstring: it only ever sees the merged audio path, not the per-source
breakdown). This module has both pieces at hand - `merge_note_audio`'s real
`SourceClip` list and the cached transcript - so it splices the real clips
back in and re-persists the corrected transcript, all through
`RawTranscript`'s public fields and `RunCache`'s public store/load, without
touching `asr.py`. `.clips` is provenance-only today (no downstream stage
reads it back for computation), so this is a data-quality fix with no
behavioural effect - but it is one the maintenance notes explicitly called
for, so it's done here rather than deferred.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..ai_usage import AITelemetry
from ..claude import AgentSpec, ClaudeAgent
from .adapt import run_adapt
from .asr import Transcriber, transcribe_merged_audio
from .audio import AudioTool, merge_note_audio, run_id_for
from .cache import RunCache, sha256_of
from .chapters import run_chapterisation
from .integrate import IntegrationReport, load_products
from .minutes import run_minutes
from .models import PolishedChapter, RawTranscript, SnippetRequest, StageTiming
from .note import ParsedNote, parse_note
from .obsidian import VaultClient
from .polish import run_polish
from .product_review import export_product_review
from .speakers import RAW_TRANSCRIPT_CACHE_NAME, run_speaker_resolution

_AUDIO_SUFFIXES = (".m4a",)
_TRANSCRIPT_SUFFIXES = (".vtt", ".txt")

# `run_chapterisation`'s CLI wrapper (`transcript chapterise`) layers
# effort="low" onto the agent spec whenever `--effort` was not explicitly
# passed, on the grounds that chapterisation is a cheap-model stage. That
# choice lives in the CLI handler, not in `chapters.py`'s library code, so
# faithfully composing the same stage here means reproducing it, not
# depending on `chapters.py` to have it.
_CHAPTERISE_DEFAULT_EFFORT = "low"


class PipelineError(RuntimeError):
    """Base for pipeline-composition domain errors (not a stage's own error)."""


class NoEntryRampError(PipelineError):
    """Raised when a note has neither an audio embed nor a transcript embed."""

    def __init__(self, note_path: Path) -> None:
        super().__init__(
            f"note {str(note_path)!r} has no audio embed (e.g. "
            "![[Recording ....m4a]]) and no recognisable pre-diarised "
            f"transcript embed ({'/'.join(_TRANSCRIPT_SUFFIXES)}) to start "
            "the pipeline from."
        )
        self.note_path = note_path


@dataclass(frozen=True)
class PipelineFactories:
    """Zero-argument constructors for every dependency `run_pipeline` might need.

    Each field mirrors one of `transcript_options.py`'s options-object
    dependency-constructor methods (`ObsidianOptions.vault_client`, ...): the
    CLI passes the bound method itself (`obsidian_options.vault_client`,
    ...), a plain function call with no `ctx.obj` involved anywhere - this is
    a dataclass parameter of an ordinary function, not Click state. Tests
    pass a lambda returning a fake instead. Deferred construction (a
    callable, not the object) means a ramp this run never takes never pays
    to construct the dependency it would have needed - a text-ramp run never
    constructs a `Transcriber` (potentially a real model load), for example.
    """

    vault: Callable[[], VaultClient]
    audio_tool: Callable[[], AudioTool]
    transcriber: Callable[[], Transcriber]
    agent: Callable[[], ClaudeAgent]
    cache: Callable[[], RunCache]


class RunReport(BaseModel):
    """What one completed pipeline run produced, for Michael's skim.

    Informational only, never gating - nothing in `run_pipeline` inspects
    these values to decide success or failure. Additive: timing and durable
    API-rate-equivalent telemetry can be layered on without breaking callers
    that only read the original fields. Costs are list-price/API-equivalent
    estimates, not invoices or subscription marginal costs.
    """

    run_id: str
    chapters: int
    unknown_turn_ratio: float  # Unknown-speaker turns / total resolved turns
    resolved_unknown_turns: int = 0
    resolved_unknown_words: int = 0
    fixer_issues: list[str]  # from polish_issues.json
    integration: IntegrationReport
    timings: list[StageTiming] = Field(default_factory=list)
    telemetry: AITelemetry | None = None


class PipelineOutcome(BaseModel):
    """What one `run_pipeline` call produced.

    `run_id` is populated as soon as the entry ramp resolves one (before
    speaker resolution runs), in both branches - `needs_input` included, so
    a caller can print the exact same `{"status": "needs_input", "run_id":
    ..., "requests": [...]}` shape `transcript speakers` prints, keeping the
    two entry points' contracts identical.
    """

    status: Literal["complete", "needs_input", "review_required"]
    run_id: str | None = None
    requests: list[SnippetRequest] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    telemetry: AITelemetry | None = None
    report: RunReport | None = None
    review_id: str | None = None
    candidate: str | None = None
    pack: str | None = None


def _known_speaker_count(note: ParsedNote) -> int | None:
    attendees = {attendee.strip() for attendee in note.attendees if attendee.strip()}
    return len(attendees) or None


def _embed_target(embed: str) -> str:
    # `target|alias` embeds carry their alias verbatim in `ParsedNote.embeds`
    # (see `merge_note_audio`'s docstring) - strip it before matching a
    # suffix, the same way `merge_note_audio` does for audio embeds.
    return embed.split("|", 1)[0]


def _has_audio_embed(note: ParsedNote) -> bool:
    return any(
        _embed_target(embed).lower().endswith(_AUDIO_SUFFIXES) for embed in note.embeds
    )


def _transcript_embed_target(note: ParsedNote) -> str | None:
    for embed in note.embeds:
        target = _embed_target(embed)
        if target.lower().endswith(_TRANSCRIPT_SUFFIXES):
            return target
    return None


def _polished_unknown_ratio(chapters: Sequence[PolishedChapter]) -> float:
    turns = [turn for chapter in chapters for turn in chapter.turns]
    if not turns:
        return 0.0
    return sum(turn.speaker == "Unknown" for turn in turns) / len(turns)


def _unknown_evidence(transcript: RawTranscript) -> tuple[float, int, int]:
    unknown = [u for u in transcript.utterances if u.speaker == "Unknown"]
    total = len(transcript.utterances)
    words = sum(len(u.text.split()) for u in unknown)
    return (len(unknown) / total if total else 0.0, len(unknown), words)


async def _acquire_raw_transcript(
    note: ParsedNote,
    note_path: Path,
    *,
    vault: VaultClient,
    audio_tool: AudioTool,
    transcriber_factory: Callable[[], Transcriber],
    agent: ClaudeAgent,
    cache: RunCache,
) -> str:
    """Run this note's entry ramp, returning the run id its transcript lives under.

    Audio ramp: merges (never itself cache-gated - see the module
    docstring), then transcribes (cache-gated inside
    `transcribe_merged_audio`), then splices the real per-source `clips`
    back into the cached transcript. Text ramp: hashes the transcript file
    to derive a stable run id (`cache.py`'s own contract - "computed by
    callers, not here" - text sources have no audio to hash instead), then
    adapts only on a cache miss (`adapt_transcript` has no cache of its own;
    this is the porcelain doing for it what `transcribe_merged_audio` does
    natively for ASR).
    """
    if _has_audio_embed(note):
        merge_result = merge_note_audio(
            note, vault=vault, audio_tool=audio_tool, cache=cache
        )
        run_id = merge_result.run_id
        transcriber = transcriber_factory()
        raw = transcribe_merged_audio(
            Path(merge_result.merged_path),
            run_id=run_id,
            transcriber=transcriber,
            cache=cache,
            num_speakers=_known_speaker_count(note),
            media_duration_seconds=sum(
                clip.duration_seconds for clip in merge_result.clips
            ),
        )
        if raw.clips != merge_result.clips:
            corrected = raw.model_copy(update={"clips": merge_result.clips})
            cache.store(run_id, RAW_TRANSCRIPT_CACHE_NAME, corrected)
        return run_id

    transcript_target = _transcript_embed_target(note)
    if transcript_target is None:
        raise NoEntryRampError(note_path)

    transcript_path = vault.resolve_embed(transcript_target)
    content_sha256 = sha256_of(transcript_path)
    run_id = run_id_for(Path(note.path), content_sha256)

    await run_adapt(transcript_path, run_id=run_id, agent=agent, cache=cache)
    return run_id


async def run_pipeline(
    note_path: Path,
    ctx_factories: PipelineFactories,
    spec: AgentSpec,
    *,
    assignments: Sequence[str] = (),
    corrections: Sequence[str] = (),
    finalise: bool = False,
    max_concurrency: int = 4,
) -> PipelineOutcome:
    """Run the full transcription pipeline over `note_path`, one call, full product set.

    1. Parses the note and picks the entry ramp (`_acquire_raw_transcript`).
    2. Resolves speakers (`run_speaker_resolution`, 006). Unresolved
       clusters stop the pipeline here with a `needs_input` outcome -
       `chapterise`/`polish`/`minutes`/`integrate` never run this call.
    3. Chapterises (007, with the same `effort="low"` default `transcript
       chapterise` applies when `--effort` wasn't explicit), polishes (008),
       reviews minutes (009b), and exports an exact product candidate (011).
       Canonical integration is deliberately not performed here.
    4. Returns `review_required` with the review id, candidate, timings, and
       AI telemetry. `needs_input` remains the earlier human checkpoint.

    `ctx_factories` supplies every dependency lazily (see `PipelineFactories`);
    `spec` carries `--model`/`--effort` through to every LLM stage's agent.
    """
    cache = ctx_factories.cache()
    vault = ctx_factories.vault()
    audio_tool = ctx_factories.audio_tool()
    agent = ctx_factories.agent()

    note = parse_note(note_path)
    run_id = await _acquire_raw_transcript(
        note,
        note_path,
        vault=vault,
        audio_tool=audio_tool,
        transcriber_factory=ctx_factories.transcriber,
        agent=agent,
        cache=cache,
    )

    speakers_result = await run_speaker_resolution(
        note_path,
        run_id,
        assign=assignments,
        correct=corrections,
        finalise=finalise,
        agent=agent,
        audio_tool=audio_tool,
        cache=cache,
    )
    if speakers_result.status == "needs_input":
        return PipelineOutcome(
            status="needs_input",
            run_id=run_id,
            requests=speakers_result.requests,
            timings=cache.load_timings(run_id),
            telemetry=cache.load(run_id, "ai_telemetry", AITelemetry),
        )

    chapter_spec = (
        spec
        if spec.effort is not None
        else spec.model_copy(update={"effort": _CHAPTERISE_DEFAULT_EFFORT})
    )
    chapter_agent = ClaudeAgent(defaults=chapter_spec, run_query=agent.run_query)
    await run_chapterisation(run_id, agent=chapter_agent, cache=cache)

    await run_polish(
        note_path,
        run_id,
        chapter=None,
        max_concurrency=max_concurrency,
        agent=agent,
        vault=vault,
        cache=cache,
    )

    await run_minutes(note_path, run_id, agent=agent, vault=vault, cache=cache)

    products = load_products(note_path, run_id, cache)
    review_pack = export_product_review(note_path, run_id, cache=cache)
    resolved = cache.load(run_id, "resolved_transcript", RawTranscript)
    evidence_ratio, unknown_turns, unknown_words = _unknown_evidence(
        resolved if resolved is not None else RawTranscript(clips=[], utterances=[])
    )
    telemetry = cache.load(run_id, "ai_telemetry", AITelemetry)
    report = RunReport(
        run_id=run_id,
        chapters=len(products.chapters),
        unknown_turn_ratio=(
            evidence_ratio
            if any(
                turn.source_turn_indices
                for chapter in products.chapters
                for turn in chapter.turns
            )
            else _polished_unknown_ratio(products.chapters)
        ),
        resolved_unknown_turns=unknown_turns,
        resolved_unknown_words=unknown_words,
        fixer_issues=[],
        integration=IntegrationReport(
            run_id=run_id, note_path=str(note_path), sections=[]
        ),
        timings=cache.load_timings(run_id),
        telemetry=telemetry,
    )
    return PipelineOutcome(
        status="review_required",
        run_id=run_id,
        timings=cache.load_timings(run_id),
        telemetry=telemetry,
        report=report,
        review_id=review_pack.review_id,
        candidate=review_pack.candidate_path,
        pack=str(cache.run_dir(run_id) / "product_review.json"),
    )
