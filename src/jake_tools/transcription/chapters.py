"""Chapterise a raw transcript into topic-based spans, before any polishing.

Chapterising runs on the raw transcript, not the polished one, by explicit
design: chapters partition the meeting so each chapter can be polished and
summarised independently (plans 008-009), with bounded context and cheaper
per-chapter models. Chapter boundaries need topic flow, not clean prose, so
raw ASR text is sufficient input here.

The model is only ever trusted to propose *where* a chapter starts and
*what* to call it. Everything about how those proposals become a valid
partition of the utterance sequence - sorting, clamping to valid indexes,
dropping duplicates, deriving each span's inclusive end from the next
boundary - happens deterministically in code (`_repair_boundaries`,
`_spans_from_boundaries`). Every utterance ends up in exactly one chapter;
that invariant is enforced here rather than trusted to model arithmetic.

`_render_utterances` includes each utterance's elapsed-time prefix
(`MM:SS`), not just its index, so the model can actually judge how long a
candidate chapter would span - utterance *count* is a catastrophic proxy
for duration on heavily crosstalk-fragmented ASR, where a 30-second burst
of interruptions produces as many utterances as two minutes of a clean
monologue. After the model proposes boundaries, `_merge_short_spans` is a
second deterministic backstop, mirroring `_repair_boundaries`'s asymmetric
existing guard: `_RETRY_NOTE` catches a reply that is too coarse (collapses
to one chapter), but nothing previously caught a reply that is too fine
(many chapters under a minute). `_merge_short_spans` mechanically folds any
chapter shorter than `MIN_CHAPTER_SECONDS` into its shorter-duration
neighbour - never another LLM call - so a too-fine reply can no longer ship
regardless of what the model returns.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

from pydantic import BaseModel, TypeAdapter

from ..claude import AgentSpec, ClaudeAgent
from ..prompting import StructuredPrompt
from .cache import RunCache
from .models import ChapterSpan, RawTranscript, Utterance
from .speakers import RESOLVED_TRANSCRIPT_CACHE_NAME

# The cache name `transcript chapterise` stores its output under.
CHAPTERS_CACHE_NAME = "chapters"

# Adapter for the CLI's stdout print, which is a bare `list[ChapterSpan]`
# (the porcelain contract) - not what's written to disk. See `ChapterList`
# for the on-disk shape.
CHAPTERS_ADAPTER: TypeAdapter[list[ChapterSpan]] = TypeAdapter(list[ChapterSpan])


class ChapterList(BaseModel):
    """On-disk wrapper for `chapters.json`.

    `chapterise`/`run_chapterisation`'s real output is a bare
    `list[ChapterSpan]`, but `RunCache.store`/`.load` only round-trip a
    `BaseModel` (like every sibling stage's cache artefact), so this wraps
    the list for storage only. Plans 008-009 import this to load
    `chapters.json` back out of the run cache.
    """

    chapters: list[ChapterSpan]


MIN_CHAPTER_SECONDS = 60.0
"""Floor for a chapter's duration before `_merge_short_spans` folds it into
a neighbour. Matches the prompt's own "spans minutes, not seconds"
guidance - this is what actually enforces it when the model doesn't."""

_RETRY_NOTE = (
    "Your previous answer collapsed the whole meeting into a single "
    "chapter (or returned no boundaries at all), which is too coarse to be "
    "useful. Look again for places the topic actually changes and return "
    "at least two boundaries."
)


class ChaptersError(RuntimeError):
    """Base for chapterisation domain errors."""


class MissingResolvedTranscriptError(ChaptersError):
    """Raised when no `resolved_transcript.json` is cached for a run id."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"no cached resolved transcript for run {run_id!r}; run "
            f"`transcript speakers --run-id {run_id} ...` first."
        )
        self.run_id = run_id


class DegenerateChaptersError(ChaptersError):
    """Raised when chapterisation still yields <=1 usable chapter after a retry."""

    def __init__(self) -> None:
        super().__init__(
            "chapterisation produced a single chapter for the whole meeting "
            "even after a corrective retry; refusing to fabricate boundaries."
        )


# --- LLM-backed boundary proposal --------------------------------------------


class ChapterBoundary(BaseModel):
    """One chapter's proposed start, before deterministic post-processing."""

    title: str
    start_utterance: int


class ChapterisationResponse(BaseModel):
    """The LLM's reply: chapter boundaries in whatever order it produced them."""

    chapters: list[ChapterBoundary]


class ChapterisationPrompt(StructuredPrompt[ChapterisationResponse]):
    template = textwrap.dedent("""\
        You are chapterising a raw, unproofread meeting transcript into
        topic-based chapters, before any polishing pass. Expect ASR noise -
        filler words, fragments, mis-transcriptions - and judge boundaries
        by topic flow, not grammar.

        Rules:
        - A boundary marks a natural topic shift, not merely a change of
          speaker.
        - A chapter title is a topic-descriptive noun phrase in sentence
          case, e.g. "Beachhead use cases and product vs vehicle role
          classification" or "Scheduling on-road training" - never a
          question, and never generic filler like "Discussion" or
          "Introduction".
        - A chapter typically spans minutes of dialogue, not seconds -
          don't split on every remark.
        - The first chapter starts at utterance 0.
        - Each `start_utterance` must be one of the utterance indexes shown
          below, and each index should be used at most once.

        Transcript (utterance index | elapsed time MM:SS | speaker: text) -
        use the elapsed time to judge how long a candidate chapter would
        actually span; utterance count is not a reliable proxy, since
        crosstalk can fragment a short burst of interruptions into as many
        utterances as minutes of a clean monologue:
        {{ transcript_lines }}
        {% if corrective_note %}

        {{ corrective_note }}
        {% endif %}
        """)
    response_model = ChapterisationResponse

    transcript_lines: str
    corrective_note: str = ""


def _format_elapsed(seconds: float) -> str:
    """`MM:SS` (or `H:MM:SS` past the first hour) for a meeting-relative offset."""

    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _render_utterances(utterances: Sequence[Utterance]) -> str:
    """Compact index/elapsed-time/speaker/text rendering.

    Utterance index is the contract between this rendering and the
    boundaries the model proposes; `start_seconds` is derived from it
    afterwards, in code, not asked of the model. The elapsed-time column is
    the actual duration signal the model needs to obey the prompt's "spans
    minutes, not seconds" rule - without it, utterance count is the only
    proxy available, and it is a bad one whenever crosstalk fragments the
    ASR (see the module docstring).
    """

    return "\n".join(
        f"{index} | {_format_elapsed(utterance.start)} | {utterance.speaker}: "
        f"{utterance.text}"
        for index, utterance in enumerate(utterances)
    )


def _repair_boundaries(
    boundaries: Sequence[ChapterBoundary], utterance_count: int
) -> list[ChapterBoundary]:
    """Deterministically repair whatever boundaries the model returned.

    Never trusts model arithmetic: clamps every index into the valid
    range, sorts by `start_utterance`, drops duplicates (keeping the first
    title seen for a given index once sorted), and forces the earliest
    surviving boundary to start at utterance 0 - reusing its title rather
    than fabricating a new one - so the eventual spans cover every
    utterance with no gap at the front. The result is always a strictly
    increasing sequence of indexes starting at 0 (or empty, for an empty
    transcript), which `_spans_from_boundaries` relies on to guarantee the
    partition invariant.
    """

    if utterance_count == 0:
        return []

    clamped = [
        boundary.model_copy(
            update={
                "start_utterance": max(
                    0, min(boundary.start_utterance, utterance_count - 1)
                )
            }
        )
        for boundary in boundaries
    ]
    ordered = sorted(clamped, key=lambda boundary: boundary.start_utterance)

    deduped: list[ChapterBoundary] = []
    seen: set[int] = set()
    for boundary in ordered:
        if boundary.start_utterance in seen:
            continue
        seen.add(boundary.start_utterance)
        deduped.append(boundary)

    if deduped and deduped[0].start_utterance != 0:
        deduped[0] = deduped[0].model_copy(update={"start_utterance": 0})

    return deduped


def _spans_from_boundaries(
    boundaries: Sequence[ChapterBoundary], utterances: Sequence[Utterance]
) -> list[ChapterSpan]:
    """Turn a repaired (strictly increasing, 0-based) boundary sequence into spans.

    Each span's `end_utterance` is the next boundary's start minus one (the
    last span runs to the final utterance), and `start_seconds` is read
    from the span's first utterance - never computed by the model.
    """

    spans: list[ChapterSpan] = []
    for index, boundary in enumerate(boundaries):
        end = (
            boundaries[index + 1].start_utterance - 1
            if index + 1 < len(boundaries)
            else len(utterances) - 1
        )
        spans.append(
            ChapterSpan(
                title=boundary.title,
                start_utterance=boundary.start_utterance,
                end_utterance=end,
                start_seconds=utterances[boundary.start_utterance].start,
            )
        )
    return spans


def _span_duration(
    index: int, spans: Sequence[ChapterSpan], utterances: Sequence[Utterance]
) -> float:
    """How long `spans[index]` actually runs: to the next span's start, or
    to the last utterance's end for the final span."""

    if index + 1 < len(spans):
        return spans[index + 1].start_seconds - spans[index].start_seconds
    return utterances[-1].end - spans[index].start_seconds


def _merge_spans(
    spans: list[ChapterSpan], short_index: int, neighbour_index: int
) -> list[ChapterSpan]:
    """Fold `spans[short_index]` into `spans[neighbour_index]` (adjacent, either order).

    The merged span keeps the *neighbour's* title - the short span
    dissolves into an existing chapter's identity rather than renaming it,
    since nothing here is asking the model for a new title.
    """

    lo, hi = sorted((short_index, neighbour_index))
    merged = ChapterSpan(
        title=spans[neighbour_index].title,
        start_utterance=spans[lo].start_utterance,
        end_utterance=spans[hi].end_utterance,
        start_seconds=spans[lo].start_seconds,
    )
    return spans[:lo] + [merged] + spans[hi + 1 :]


def _merge_short_spans(
    spans: Sequence[ChapterSpan],
    utterances: Sequence[Utterance],
    *,
    min_seconds: float = MIN_CHAPTER_SECONDS,
) -> list[ChapterSpan]:
    """Deterministically merge every chapter shorter than `min_seconds`.

    Repeatedly finds the shortest surviving chapter; if it clears
    `min_seconds`, stops. Otherwise merges it into whichever adjacent
    neighbour is itself shorter (ties favour the earlier neighbour) - two
    over-split chapters are more likely to be one over-split topic than
    either is to belong with a neighbour that is already a healthy length.
    Never merges below two chapters, mirroring `_RETRY_NOTE`'s "at least
    two boundaries" floor on the too-coarse side. Pure post-processing, not
    another model call: the one guard the review found nothing protecting.
    """

    result = list(spans)
    while len(result) > 2:
        durations = [_span_duration(i, result, utterances) for i in range(len(result))]
        shortest_index = min(range(len(result)), key=lambda i: durations[i])
        if durations[shortest_index] >= min_seconds:
            break

        if shortest_index == 0:
            neighbour_index = 1
        elif (
            shortest_index == len(result) - 1
            or durations[shortest_index - 1] <= durations[shortest_index + 1]
        ):
            neighbour_index = shortest_index - 1
        else:
            neighbour_index = shortest_index + 1

        result = _merge_spans(result, shortest_index, neighbour_index)

    return result


async def chapterise(
    transcript: RawTranscript,
    *,
    agent: ClaudeAgent,
    spec: AgentSpec | None = None,
) -> list[ChapterSpan]:
    """Chapterise `transcript` into topic-based `ChapterSpan`s.

    Renders a compact index/elapsed-time/speaker/text view of the
    transcript and asks the model for chapter boundaries. Post-processing
    into spans is entirely deterministic (see `_repair_boundaries`/
    `_spans_from_boundaries`/`_merge_short_spans`): every utterance ends up
    in exactly one chapter regardless of what the model returned, and no
    chapter survives shorter than `MIN_CHAPTER_SECONDS` regardless of how
    finely the model split the meeting.

    A degenerate reply (no boundaries, or boundaries that all collapse to
    a single chapter covering the whole meeting) gets one retry with a
    corrective note appended to the prompt. Still degenerate after that
    raises `DegenerateChaptersError` - the caller surfaces the error rather
    than this function fabricating boundaries.
    """

    lines = _render_utterances(transcript.utterances)
    boundaries = await _propose_and_repair(
        lines, "", agent=agent, spec=spec, utterance_count=len(transcript.utterances)
    )

    if len(boundaries) <= 1:
        boundaries = await _propose_and_repair(
            lines,
            _RETRY_NOTE,
            agent=agent,
            spec=spec,
            utterance_count=len(transcript.utterances),
        )

    if len(boundaries) <= 1:
        raise DegenerateChaptersError()

    spans = _spans_from_boundaries(boundaries, transcript.utterances)
    return _merge_short_spans(spans, transcript.utterances)


async def _propose_and_repair(
    transcript_lines: str,
    corrective_note: str,
    *,
    agent: ClaudeAgent,
    spec: AgentSpec | None,
    utterance_count: int,
) -> list[ChapterBoundary]:
    response, _reply = await agent.run_structured(
        ChapterisationPrompt(
            transcript_lines=transcript_lines, corrective_note=corrective_note
        ),
        spec,
        stage="chapterise",
    )
    return _repair_boundaries(response.chapters, utterance_count)


# --- run orchestration: cache state around one `chapterise` call ------------


async def run_chapterisation(
    run_id: str, *, agent: ClaudeAgent, cache: RunCache
) -> list[ChapterSpan]:
    """Drive one `jake-tools transcript chapterise` invocation.

    Loads the cached resolved transcript (`MissingResolvedTranscriptError`
    if there isn't one, naming `transcript speakers` as the prerequisite),
    runs `chapterise`, and stores the result as `chapters.json` before
    returning it.
    """

    transcript = cache.load(run_id, RESOLVED_TRANSCRIPT_CACHE_NAME, RawTranscript)
    if transcript is None:
        raise MissingResolvedTranscriptError(run_id)

    stage_agent = agent.for_stage("chapterise").with_telemetry(
        cache.telemetry_sink(run_id)
    )
    manifest = cache.stage_manifest(
        "chapterise",
        inputs={"resolved_transcript": transcript.model_dump(mode="json")},
        config={
            "agent": stage_agent.defaults.model_dump(mode="json"),
            "prompt": ChapterisationPrompt.template,
            "response_schema": ChapterisationResponse.model_json_schema(),
            "retry_note": _RETRY_NOTE,
        },
    )
    cached = cache.load(run_id, CHAPTERS_CACHE_NAME, ChapterList)
    if cached is not None and cache.load_manifest(run_id, "chapterise") == manifest:
        stage_agent.record_cache_hit()
        return cached.chapters
    if cached is not None or cache.load_manifest(run_id, "chapterise") is not None:
        cache.invalidate_artefacts(
            run_id,
            {
                "polished",
                "polish_issues",
                "polish.manifest",
                "minutes",
                "minutes.manifest",
            },
        )

    chapters = await chapterise(transcript, agent=stage_agent)
    cache.store(run_id, CHAPTERS_CACHE_NAME, ChapterList(chapters=chapters))
    cache.store_manifest(run_id, manifest)
    return chapters
