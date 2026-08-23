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

        Transcript (utterance index | speaker: text):
        {{ transcript_lines }}
        {% if corrective_note %}

        {{ corrective_note }}
        {% endif %}
        """)
    response_model = ChapterisationResponse

    transcript_lines: str
    corrective_note: str = ""


def _render_utterances(utterances: Sequence[Utterance]) -> str:
    """Compact index/speaker/text rendering - timestamps omitted to save tokens.

    Utterance index is the contract between this rendering and the
    boundaries the model proposes; `start_seconds` is derived from it
    afterwards, in code, not asked of the model.
    """

    return "\n".join(
        f"{index} | {utterance.speaker}: {utterance.text}"
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


async def chapterise(
    transcript: RawTranscript,
    *,
    agent: ClaudeAgent,
    spec: AgentSpec | None = None,
) -> list[ChapterSpan]:
    """Chapterise `transcript` into topic-based `ChapterSpan`s.

    Renders a compact index/speaker/text view of the transcript (no
    timestamps in the prompt - indexes are the contract) and asks the model
    for chapter boundaries. Post-processing into spans is entirely
    deterministic (see `_repair_boundaries`/`_spans_from_boundaries`):
    every utterance ends up in exactly one chapter regardless of what the
    model returned.

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

    return _spans_from_boundaries(boundaries, transcript.utterances)


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

    chapters = await chapterise(transcript, agent=agent)
    cache.store(run_id, CHAPTERS_CACHE_NAME, ChapterList(chapters=chapters))
    return chapters
