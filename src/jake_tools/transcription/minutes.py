"""Generate the meeting summary and Discussion Notes minutes from polished chapters.

These minutes are the first thing Michael reads on every run - his
validation ritual is "read the minutes and chapters, then skim the
transcript". He has a named pet hate: AI minutes that *tell you what needs
to be done* instead of reporting what was said and letting the human
decide. One `StructuredPrompt` over every polished chapter (this stage does
not chunk-and-merge - a long meeting either fits one call or this stage
stops and hands back, per the plan's STOP conditions) returns two fields:

- `meeting_summary` - a short callout body (no callout syntax; rendering
  that is plan 010's job).
- `discussion_notes` - nested Markdown bullets, no "## Discussion Notes"
  heading.

The report-never-prescribe rule is stated imperatively in the prompt
(`MinutesPrompt.template`) and is a personality-level requirement, not a
style nicety: weakening it is a regression even if the output "looks
helpful". A second, equally imperative rule sits alongside it: GROUNDING -
every bullet must trace to something a named speaker actually said, never
an inferred date, owner, or action item, and never a statement folded onto
the wrong speaker. Prefer omission over a confident guess. Wikilinks reuse
`polish.py`'s `build_lexicon` (attendees, wikilink
targets already in the note, vault note titles) as the candidate set -
attendees are always linked, everything else only "plausibly", so link
quality is tuned by tightening that lexicon, not by loosening the linking
rule here.

Like `polish.py`, this module is tested at two layers (see
`tests/test_transcription_minutes.py` and, for the pattern this follows,
`tests/test_transcription_polish.py`/`tests/test_transcription_chapters.py`):
fast fake-`run_query` tests pin the plumbing and the load-bearing prompt
instructions (a silent prompt regression - e.g. the report-never-prescribe
sentence quietly disappearing - must fail a test even though a fake can't
judge output quality), and a `@pytest.mark.slow` real-model test (skipped by
default, run with `--slow`) is the only layer that can actually tell whether
the prompt teaches the model to write minutes the way Michael wants.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from ..claude import AgentSpec, ClaudeAgent
from ..prompting import StructuredPrompt
from .cache import RunCache
from .models import PolishedChapter
from .note import parse_note
from .obsidian import VaultClient
from .polish import POLISHED_CACHE_NAME, PolishedChapterList, build_lexicon

# The cache name `transcript minutes` stores its output under.
MINUTES_CACHE_NAME = "minutes"


class MinutesError(RuntimeError):
    """Base for minutes domain errors."""


class MissingPolishedChaptersError(MinutesError):
    """Raised when no `polished.json` is cached for a run id."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"no cached polished chapters for run {run_id!r}; run "
            f"`transcript polish --run-id {run_id} ...` first."
        )
        self.run_id = run_id


# --- The minutes prompt -------------------------------------------------------


class MinutesResponse(BaseModel):
    """The prompt's reply: the summary callout body and the Discussion Notes body."""

    meeting_summary: str
    discussion_notes: str


class MinutesPrompt(StructuredPrompt[MinutesResponse]):
    template = textwrap.dedent("""\
        You are writing the meeting summary and Discussion Notes for an
        Obsidian meeting note, from this meeting's polished chapters below.
        The person reading this trusts it to tell them what actually
        happened in the room - not what an AI thinks should happen next.

        The single rule that matters most, imperatively: REPORT, NEVER
        PRESCRIBE. Minutes record what was said - facts presented,
        positions taken, decisions made, and questions left open - never
        recommendations, next steps, or imperatives the meeting itself did
        not contain. Do not add a "Recommendations" or "Action Items"
        section, and do not let recommending language ("should",
        "consider", "it would be good to", "needs to") slip into a bullet
        about something that was not actually decided or committed to in
        the meeting.

        An action item appears only when someone actually took it on
        during the meeting, attributed exactly as they said it (e.g. "Rob
        will work through available dates with Steve", "[[Nikki
        Staltari]] - update the CAD body-axis orientation to FRD") - never
        invented, never synthesised from what "should" happen next, and
        never attributed to whoever it was suggested to instead of
        whoever actually said they'd do it. A past-tense or ambiguous
        remark ("I've allocated it to an account") is not a future
        commitment - do not turn it into one.

        When something was left undecided, report it as open, naming the
        alternatives that were actually discussed (e.g. "Long-term
        transport remains open: keep MAVLink, or bridge the internal bus
        through Zenoh / DDS") - never resolve it, and never suggest which
        alternative to pick.

        GROUNDING, imperatively: every bullet must be traceable to
        something a named speaker actually said in the chapters below.
        Never invent or infer a date, a deadline, an owner, an amount, or
        an action item that the transcript does not state, even when it
        seems like the obvious implication - when the chapters leave
        something ambiguous, unstated, or contradictory (e.g. two
        different dates given for the same event), prefer to omit it, or
        report the ambiguity itself, over silently picking one version and
        presenting it as confirmed. Attribute every statement, quote, and
        action to the speaker the chapters actually show saying it - never
        to the other participant, and never fold one speaker's words into
        another's turn. Use the wording and terminology already present in
        the chapters below - do not substitute your own paraphrase for a
        technical term, a name, or a figure that appears there; rephrase
        only the surrounding sentence structure, never the substance.

        Structure the Discussion Notes as nested Markdown bullets, not
        prose: one top-level bullet per topic, broadly following the
        chapter order below (a chapter can produce more than one top-level
        bullet if it genuinely covers more than one topic, and adjacent
        chapters can share one bullet if they are really the same topic),
        with indented sub-bullets underneath for the supporting detail,
        positions, decisions, and attributed actions. Prefer nesting detail
        under a topic bullet over writing paragraphs. Do not include a
        "## Discussion Notes" heading - just the bullets.

        Wikilink names and terms in [[Double Bracket]] form:
        - every attendee name, every time they are meaningfully involved in
          a bullet: {{ attendees | json }}
        - any other term below that plausibly has (or should have) its own
          vault note - people, organisations, named systems, projects, or
          recurring concepts - wherever it is genuinely relevant to what is
          being reported. Do not wikilink a term just because it appears in
          this list, and never wikilink a generic word. Candidate terms:
          {{ lexicon | json }}

        Length should track the meeting, not a target: a short, narrow
        meeting produces fewer bullets, a long, wide-ranging one produces
        more. Never pad to reach a length, and never compress away
        substantive detail just to be brief.

        Also write `meeting_summary`: one sentence of no more than 25
        words stating what the meeting was about or its central outcome,
        followed by one to four short bullets (no more than 25 words each)
        covering the other main topics, decisions, or open questions. The
        same report-never-prescribe rule and the same wikilink rules apply
        here too. Do not wrap it in a callout block (no "> [!summary]" or
        similar) - return plain Markdown, a sentence then bullets; the
        callout syntax is added separately when the note is written.

        Chapters, in order:
        {{ chapters_text }}
        """)
    response_model = MinutesResponse

    attendees: list[str]
    lexicon: list[str]
    chapters_text: str


def _render_chapters(chapters: Sequence[PolishedChapter]) -> str:
    parts: list[str] = []
    for chapter in chapters:
        turns = "\n".join(f"  {turn.speaker}: {turn.text}" for turn in chapter.turns)
        parts.append(f"### {chapter.title}\nSummary: {chapter.summary}\n{turns}")
    return "\n\n".join(parts)


async def generate_minutes(
    chapters: Sequence[PolishedChapter],
    attendees: Sequence[str],
    lexicon: Sequence[str],
    agent: ClaudeAgent,
    spec: AgentSpec | None = None,
) -> MinutesResponse:
    """Generate the meeting summary and Discussion Notes in one call.

    The full set of polished chapters goes into one `StructuredPrompt` call
    - a long meeting that doesn't fit is this stage's STOP condition
    (chunk-and-merge minutes is a quality-sensitive fork: topic bullets can
    span chapters, so a mechanical split would break that), not something
    handled here.
    """

    response, _reply = await agent.run_structured(
        MinutesPrompt(
            attendees=list(attendees),
            lexicon=list(lexicon),
            chapters_text=_render_chapters(chapters),
        ),
        spec,
        stage="minutes",
    )
    return response


# --- run orchestration: cache state around one `minutes` call ---------------


class MinutesResult(BaseModel):
    """On-disk wrapper for `minutes.json`, and what the CLI prints to stdout.

    `MinutesResponse` is already a plain `BaseModel`, so nothing stops
    `RunCache.store` from writing it directly - this wraps it with `run_id`
    for traceability, the same reason `polish.py`'s `PolishResponse` is a
    distinct shape from its own prompt's response model.
    """

    run_id: str
    meeting_summary: str
    discussion_notes: str


async def run_minutes(
    note_path: Path,
    run_id: str,
    *,
    agent: ClaudeAgent,
    vault: VaultClient,
    cache: RunCache,
) -> MinutesResult:
    """Drive one `jake-tools transcript minutes` invocation.

    Loads the cached polished chapters for `run_id` (`MissingPolishedChaptersError`,
    naming `transcript polish` as the prerequisite, if there aren't any),
    builds the same vault lexicon `transcript polish` builds (attendees,
    wikilink targets already in the note, vault note titles) as the
    wikilink candidate set, generates the summary and Discussion Notes in
    one call, stores the result as `minutes.json`, and returns it.
    """

    note = parse_note(note_path)

    polished = cache.load(run_id, POLISHED_CACHE_NAME, PolishedChapterList)
    if polished is None:
        raise MissingPolishedChaptersError(run_id)

    lexicon = build_lexicon(note, vault)
    stage_agent = agent.for_stage("minutes").with_telemetry(
        cache.telemetry_sink(run_id)
    )
    manifest = cache.stage_manifest(
        "minutes",
        inputs={
            "polished": polished.model_dump(mode="json"),
            "note": note.model_dump(mode="json"),
            "lexicon": lexicon,
        },
        config={
            "agent": stage_agent.defaults.model_dump(mode="json"),
            "prompt": MinutesPrompt.template,
            "response_schema": MinutesResponse.model_json_schema(),
        },
    )
    cached = cache.load(run_id, MINUTES_CACHE_NAME, MinutesResult)
    if cached is not None and cache.load_manifest(run_id, "minutes") == manifest:
        stage_agent.record_cache_hit()
        return cached
    response = await generate_minutes(
        polished.chapters, note.attendees, lexicon, stage_agent
    )

    result = MinutesResult(
        run_id=run_id,
        meeting_summary=response.meeting_summary,
        discussion_notes=response.discussion_notes,
    )
    cache.store(run_id, MINUTES_CACHE_NAME, result)
    cache.store_manifest(run_id, manifest)
    return result
