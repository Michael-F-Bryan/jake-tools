"""Polish each chapter's raw dialogue, then check it with a fresh adversarial pass.

This is the stage the whole product hangs on: Michael's stated trust
threshold is that terrible or absent polishing kills trust in the tool
entirely. Raw ASR output is *very* raw - crosstalk shredded into syllable
fragments, split words, filler noise. Two properties are non-negotiable and
both are enforced structurally here, not just by convention:

1. **Polish, then a fresh adversarial fix - never a "check your work" turn.**
   Each chapter gets *two separate* `ClaudeAgent.run_structured` calls: one
   that polishes, and a second, brand-new call (no shared conversation, no
   memory of the first) that reviews the polish adversarially against the
   raw utterances and returns a corrected chapter. Michael's explicit
   requirement, on the grounds that agents are really bad at checking their
   own work. `polish_chapters`/`_polish_one_chapter` never collapse this
   into one call, even under cost/latency pressure - see plan
   008-polish-and-fix.md's STOP conditions.
2. **Meaning preservation is the invariant: rewrite form, never content.**
   No summarising, no dropping substantive statements, no inventing. Both
   prompts state this imperatively, and the fixer is explicitly told to
   hunt for meaning drift as well as surface defects.

Chapters run concurrently (`asyncio.gather` bounded by an
`asyncio.Semaphore(max_concurrency)`), each independently polished against a
lexicon of meeting-specific vocabulary (`build_lexicon`) drawn from the
note's attendees, its wikilinks, and vault note titles - the domain terms an
ASR model is most likely to mishear. One chapter's failure fails the whole
run, naming the chapter: a silently missing chapter is worse than an error.
"""

from __future__ import annotations

import asyncio
import re
import textwrap
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from ..cache_models import CacheEnvelope
from ..claude import AgentSpec, ClaudeAgent, ClaudeAgentError
from ..prompting import StructuredPrompt
from .cache import RunCache, stable_hash
from .chapters import CHAPTERS_CACHE_NAME, ChapterList
from .models import (
    ChapterSpan,
    DroppedSourceTurn,
    PolishedChapter,
    PolishedTurn,
    RawTranscript,
    Utterance,
)
from .note import ParsedNote, human_owned_note_context, parse_note, render_body
from .obsidian import VaultClient
from .speakers import RESOLVED_TRANSCRIPT_CACHE_NAME

# The cache names `transcript polish` stores its outputs under.
POLISHED_CACHE_NAME = "polished"
POLISH_ISSUES_CACHE_NAME = "polish_issues"

_LEXICON_CAP = 300
_WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


class PolishError(RuntimeError):
    """Base for polish domain errors."""


class MissingResolvedTranscriptError(PolishError):
    """Raised when no `resolved_transcript.json` is cached for a run id."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"no cached resolved transcript for run {run_id!r}; run "
            f"`transcript speakers --run-id {run_id} ...` first."
        )
        self.run_id = run_id


class MissingChaptersError(PolishError):
    """Raised when no `chapters.json` is cached for a run id."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"no cached chapters for run {run_id!r}; run "
            f"`transcript chapterise --run-id {run_id}` first."
        )
        self.run_id = run_id


class MissingPolishedChaptersError(PolishError):
    """Raised when `--chapter N` is used but no full `polished.json` exists yet."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"no cached polished chapters for run {run_id!r} to merge --chapter "
            f"into; run `transcript polish --run-id {run_id} ...` for every "
            "chapter first."
        )
        self.run_id = run_id


class InvalidChapterIndexError(PolishError):
    """Raised when `--chapter N` is out of range for the cached chapters."""

    def __init__(self, chapter: int, chapter_count: int) -> None:
        super().__init__(
            f"--chapter {chapter} is out of range: this run has "
            f"{chapter_count} chapter(s) (valid indexes 0-{chapter_count - 1})."
        )
        self.chapter = chapter
        self.chapter_count = chapter_count


class ChapterPolishError(PolishError):
    """Raised when polishing or fixing one chapter fails.

    Names the chapter that failed: one chapter's failure fails the whole
    run rather than silently going missing from the output.
    """

    def __init__(self, chapter_title: str, cause: Exception) -> None:
        super().__init__(f"failed to polish chapter {chapter_title!r}: {cause}")
        self.chapter_title = chapter_title


# --- Step 1: the vault lexicon ------------------------------------------------


def build_lexicon(note: ParsedNote, vault: VaultClient) -> list[str]:
    """Assemble the domain vocabulary the polish prompt repairs mishearings against.

    Sources, in priority order (earlier entries survive dedupe over later
    duplicates): attendee names, wikilink targets appearing anywhere in the
    note body, and vault note titles (a recursive vault-root glob of
    `**/*.md` stems - cheap and good enough; no need for the Obsidian CLI's
    `files` listing).
    Deduped case-insensitively (keeping the first-seen casing) and capped so
    the prompt stays a reasonable size.
    """

    body = render_body(note.sections)
    candidates = [
        *note.attendees,
        *_wikilink_targets(body),
        *_vault_note_titles(vault),
    ]
    return _dedupe(candidates)[:_LEXICON_CAP]


def _wikilink_targets(body: str) -> list[str]:
    return [match.group(1).strip() for match in _WIKILINK_RE.finditer(body)]


def _vault_note_titles(vault: VaultClient) -> list[str]:
    root = vault.vault_root()
    return sorted({path.stem for path in root.glob("**/*.md")})


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        item = raw.strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


# --- Step 2: polish prompt + adversarial fixer prompt ------------------------


class PolishedChapterResponse(BaseModel):
    """The polish prompt's reply: a cleaned-up chapter, form only rewritten."""

    summary: str
    turns: list[PolishedTurn]
    dropped_source_turns: list[DroppedSourceTurn] = Field(default_factory=list)


class ChapterFixResponse(PolishedChapterResponse):
    """The fixer's reply: a corrected chapter, plus the issues it found.

    `issues` is for plan 011's run report - keep entries human-readable
    sentences, not codes. Empty when the polish under review was already
    clean. `summary` inherits `PolishedChapterResponse`'s contract (a
    reader-facing chapter summary) - `_polish_one_chapter` does not trust
    it blindly when `issues` is empty, see its docstring.
    """

    issues: list[str]


class PolishValidationRepairResponse(PolishedChapterResponse):
    """A single bounded repair of a structurally invalid model proposal."""

    issues: list[str]


class PolishPrompt(StructuredPrompt[PolishedChapterResponse]):
    template = textwrap.dedent("""\
        You are polishing one chapter of a raw, unproofread meeting
        transcript for a human reader.

        The single invariant that matters most: rewrite FORM, never
        CONTENT. Every substantive statement in the raw dialogue below must
        survive in your output, worded clearly - never summarised, never
        dropped, never invented. If you are unsure whether something is
        substantive, keep it.

        Rules:
        - Remove filler words ("um", "uh", "you know", false starts) that
          carry no content.
        - Merge adjacent turns from the same speaker into one turn.
        - The raw utterances below are ordered by when they started, so
          genuine cross-talk (two speakers overlapping in time) can appear
          interleaved rather than as a clean back-and-forth - untangle it
          into an intelligible sequence of turns without losing what either
          speaker said.
        - Repair ASR fragmentation: reattach split syllables or words to
          the sentence they belong to (e.g. a dangling fragment like "ocity"
          next to "vel" is one mis-split word, "velocity").
        - A word that looks out of place - a plausible mishearing - should
          be checked against this lexicon of names and terms from this
          meeting's context, and corrected if it matches:
          {{ lexicon | json }}
        - Every speaker in your output must be one of the speakers present
          in the raw utterances below, or "Unknown" - never invent a
          speaker that isn't in that set.
        - Turns must stay in chronological order.
        - Every raw utterance index must appear exactly once in
          `source_turn_indices` or in `dropped_source_turns`. Drops are only
          filler-only backchannels, immediate duplicate ASR fragments, or
          unintelligible fragments with no recoverable substance.
        - Preserve the raw speaker for every source index: Unknown must remain
          Unknown and a named speaker must never become Unknown.
        - A turn may merge only source indices belonging to the same speaker.
        - `summary` is 1-3 sentences describing what this chapter covered,
          written for a callout at the top of the chapter - not a
          transcript of who said what.

        Chapter: {{ chapter_title }}

        Raw utterances (speaker: text), in order:
        {{ raw_lines }}
        """)
    response_model = PolishedChapterResponse

    chapter_title: str
    lexicon: list[str]
    raw_lines: str


class ChapterFixPrompt(StructuredPrompt[ChapterFixResponse]):
    template = textwrap.dedent("""\
        You are reviewing someone else's polish of one chapter of a raw
        meeting transcript. You did not write this polish and had no part
        in producing it - review it adversarially, the way a skeptical
        editor checks another editor's work, not the way an author checks
        their own.

        Hunt for:
        - surviving fragments or split words the polish failed to repair
        - garbled or still-interleaved crosstalk
        - a term in the polish that doesn't match this lexicon of meeting
          names and terms and looks like an uncorrected mishearing (e.g. an
          unrelated real company name like "Suncorp" surviving when the
          lexicon contains "Sunfish" is exactly this failure - fix it):
          {{ lexicon | json }}
        - meaning drift: any statement in the polish that the raw text
          does not support, or any substantive statement in the raw text
          that the polish dropped or waters down

        The lexicon is authoritative for proper nouns. If the polish
        already corrected a word to match a lexicon term, that correction
        is settled - do NOT revert it back to the raw wording, even if the
        raw wording is right there in the utterances below and looks like
        a plausible word on its own. A raw ASR word merely existing in the
        transcript is not evidence it is correct; the lexicon is. The same
        rule runs the other way: never introduce, restore, or invent a
        proper noun (a person, company, or place) that appears nowhere in
        the lexicon above. A name that occurs only once in the raw
        utterances, with no corroboration anywhere else in the transcript,
        is more likely ASR noise than a real person - do not promote it to
        a named individual in your corrected output.

        Meaning preservation is the invariant that matters most: the
        polish must rewrite form, never content. Fix whatever you find and
        return the corrected chapter IN FULL (not a diff), plus one plain
        English sentence per issue you fixed, for a run report a human
        will read (an empty list if the polish under review was already
        clean - do not invent issues to have something to report).

        Provenance is also mandatory: Every raw utterance index must appear
        exactly once in `source_turn_indices` or in the complete
        `dropped_source_turns` ledger. Preserve the raw speaker for every
        source index, keep `Unknown` as `Unknown`, and never merge turns
        across speakers. A merged output turn may cite non-adjacent source
        indices only when every intervening non-overlapping source index is
        present in the dropped ledger; otherwise split the turn.

        `summary` must always read exactly like the polish's own summary
        field: 1-3 sentences describing what this chapter covered, for a
        reader's callout. Never write about this review itself - not the
        lexicon, not what you changed or why, not phrases like "reviewed
        the polish" or "returned unchanged" - that commentary belongs only
        in `issues`, never in `summary`. If you found nothing to fix,
        return the polish's summary verbatim.

        Chapter: {{ chapter_title }}

        Raw utterances (speaker: text), in order - the ground truth for
        what was actually said:
        {{ raw_lines }}

        The polish under review:
        Summary: {{ polished_summary }}
        Turns:
        {{ polished_lines }}
        Dropped source-turn ledger:
        {{ polished_dropped }}
        """)
    response_model = ChapterFixResponse

    chapter_title: str
    lexicon: list[str]
    raw_lines: str
    polished_summary: str
    polished_lines: str
    polished_dropped: str = "[]"


class PolishValidationRepairPrompt(StructuredPrompt[PolishValidationRepairResponse]):
    template = textwrap.dedent("""\
        Repair an invalid proposed polished chapter. Return the corrected full
        chapter, not a diff, and one plain-English issue sentence per repair.

        The raw indexed and timestamped turns below are the only source of
        truth. Preserve exact source partition, source speaker attribution,
        Unknown attribution, and the existing dropped-turn ledger rules. Do
        not invent, drop, duplicate, or merge source turns unless the
        deterministic errors below prove the proposed chapter violated a
        rule and the raw evidence supports the correction.

        Chapter: {{ chapter_title }}

        Raw indexed and timestamped turns:
        {{ raw_lines }}

        Invalid proposed chapter:
        Summary: {{ invalid_summary }}
        Turns:
        {{ invalid_lines }}
        Dropped source-turn ledger:
        {{ invalid_dropped }}

        Exact deterministic validation errors:
        {{ validation_errors | json }}

        Always return the complete `dropped_source_turns` ledger as a
        structured field, even when it is unchanged or empty. Do not merely
        describe ledger changes in `issues`. A merged output turn may cite
        non-adjacent source indices only when every intervening non-overlapping
        source index appears in `dropped_source_turns`; otherwise split the
        output turn.

        Return the complete corrected chapter. `summary` must remain a
        reader-facing chapter summary, never commentary about this repair.
        """)
    response_model = PolishValidationRepairResponse

    chapter_title: str
    raw_lines: str
    invalid_summary: str
    invalid_lines: str
    invalid_dropped: str
    validation_errors: list[str]


def validate_polished_chapter(
    chapter: PolishedChapter,
    source_utterances: Sequence[Utterance],
    *,
    source_indices: Sequence[int] | None = None,
) -> PolishedChapter:
    """Validate exact source coverage and attribution before promotion."""
    expected = list(source_indices or range(len(source_utterances)))
    source_by_index = dict(zip(expected, source_utterances, strict=True))
    dropped = [item.source_turn_index for item in chapter.dropped_source_turns]
    if len(dropped) != len(set(dropped)):
        raise ValueError("polish dropped-source provenance is duplicated")
    if any(index not in source_by_index for index in dropped):
        raise ValueError("polish dropped-source provenance index is out of range")
    dropped_set = set(dropped)
    covered: list[int] = []
    genuine_overlap_reflow = False
    for turn in chapter.turns:
        if not turn.source_turn_indices:
            raise ValueError("polish provenance is missing from an output turn")
        indices = turn.source_turn_indices
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("polish provenance is not chronological")
        for previous, current in zip(indices, indices[1:], strict=False):
            intervening = [
                index
                for index in range(previous + 1, current)
                if index not in dropped_set
            ]
            if any(
                not any(
                    source_by_index[index].start < source_by_index[cited].end
                    and source_by_index[cited].start < source_by_index[index].end
                    for cited in indices
                )
                for index in intervening
            ):
                raise ValueError(
                    "a polished turn may merge only adjacent source turns unless "
                    "intervening turns are explicitly dropped or genuinely overlap"
                )
            if intervening:
                genuine_overlap_reflow = True
        for index in indices:
            if index not in source_by_index:
                raise ValueError(f"polish provenance index {index} is out of range")
            if index in covered:
                raise ValueError(f"polish provenance index {index} is duplicated")
            if turn.speaker != source_by_index[index].speaker:
                raise ValueError(
                    f"polish output speaker {turn.speaker!r} does not match source speaker"
                )
            covered.append(index)
    if set(covered) & set(dropped) or set(covered) | set(dropped) != set(expected):
        raise ValueError("polish provenance does not form an exact source partition")
    if covered != sorted(covered) and not genuine_overlap_reflow:
        raise ValueError("polish provenance is not chronological")
    return chapter


def _render_utterances(utterances: Sequence[Utterance], *, start_index: int = 0) -> str:
    return "\n".join(
        f"{start_index + index} | {u.start:.3f}-{u.end:.3f} | {u.speaker}: {u.text}"
        for index, u in enumerate(utterances)
    )


def _render_turns(turns: Sequence[PolishedTurn]) -> str:
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)


def _render_response_turns(response: PolishedChapterResponse) -> str:
    return "\n".join(
        f"{turn.speaker} [{','.join(str(index) for index in turn.source_turn_indices)}]: "
        f"{turn.text}"
        for turn in response.turns
    )


def _chapter_from_response(
    span: ChapterSpan, response: PolishedChapterResponse
) -> PolishedChapter:
    return PolishedChapter(
        title=span.title,
        start_seconds=span.start_seconds,
        summary=response.summary,
        turns=response.turns,
        dropped_source_turns=response.dropped_source_turns,
    )


async def _validate_or_repair_response(
    response: PolishedChapterResponse,
    span: ChapterSpan,
    utterances: Sequence[Utterance],
    *,
    raw_lines: str,
    agent: ClaudeAgent,
    spec: AgentSpec | None,
    scope: str,
    force_validation: bool,
    repair_allowed: bool,
) -> tuple[PolishedChapterResponse, list[str], bool]:
    has_provenance = any(turn.source_turn_indices for turn in response.turns) or bool(
        response.dropped_source_turns
    )
    if not force_validation and not has_provenance:
        return response, [], False
    chapter = _chapter_from_response(span, response)
    try:
        validate_polished_chapter(
            chapter,
            utterances,
            source_indices=range(span.start_utterance, span.end_utterance + 1),
        )
        return response, [], False
    except ValueError as error:
        if not repair_allowed:
            raise
        repair_response, _reply = await agent.for_stage(
            "polish-validation-repair"
        ).run_structured(
            PolishValidationRepairPrompt(
                chapter_title=span.title,
                raw_lines=raw_lines,
                invalid_summary=response.summary,
                invalid_lines=_render_response_turns(response),
                invalid_dropped=response.dropped_source_turns.__repr__(),
                validation_errors=[str(error)],
            ),
            spec,
            stage="polish-validation-repair",
            scope=scope,
        )
        if "dropped_source_turns" not in repair_response.model_fields_set:
            repaired_indices = {
                index
                for turn in repair_response.turns
                for index in turn.source_turn_indices
            }
            repair_response = repair_response.model_copy(
                update={
                    "dropped_source_turns": [
                        item
                        for item in response.dropped_source_turns
                        if item.source_turn_index not in repaired_indices
                    ]
                }
            )
        repaired_chapter = _chapter_from_response(span, repair_response)
        validate_polished_chapter(
            repaired_chapter,
            utterances,
            source_indices=range(span.start_utterance, span.end_utterance + 1),
        )
        return repair_response, repair_response.issues, True


# --- Step 3: orchestration ----------------------------------------------------


async def _polish_one_chapter(
    span: ChapterSpan,
    utterances: Sequence[Utterance],
    *,
    lexicon: Sequence[str],
    agent: ClaudeAgent,
    spec: AgentSpec | None,
) -> tuple[PolishedChapter, list[str]]:
    """Polish one chapter, then fix it with a fresh, independent agent call.

    The fixer call is a brand-new `run_structured` invocation - never a
    follow-up turn on the polisher's conversation - and its corrected
    turns (not the polisher's) are what win.

    `summary` is the one field this deliberately does NOT always take from
    the fixer: when `fix_response.issues` is empty, the fixer found
    nothing to fix, so there is no reason its `summary` should differ from
    the polish's own - and a structural guard against a leaked-commentary
    `summary` (the reviewer's own prose about the review shipping as the
    chapter summary, e.g. "Reviewed the polish against the raw transcript
    ... so the chapter is returned unchanged") is worth more here than
    trusting the model to have followed the prompt's instruction not to
    write it in the first place. When the fixer *did* find something to
    fix, its `summary` is trusted, since it may have needed to change to
    reflect a corrected turn.
    """

    raw_lines = _render_utterances(utterances, start_index=span.start_utterance)
    lexicon_list = list(lexicon)
    scope = (
        f"chapter:{span.start_utterance}-{span.end_utterance}:"
        f"{stable_hash(raw_lines)[:16]}"
    )

    polish_response, _reply = await agent.for_stage("polish-generation").run_structured(
        PolishPrompt(
            chapter_title=span.title,
            lexicon=lexicon_list,
            raw_lines=raw_lines,
        ),
        spec,
        stage="polish-generation",
        scope=scope,
    )
    (
        polish_response,
        polish_repair_issues,
        _polish_repair_used,
    ) = await _validate_or_repair_response(
        polish_response,
        span,
        utterances,
        raw_lines=raw_lines,
        agent=agent,
        spec=spec,
        scope=scope,
        force_validation=False,
        repair_allowed=True,
    )
    polish_has_provenance = any(
        turn.source_turn_indices for turn in polish_response.turns
    ) or bool(polish_response.dropped_source_turns)

    fix_response, _reply = await agent.for_stage("polish-review").run_structured(
        ChapterFixPrompt(
            chapter_title=span.title,
            lexicon=lexicon_list,
            raw_lines=raw_lines,
            polished_summary=polish_response.summary,
            polished_lines=_render_response_turns(polish_response),
            polished_dropped=polish_response.dropped_source_turns.__repr__(),
        ),
        spec,
        stage="polish-review",
        scope=scope,
    )
    (
        fix_response,
        _fix_repair_issues,
        _fix_repair_used,
    ) = await _validate_or_repair_response(
        fix_response,
        span,
        utterances,
        raw_lines=raw_lines,
        agent=agent,
        spec=spec,
        scope=scope,
        force_validation=polish_has_provenance,
        repair_allowed=True,
    )

    fix_issues = getattr(fix_response, "issues", [])
    summary = fix_response.summary if fix_issues else polish_response.summary
    chapter = _chapter_from_response(span, fix_response).model_copy(
        update={"summary": summary}
    )
    issues = [
        f"{span.title}: {issue}" for issue in (*polish_repair_issues, *fix_issues)
    ]
    return chapter, issues


async def polish_chapters(
    transcript: RawTranscript,
    spans: Sequence[ChapterSpan],
    note: ParsedNote,
    vault: VaultClient,
    agent: ClaudeAgent,
    spec: AgentSpec | None = None,
    *,
    max_concurrency: int = 4,
) -> tuple[list[PolishedChapter], list[str]]:
    """Polish and fix every chapter in `spans`, bounded to `max_concurrency` at once.

    Builds the lexicon once and shares it across every chapter. Chapters run
    concurrently under an `asyncio.Semaphore(max_concurrency)` - each
    chapter still does its own two *sequential* calls (polish, then fix),
    the semaphore just bounds how many chapters are in flight at once.
    Returns the polished chapters in `spans` order and every fixer issue,
    prefixed with its chapter's title, for the run report. One chapter's
    failure (`ChapterPolishError`, naming the chapter) fails the whole call
    - a silently missing chapter is worse than an error.
    """

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    lexicon = build_lexicon(note, vault)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(span: ChapterSpan) -> tuple[PolishedChapter, list[str]]:
        utterances = transcript.utterances[
            span.start_utterance : span.end_utterance + 1
        ]
        async with semaphore:
            try:
                return await _polish_one_chapter(
                    span, utterances, lexicon=lexicon, agent=agent, spec=spec
                )
            except ClaudeAgentError as exc:
                raise ChapterPolishError(span.title, exc) from exc

    results = await asyncio.gather(*(run_one(span) for span in spans))
    chapters = [chapter for chapter, _issues in results]
    issues = [issue for _chapter, chapter_issues in results for issue in chapter_issues]
    return chapters, issues


# --- Step 4: run orchestration around one `transcript polish` invocation ----


class PolishedChapterList(CacheEnvelope):
    """On-disk wrapper for `polished.json` (`RunCache.store` needs a `BaseModel`)."""

    chapters: list[PolishedChapter]


class PolishIssueList(CacheEnvelope):
    """On-disk wrapper for `polish_issues.json`."""

    issues: list[str]


class PolishResponse(BaseModel):
    """The JSON summary `jake-tools transcript polish` prints."""

    run_id: str
    chapter_count: int
    issue_count: int
    issues: list[str]


async def run_polish(
    note_path: Path,
    run_id: str,
    *,
    chapter: int | None,
    max_concurrency: int,
    agent: ClaudeAgent,
    vault: VaultClient,
    cache: RunCache,
) -> PolishResponse:
    """Drive one `jake-tools transcript polish` invocation.

    Loads the cached resolved transcript and chapters (naming the
    prerequisite command if either is missing), then either polishes every
    chapter or - when `chapter` is given - re-polishes just that one
    chapter (its 0-based index into `chapters.json`) and merges the result
    into the existing `polished.json`/`polish_issues.json`, for an
    agent-driven retry of a single bad chapter. `--chapter` requires a
    prior full run: there is nothing sensible to merge a single chapter
    into otherwise.

    Stores `polished.json` (`list[PolishedChapter]`) and
    `polish_issues.json` through the typed cache path (`RunCache.store`),
    so the on-disk filenames are exactly those `.json` names.
    """

    note = parse_note(note_path)

    transcript = cache.load(run_id, RESOLVED_TRANSCRIPT_CACHE_NAME, RawTranscript)
    if transcript is None:
        raise MissingResolvedTranscriptError(run_id)

    chapter_list = cache.load(run_id, CHAPTERS_CACHE_NAME, ChapterList)
    if chapter_list is None:
        raise MissingChaptersError(run_id)
    spans = chapter_list.chapters

    stage_agent = agent.with_telemetry(cache.telemetry_sink(run_id))
    lexicon = build_lexicon(note, vault)
    manifest = cache.stage_manifest(
        "polish",
        inputs={
            "resolved_transcript": transcript.model_dump(mode="json"),
            "chapters": chapter_list.model_dump(mode="json"),
            "human_context": human_owned_note_context(note),
            "lexicon": lexicon,
        },
        input_hashes={
            "resolved_transcript": stable_hash(transcript.model_dump(mode="json")),
            "chapters": stable_hash(chapter_list.model_dump(mode="json")),
            "human_context": stable_hash(human_owned_note_context(note)),
            "lexicon": stable_hash(lexicon),
        },
        config={
            "agent": stage_agent.defaults.model_dump(mode="json"),
            "polish_prompt": PolishPrompt.template,
            "review_prompt": ChapterFixPrompt.template,
            "validation_repair_prompt": PolishValidationRepairPrompt.template,
            "polish_schema": PolishedChapterResponse.model_json_schema(),
            "review_schema": ChapterFixResponse.model_json_schema(),
            "validation_repair_schema": PolishValidationRepairResponse.model_json_schema(),
        },
    )

    current_manifest = cache.load_manifest(run_id, "polish")
    if chapter is None:
        existing = cache.load(run_id, POLISHED_CACHE_NAME, PolishedChapterList)
        existing_issues = cache.load(run_id, POLISH_ISSUES_CACHE_NAME, PolishIssueList)
        if (
            existing is not None
            and existing_issues is not None
            and cache.manifest_matches(current_manifest, manifest)
            and current_manifest is not None
            and current_manifest.output_hash
            == stable_hash(existing.model_dump(mode="json"))
        ):
            stage_agent.for_stage("polish-generation").record_cache_hit()
            stage_agent.for_stage("polish-review").record_cache_hit()
            return PolishResponse(
                run_id=run_id,
                chapter_count=len(existing.chapters),
                issue_count=len(existing_issues.issues),
                issues=existing_issues.issues,
            )
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
        new_chapters, new_issues = await polish_chapters(
            transcript, spans, note, vault, stage_agent, max_concurrency=max_concurrency
        )
        final_chapters = new_chapters
        final_issues = new_issues
    else:
        if chapter < 0 or chapter >= len(spans):
            raise InvalidChapterIndexError(chapter, len(spans))

        existing = cache.load(run_id, POLISHED_CACHE_NAME, PolishedChapterList)
        if (
            existing is None
            or len(existing.chapters) != len(spans)
            or not cache.manifest_matches(current_manifest, manifest)
            or current_manifest is None
            or current_manifest.output_hash
            != stable_hash(existing.model_dump(mode="json"))
        ):
            raise MissingPolishedChaptersError(run_id)

        existing_issue_list = cache.load(
            run_id, POLISH_ISSUES_CACHE_NAME, PolishIssueList
        )
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
        new_chapters, new_issues = await polish_chapters(
            transcript,
            [spans[chapter]],
            note,
            vault,
            stage_agent,
            max_concurrency=max_concurrency,
        )

        final_chapters = list(existing.chapters)
        final_chapters[chapter] = new_chapters[0]

        existing_issues = (
            existing_issue_list.issues if existing_issue_list is not None else []
        )
        title_prefix = f"{spans[chapter].title}: "
        kept_issues = [
            issue for issue in existing_issues if not issue.startswith(title_prefix)
        ]
        final_issues = kept_issues + new_issues

    cache.store(
        run_id, POLISHED_CACHE_NAME, PolishedChapterList(chapters=final_chapters)
    )
    cache.store(run_id, POLISH_ISSUES_CACHE_NAME, PolishIssueList(issues=final_issues))
    cache.store_manifest(
        run_id,
        manifest.model_copy(
            update={
                "output_hash": stable_hash(
                    PolishedChapterList(chapters=final_chapters).model_dump(mode="json")
                )
            }
        ),
    )

    return PolishResponse(
        run_id=run_id,
        chapter_count=len(final_chapters),
        issue_count=len(final_issues),
        issues=final_issues,
    )
