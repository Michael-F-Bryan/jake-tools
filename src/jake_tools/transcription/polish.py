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

from ..claude import AgentSpec, ClaudeAgent, ClaudeAgentError
from ..prompting import StructuredPrompt
from .cache import RunCache
from .chapters import CHAPTERS_CACHE_NAME, ChapterList
from .models import ChapterSpan, PolishedChapter, PolishedTurn, RawTranscript, Utterance
from .note import ParsedNote, parse_note, render_body
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
    note body, and vault note titles (a vault-root glob of `*.md` stems -
    cheap and good enough; no need for the Obsidian CLI's `files` listing).
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


class ChapterFixResponse(PolishedChapterResponse):
    """The fixer's reply: a corrected chapter, plus the issues it found.

    `issues` is for plan 011's run report - keep entries human-readable
    sentences, not codes. Empty when the polish under review was already
    clean.
    """

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
          names and terms and looks like an uncorrected mishearing:
          {{ lexicon | json }}
        - meaning drift: any statement in the polish that the raw text
          does not support, or any substantive statement in the raw text
          that the polish dropped or waters down

        Meaning preservation is the invariant that matters most: the
        polish must rewrite form, never content. Fix whatever you find and
        return the corrected chapter IN FULL (not a diff), plus one plain
        English sentence per issue you fixed, for a run report a human
        will read (an empty list if the polish under review was already
        clean - do not invent issues to have something to report).

        Chapter: {{ chapter_title }}

        Raw utterances (speaker: text), in order - the ground truth for
        what was actually said:
        {{ raw_lines }}

        The polish under review:
        Summary: {{ polished_summary }}
        Turns:
        {{ polished_lines }}
        """)
    response_model = ChapterFixResponse

    chapter_title: str
    lexicon: list[str]
    raw_lines: str
    polished_summary: str
    polished_lines: str


def _render_utterances(utterances: Sequence[Utterance]) -> str:
    return "\n".join(f"{u.speaker}: {u.text}" for u in utterances)


def _render_turns(turns: Sequence[PolishedTurn]) -> str:
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)


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
    output (not the polisher's) is what wins.
    """

    raw_lines = _render_utterances(utterances)
    lexicon_list = list(lexicon)

    polish_response, _reply = await agent.run_structured(
        PolishPrompt(
            chapter_title=span.title,
            lexicon=lexicon_list,
            raw_lines=raw_lines,
        ),
        spec,
    )

    fix_response, _reply = await agent.run_structured(
        ChapterFixPrompt(
            chapter_title=span.title,
            lexicon=lexicon_list,
            raw_lines=raw_lines,
            polished_summary=polish_response.summary,
            polished_lines=_render_turns(polish_response.turns),
        ),
        spec,
    )

    chapter = PolishedChapter(
        title=span.title,
        start_seconds=span.start_seconds,
        summary=fix_response.summary,
        turns=fix_response.turns,
    )
    issues = [f"{span.title}: {issue}" for issue in fix_response.issues]
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


class PolishedChapterList(BaseModel):
    """On-disk wrapper for `polished.json` (`RunCache.store` needs a `BaseModel`)."""

    chapters: list[PolishedChapter]


class PolishIssueList(BaseModel):
    """On-disk wrapper for `polish_issues.json`."""

    issues: list[str] = Field(default_factory=list)


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

    if chapter is None:
        new_chapters, new_issues = await polish_chapters(
            transcript, spans, note, vault, agent, max_concurrency=max_concurrency
        )
        final_chapters = new_chapters
        final_issues = new_issues
    else:
        if chapter < 0 or chapter >= len(spans):
            raise InvalidChapterIndexError(chapter, len(spans))

        existing = cache.load(run_id, POLISHED_CACHE_NAME, PolishedChapterList)
        if existing is None or len(existing.chapters) != len(spans):
            raise MissingPolishedChaptersError(run_id)

        new_chapters, new_issues = await polish_chapters(
            transcript,
            [spans[chapter]],
            note,
            vault,
            agent,
            max_concurrency=max_concurrency,
        )

        final_chapters = list(existing.chapters)
        final_chapters[chapter] = new_chapters[0]

        existing_issue_list = cache.load(
            run_id, POLISH_ISSUES_CACHE_NAME, PolishIssueList
        )
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

    return PolishResponse(
        run_id=run_id,
        chapter_count=len(final_chapters),
        issue_count=len(final_issues),
        issues=final_issues,
    )
