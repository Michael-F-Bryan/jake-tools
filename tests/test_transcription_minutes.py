"""Behaviour of minutes generation (`transcription/minutes.py`) and the
`jake-tools transcript minutes` CLI command.

Per the CLI-options memo (rule 4) and following `tests/test_transcription_polish.py`,
this module tests two different things at two different layers:

- **Plumbing/contract tests (fast, run by default).** A fake `ClaudeAgent`
  (`ScriptedQuery`, the same pattern as `test_transcription_polish.py`/
  `test_transcription_chapters.py`) proves the rendered prompt actually
  carries the load-bearing instructions - the report-never-prescribe rule,
  the attendees, and the chapter content - so a silent prompt regression
  (e.g. that sentence quietly disappearing during a refactor) fails a test
  even though a fake can only ever replay what the test already wrote down.
  They also prove `run_minutes`'s plumbing: the cached-prerequisite check,
  storage under the literal `minutes.json` name, and the CLI's flag
  parsing/delegation/error mapping.
- **The `@pytest.mark.slow` real-model test.** Everything above cannot
  prove the *prompt* actually gets a real model to report rather than
  prescribe - that's this file's one slow test, skipped by default
  (pytest-skip-slow; run with `uv run pytest --slow -k live`), which makes
  a real `claude-sonnet-5` call over a small, deliberately-constructed
  fixture and checks observable properties of the reply.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from click.testing import CliRunner

from jake_tools.claude import AgentSpec, ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.chapters import CHAPTERS_CACHE_NAME, ChapterList
from jake_tools.transcription.minutes import (
    MINUTES_CACHE_NAME,
    MinutesResult,
    MissingPolishedChaptersError,
    generate_minutes,
    run_minutes,
)
from jake_tools.transcription.models import (
    ChapterSpan,
    PolishedChapter,
    PolishedTurn,
    RawTranscript,
    Utterance,
)
from jake_tools.transcription.note import NoteSection, ParsedNote
from jake_tools.transcription.obsidian import VaultClient
from jake_tools.transcription.polish import (
    POLISHED_CACHE_NAME,
    PolishedChapterList,
    run_polish,
)
from jake_tools.transcription.speakers import RESOLVED_TRANSCRIPT_CACHE_NAME

FIXTURES = Path(__file__).parent / "fixtures"

# `jake_tools.cli`'s __init__ rebinds the name `transcript` to the Click
# group, shadowing the submodule (see `test_transcript_cli.py`) — fetch the
# actual module via importlib to monkeypatch its `run_minutes` binding.
transcript_cli = importlib.import_module("jake_tools.cli.transcript")


# --- fakes -------------------------------------------------------------------


def _structured_result(payload: object) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id="session-1",
        total_cost_usd=0.01,
        usage=None,
        result=None,
        structured_output=payload,
        errors=None,
    )


class ScriptedQuery:
    """A fake `run_query` that replays a different structured payload per call."""

    def __init__(self, *payloads: object) -> None:
        self._payloads = list(payloads)
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        index = len(self.calls)
        self.calls.append((prompt, options))
        payload = self._payloads[index]

        async def stream() -> AsyncIterator[Message]:
            yield _structured_result(payload)

        return stream()


class FakeVaultClient:
    """Fake `VaultClient`: `vault_root` points at a real tmp_path tree so
    `build_lexicon`'s glob has real files to find; `resolve_embed` is
    unused by minutes.py and left unimplemented."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def vault_root(self) -> Path:
        return self._root

    def resolve_embed(self, target: str) -> Path:  # pragma: no cover - unused here
        raise NotImplementedError


def _empty_vault(tmp_path: Path) -> VaultClient:
    root = tmp_path / "empty-vault"
    root.mkdir(exist_ok=True)
    return FakeVaultClient(root)


def _note(*, attendees: list[str], body: str = "") -> ParsedNote:
    return ParsedNote(
        path="fake-note.md",
        frontmatter={},
        context="meeting",
        attendees=attendees,
        diarisation_hints=[],
        embeds=[],
        sections=[NoteSection(heading=None, level=0, body=body)],
    )


def _chapter(
    title: str, *, summary: str, turns: list[tuple[str, str]]
) -> PolishedChapter:
    return PolishedChapter(
        title=title,
        start_seconds=0.0,
        summary=summary,
        turns=[PolishedTurn(speaker=speaker, text=text) for speaker, text in turns],
    )


# --- generate_minutes: the prompt's load-bearing instructions -----------------


async def test_minutes_prompt_states_report_dont_prescribe_rule_imperatively() -> None:
    chapters = [_chapter("Opening", summary="Kickoff", turns=[("Ada", "hello")])]
    fake = ScriptedQuery({"meeting_summary": "s", "discussion_notes": "- bullet"})
    agent = ClaudeAgent(run_query=fake)

    await generate_minutes(chapters, ["Ada"], [], agent)

    # Normalise whitespace so a template rewrap can't break a substring
    # check that happens to straddle a line break - the template wraps
    # prose at ~70 columns, and the exact wrap points are an implementation
    # detail these tests should not be coupled to.
    prompt = " ".join(fake.calls[0][0].split())
    assert "REPORT, NEVER PRESCRIBE" in prompt
    # The prompt must also forbid the exact failure Michael named: an
    # invented recommendations/action-items section.
    assert "Recommendations" in prompt
    assert "Action Items" in prompt


async def test_minutes_prompt_carries_attendees_and_chapter_content() -> None:
    chapters = [
        _chapter(
            "Roadmap kickoff",
            summary="The team scoped the autonomy roadmap.",
            turns=[
                ("Ada Lovelace", "Let's get started on the roadmap."),
                ("Grace Hopper", "Sounds good, I'll take the first item."),
            ],
        )
    ]
    fake = ScriptedQuery({"meeting_summary": "s", "discussion_notes": "- bullet"})
    agent = ClaudeAgent(run_query=fake)

    await generate_minutes(
        chapters,
        attendees=["Ada Lovelace", "Grace Hopper"],
        lexicon=["Autonomy Stack"],
        agent=agent,
    )

    prompt = fake.calls[0][0]
    # Attendees are named in the prompt (the "always wikilink" set).
    assert "Ada Lovelace" in prompt
    assert "Grace Hopper" in prompt
    # The lexicon is passed through as the wikilink candidate set.
    assert "Autonomy Stack" in prompt
    # The chapter's own content actually reaches the model.
    assert "Roadmap kickoff" in prompt
    assert "The team scoped the autonomy roadmap." in prompt
    assert "Let's get started on the roadmap." in prompt
    assert "Sounds good, I'll take the first item." in prompt


async def test_minutes_prompt_states_action_items_only_as_said_and_open_questions_reported_open() -> (
    None
):
    chapters = [_chapter("Opening", summary="s", turns=[("Ada", "hello")])]
    fake = ScriptedQuery({"meeting_summary": "s", "discussion_notes": "- bullet"})
    agent = ClaudeAgent(run_query=fake)

    await generate_minutes(chapters, ["Ada"], [], agent)

    # Normalise whitespace so a template rewrap can't break a substring
    # check that happens to straddle a line break.
    prompt = " ".join(fake.calls[0][0].split())
    assert "attributed exactly as they said it" in prompt
    assert "never invented, never synthesised" in prompt
    assert "report it as open, naming the" in prompt


async def test_minutes_prompt_states_the_grounding_rule_against_inferred_facts() -> (
    None
):
    """Pins the grounding rule the acceptance review's F4/F5/F8/F19 findings
    need: a real run inverted a date (flipping which financial year a
    deduction fell in), invented an action item for the wrong person from a
    garbled past-tense line, and stated a tax conclusion as confirmed when
    the raw dialogue was actually unresolved. The rendered prompt must
    forbid all of that, imperatively, not just discourage it."""

    chapters = [_chapter("Opening", summary="s", turns=[("Ada", "hello")])]
    fake = ScriptedQuery({"meeting_summary": "s", "discussion_notes": "- bullet"})
    agent = ClaudeAgent(run_query=fake)

    await generate_minutes(chapters, ["Ada"], [], agent)

    prompt = " ".join(fake.calls[0][0].split())
    assert "GROUNDING" in prompt
    assert "Never invent or infer a date, a deadline, an owner, an amount" in prompt
    assert "prefer to omit it, or report the ambiguity itself" in prompt
    assert (
        "never to the other participant, and never fold one speaker's "
        "words into another's turn" in prompt
    )


async def test_generate_minutes_returns_the_agents_structured_reply() -> None:
    chapters = [_chapter("Opening", summary="s", turns=[("Ada", "hello")])]
    fake = ScriptedQuery(
        {
            "meeting_summary": "One sentence.",
            "discussion_notes": "- A bullet\n  - detail",
        }
    )
    agent = ClaudeAgent(run_query=fake)

    response = await generate_minutes(chapters, ["Ada"], [], agent)

    assert response.meeting_summary == "One sentence."
    assert response.discussion_notes == "- A bullet\n  - detail"
    assert len(fake.calls) == 1  # one StructuredPrompt call, not chunk-and-merge


# --- seam: run_polish's fixer output, not its pre-fix polish, reaches minutes -


async def test_minutes_prompt_is_built_from_the_fixers_output_not_the_pre_fix_polish(
    tmp_path: Path,
) -> None:
    """Regression for the acceptance review's F9 ("the minutes stage
    re-introduces content the fixer removed"): runs the real `run_polish`
    (polish, then a fresh adversarial fix call) followed by the real
    `run_minutes`, through `RunCache` exactly as `pipeline.py` chains them,
    and inspects the actual prompt the minutes stage sent to the model.

    The scripted polish deliberately writes the wrong, unrelated company
    name "Suncorp"; the scripted fix call corrects it to "Sunfish" (the
    shape of the acceptance run's F2/F9 findings). If minutes were built
    from the polisher's pre-fix output - or from anything other than what
    `run_polish` actually stored in `polished.json` - "Suncorp" would reach
    the minutes prompt instead of the fixer's correction.
    """
    cache = RunCache(tmp_path / "cache")
    cache.store(
        "run-1",
        RESOLVED_TRANSCRIPT_CACHE_NAME,
        RawTranscript(
            clips=[],
            utterances=[
                Utterance(
                    start=0.0,
                    end=2.0,
                    speaker="Ada Lovelace",
                    text="a dividend from Suncorp to the trust",
                )
            ],
            audio_sha256=None,
        ),
    )
    cache.store(
        "run-1",
        CHAPTERS_CACHE_NAME,
        ChapterList(
            chapters=[
                ChapterSpan(
                    title="Dividends",
                    start_utterance=0,
                    end_utterance=0,
                    start_seconds=0.0,
                )
            ]
        ),
    )
    # One chapter -> polish call, then fix call, in that deterministic
    # order (no concurrency to race with a single chapter); then one more
    # call for minutes. `ScriptedQuery` replays payloads by call order.
    query = ScriptedQuery(
        {
            "summary": "Discussed a dividend.",
            "turns": [
                {
                    "speaker": "Ada Lovelace",
                    "text": "a dividend from Suncorp to the trust",
                }
            ],
        },
        {
            "summary": "Discussed a dividend.",
            "turns": [
                {
                    "speaker": "Ada Lovelace",
                    "text": "a dividend from Sunfish to the trust",
                }
            ],
            "issues": ["Corrected 'Suncorp' to the lexicon's 'Sunfish'."],
        },
        {"meeting_summary": "s", "discussion_notes": "- bullet"},
    )
    agent = ClaudeAgent(run_query=query)
    vault = _empty_vault(tmp_path)

    await run_polish(
        FIXTURES / "meeting_note.md",
        "run-1",
        chapter=None,
        max_concurrency=4,
        agent=agent,
        vault=vault,
        cache=cache,
    )
    await run_minutes(
        FIXTURES / "meeting_note.md", "run-1", agent=agent, vault=vault, cache=cache
    )

    assert len(query.calls) == 3
    minutes_prompt = query.calls[2][0]
    assert "Sunfish" in minutes_prompt
    assert "Suncorp" not in minutes_prompt


# --- run_minutes orchestration -------------------------------------------------


async def test_run_minutes_raises_without_cached_polished_chapters(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    agent = ClaudeAgent(run_query=ScriptedQuery())
    vault = _empty_vault(tmp_path)

    with pytest.raises(MissingPolishedChaptersError) as excinfo:
        await run_minutes(
            FIXTURES / "meeting_note.md", "run-1", agent=agent, vault=vault, cache=cache
        )

    assert "run-1" in str(excinfo.value)
    assert "transcript polish" in str(excinfo.value)


async def test_run_minutes_stores_minutes_json_on_disk_and_returns_it(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    chapters = [
        _chapter("Opening", summary="Kickoff", turns=[("Ada Lovelace", "Let's begin.")])
    ]
    cache.store("run-1", POLISHED_CACHE_NAME, PolishedChapterList(chapters=chapters))
    fake = ScriptedQuery(
        {"meeting_summary": "A short summary.", "discussion_notes": "- A bullet"}
    )
    agent = ClaudeAgent(run_query=fake)
    vault = _empty_vault(tmp_path)

    result = await run_minutes(
        FIXTURES / "meeting_note.md", "run-1", agent=agent, vault=vault, cache=cache
    )

    assert result.run_id == "run-1"
    assert result.meeting_summary == "A short summary."
    assert result.discussion_notes == "- A bullet"

    run_dir = cache.run_dir("run-1")
    # The literal on-disk filename the typed cache path produces - plan 007
    # shipped `chapters.txt` by accident via `store_text`; this guards
    # against repeating that here.
    assert (run_dir / "minutes.json").exists()
    assert not (run_dir / "minutes.txt").exists()

    loaded = cache.load("run-1", MINUTES_CACHE_NAME, MinutesResult)
    assert loaded is not None
    assert loaded.discussion_notes == "- A bullet"


async def test_run_minutes_passes_the_note_attendees_and_lexicon_through(
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    cache = RunCache(tmp_path / "cache")
    chapters = [_chapter("Opening", summary="s", turns=[("Ada Lovelace", "hello")])]
    cache.store("run-1", POLISHED_CACHE_NAME, PolishedChapterList(chapters=chapters))
    fake = ScriptedQuery({"meeting_summary": "s", "discussion_notes": "- bullet"})
    agent = ClaudeAgent(run_query=fake)
    vault = FakeVaultClient(vault_root)

    await run_minutes(
        FIXTURES / "meeting_note.md", "run-1", agent=agent, vault=vault, cache=cache
    )

    prompt = fake.calls[0][0]
    # `meeting_note.md`'s frontmatter attendees are Ada Lovelace and Grace
    # Hopper - the note's own attendee list reaches the prompt.
    assert "Ada Lovelace" in prompt
    assert "Grace Hopper" in prompt


# --- CLI: delegation, error mapping, help -------------------------------------


def test_minutes_cli_delegates_to_run_minutes_and_prints_its_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_result = MinutesResult(
        run_id="run-1", meeting_summary="s", discussion_notes="- b"
    )
    captured: dict[str, object] = {}

    async def fake_run_minutes(
        note_path: Path,
        run_id: str,
        *,
        agent: object,
        vault: object,
        cache: object,
    ) -> object:
        captured["note_path"] = note_path
        captured["run_id"] = run_id
        return fake_result

    monkeypatch.setattr(transcript_cli, "run_minutes", fake_run_minutes)
    note_path = FIXTURES / "meeting_note.md"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "minutes",
            str(note_path),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(fake_result.model_dump_json())
    assert captured["note_path"] == note_path
    assert captured["run_id"] == "run-1"


def test_minutes_cli_reports_a_missing_prerequisite_as_a_clean_click_exception(
    tmp_path: Path,
) -> None:
    """No `--cache-root` prerequisite has been run yet: the real (not
    monkeypatched) `run_minutes` must raise `MissingPolishedChaptersError`
    and the CLI must turn that into a clean, non-zero exit that names the
    prerequisite command - without ever needing a real agent or vault
    call, since the cache check happens before either is used."""

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "minutes",
            str(FIXTURES / "meeting_note.md"),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "run-1" in result.output
    assert "transcript polish" in result.output


def test_minutes_cli_reports_claude_agent_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def raising_run_minutes(
        note_path: Path,
        run_id: str,
        *,
        agent: object,
        vault: object,
        cache: object,
    ) -> object:
        raise ClaudeAgentError("agent call failed")

    monkeypatch.setattr(transcript_cli, "run_minutes", raising_run_minutes)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "minutes",
            str(FIXTURES / "meeting_note.md"),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "agent call failed" in result.output


def test_minutes_cli_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["transcript", "minutes", "--help"])

    assert result.exit_code == 0
    assert "minutes" in result.output.lower()


# --- slow: real-LLM integration test -------------------------------------------
#
# Everything above proves the plumbing calls the model correctly and that
# the load-bearing instructions reach the prompt; it cannot prove the real
# model actually *follows* them. This is the one test that does, run with
# `uv run pytest --slow -k live tests/test_transcription_minutes.py`.
#
# Its assertions deliberately avoid "does the word appear anywhere in the
# whole reply" - that would pass unchanged for a regression that resolves
# the open question ("the team decided on canary") or that mentions the
# venue and Rob in unrelated bullets. The helpers below group the reply
# into bullet items and isolate one topic's block, so the checks below can
# require same-bullet/same-topic co-occurrence instead.

_BULLET_LINE_RE = re.compile(r"^[ \t]*[-*]\s+")
_TOP_LEVEL_BULLET_RE = re.compile(r"^[-*]\s+")  # no leading whitespace


def _bullet_items(text: str) -> list[str]:
    """Group `text`'s lines into logical bullet items.

    A line starting a new bullet (at any nesting level) starts a new item;
    any other line is a wrapped continuation of the previous bullet and is
    folded into it. This matters for the co-occurrence checks below: a
    bullet whose sentence happens to wrap onto a second line must still
    count as one item, not be split into two unrelated "lines".
    """
    items: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if _BULLET_LINE_RE.match(line) or not items:
            items.append(line)
        else:
            items[-1] += " " + line.strip()
    return items


def _topic_block(text: str, keyword: str) -> str:
    """The lines belonging to the top-level bullet whose own line mentions
    `keyword` (case-insensitively): that bullet line plus every following
    line up to (but excluding) the next top-level bullet.

    This bounds a search to one topic instead of the whole document, so an
    unrelated word elsewhere (e.g. a different bullet that happens to say
    "open") can't produce a false pass. Falls back to the whole text if no
    top-level bullet matches `keyword`, so a structural surprise (e.g. the
    model titling the topic differently than expected) widens the search
    rather than silently making the check vacuous.
    """
    lines = text.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if _TOP_LEVEL_BULLET_RE.match(line)
            and keyword.casefold() in line.casefold()
        ),
        None,
    )
    if start is None:
        return text
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if _TOP_LEVEL_BULLET_RE.match(lines[i])
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _contains_marker(text: str, marker: str) -> bool:
    """Case-insensitive containment for one curated word/phrase marker.

    A single word is matched with a `\\b` word-boundary regex so it can't
    fire on an unrelated word that merely contains it as a substring (e.g.
    "rob" inside "problem", "agreed" inside "disagreed", "will" inside
    "willing") - a real trap for the short markers these checks use. A
    multi-word phrase (e.g. "no decision") is matched as a plain substring;
    word-boundary regex adds nothing extra for a phrase.
    """
    if " " in marker:
        return marker.casefold() in text.casefold()
    return re.search(rf"\b{re.escape(marker)}\b", text, re.IGNORECASE) is not None


def _contains_any_marker(text: str, markers: Sequence[str]) -> bool:
    return any(_contains_marker(text, marker) for marker in markers)


_NEGATION_BEFORE_DECISION_RE = re.compile(
    r"\b(?:hasn'?t|has\s+not|haven'?t|have\s+not|not|never|no|isn'?t|wasn'?t)\b"
    r"(?:\s+\w+){0,3}\s+$",
    re.IGNORECASE,
)


def _contains_unnegated_decision_language(text: str, markers: Sequence[str]) -> bool:
    """Like `_contains_any_marker`, but a marker preceded (within a few
    words) by a negation does not count.

    Reporting "hasn't been decided" is exactly the report-never-prescribe,
    grounded behaviour these tests want - it restates what a speaker
    literally said (e.g. "we haven't decided") - not decision language
    sneaking onto a bullet that should report the alternatives as open. A
    bare substring/word-boundary check on "decided" cannot tell those
    apart; this looks at the words immediately before the match.
    """
    for marker in markers:
        for match in re.finditer(rf"\b{re.escape(marker)}\b", text, re.IGNORECASE):
            preceding = text[: match.start()]
            if _NEGATION_BEFORE_DECISION_RE.search(preceding):
                continue
            return True
    return False


def test_bullet_items_folds_a_wrapped_continuation_into_its_bullet() -> None:
    text = "- Top bullet\n  - Sub bullet starts here\n    and wraps onto this line\n- Next top bullet"

    items = _bullet_items(text)

    assert items == [
        "- Top bullet",
        "  - Sub bullet starts here and wraps onto this line",
        "- Next top bullet",
    ]


def test_topic_block_isolates_one_top_level_bullet_and_its_children() -> None:
    text = (
        "- Venue booking\n"
        "  - Rob will book it\n"
        "- Deployment strategy\n"
        "  - Blue-green or canary\n"
        "  - Remains open\n"
        "- Wrap-up\n"
        "  - Nothing else to add\n"
    )

    block = _topic_block(text, "deployment")

    assert "Deployment strategy" in block
    assert "Blue-green or canary" in block
    assert "Remains open" in block
    assert "Venue booking" not in block
    assert "Wrap-up" not in block


_VENUE_CHAPTER = _chapter(
    "Venue booking",
    summary="The group confirmed who is booking the November offsite venue.",
    turns=[
        (
            "Michael Bryan",
            "Let's figure out who's booking the venue for the November offsite.",
        ),
        (
            "Rob Miller",
            "I'll book the venue for the November offsite by next Friday.",
        ),
        ("Michael Bryan", "Perfect, thanks Rob."),
    ],
)
_DEPLOYMENT_CHAPTER = _chapter(
    "Deployment strategy",
    summary=(
        "The group weighed two deployment strategies for the autonomy "
        "stack rollout without deciding."
    ),
    turns=[
        (
            "Michael Bryan",
            "For the autonomy stack rollout we need to pick a deployment "
            "strategy - blue-green or canary.",
        ),
        (
            "Nikki Staltari",
            "We haven't decided. Blue-green is simpler to reason about, "
            "but canary lets us catch problems with a smaller blast radius.",
        ),
        (
            "Michael Bryan",
            "Let's leave it open for now and revisit at the next planning cycle.",
        ),
    ],
)
_ATTENDEES = ["Michael Bryan", "Rob Miller", "Nikki Staltari"]

# None of these markers appear anywhere in the raw dialogue above (checked
# by `test_the_fixtures_own_prescriptive_marker_check_is_honest` below), so
# their presence in the model's reply below would mean the model introduced
# prescriptive/advisory language rather than reporting what was said - a
# deliberately narrow, honestly-documented proxy for "report, never
# prescribe" (it cannot prove the model never prescribes in any form, only
# that it doesn't reach for these specific well-known recommendation
# markers on this fixture). If a real run trips this, the fix is the
# prompt, per the plan's STOP condition - never loosening this assertion.
_PRESCRIPTIVE_MARKERS = ("should", "recommend", "next steps", "action item")


def test_the_fixtures_own_prescriptive_marker_check_is_honest() -> None:
    """Guards the slow test's premise: none of `_PRESCRIPTIVE_MARKERS`
    appear in the raw dialogue we feed the model, so the slow test's
    assertion that they're absent from the *reply* is actually checking
    for something the model introduced, not something it merely repeated."""

    raw = " ".join(
        text
        for chapter in (_VENUE_CHAPTER, _DEPLOYMENT_CHAPTER)
        for turn in chapter.turns
        for text in (turn.text,)
    ).casefold()
    for marker in _PRESCRIPTIVE_MARKERS:
        assert marker not in raw


@pytest.mark.slow
async def test_live_minutes_report_facts_and_action_and_leave_the_open_question_open() -> (
    None
):
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))

    response = await generate_minutes(
        [_VENUE_CHAPTER, _DEPLOYMENT_CHAPTER],
        _ATTENDEES,
        [*_ATTENDEES, "Autonomy Stack"],
        agent,
    )

    assert response.meeting_summary.strip() != ""
    notes = response.discussion_notes
    assert notes.strip() != ""

    lines = notes.splitlines()
    # Discussion Notes are nested Markdown bullets, not prose: at least one
    # top-level bullet and at least one indented sub-bullet.
    assert any(line.lstrip().startswith(("-", "*")) for line in lines)
    assert any(
        line[:1] in (" ", "\t") and line.lstrip().startswith(("-", "*"))
        for line in lines
    )
    # No heading or callout syntax - those are added at integrate time.
    assert "## Discussion Notes" not in notes
    assert "[!summary]" not in response.meeting_summary.casefold()

    combined = f"{response.meeting_summary}\n{notes}"
    lowered = combined.casefold()

    # An attendee wikilink is present.
    assert "[[Rob Miller]]" in combined

    # The as-said action item is attributed: "venue" and "Rob" must
    # co-occur on the SAME bullet, not merely appear anywhere in the
    # document - two words mentioned in unrelated bullets would prove
    # nothing about attribution, and "rob"/"agreed"/"will" are short enough
    # to hit unrelated words as bare substrings (see `_contains_marker`).
    items = _bullet_items(notes)
    venue_items = [item for item in items if _contains_marker(item, "venue")]
    assert venue_items, f"no bullet mentions the venue: {notes!r}"
    venue_and_rob_items = [
        item for item in venue_items if _contains_marker(item, "rob")
    ]
    assert venue_and_rob_items, (
        f"a venue bullet exists but none also names Rob: {venue_items!r}"
    )
    # Ideally that bullet also carries a commitment cue, not just both
    # names in passing - catching a regression that drops the actual
    # as-said commitment while still mentioning venue and Rob separately.
    assert any(
        _contains_marker(item, cue) or "'ll" in item.casefold()
        for item in venue_and_rob_items
        for cue in ("will", "committed", "agreed")
    ), f"no venue/Rob bullet carries a commitment cue: {venue_and_rob_items!r}"

    # The open question is reported as open, never resolved one way. Two
    # things must both hold, checked honestly against the actual
    # regression this guards: a model that writes "the team decided to go
    # with canary" would still make both alternative names appear "nearby"
    # an unresolved-sounding word elsewhere in the topic (a bare presence
    # check alone would pass it unchanged), so the decisive check is that
    # NO decision language sits on the same bullet as the alternatives.
    deployment_block = _topic_block(notes, "deployment")
    assert "blue-green" in deployment_block.casefold()
    assert "canary" in deployment_block.casefold()

    unresolved_cues = ("open", "undecided", "unresolved", "remains", "no decision")
    assert _contains_any_marker(deployment_block, unresolved_cues), (
        "no unresolved cue found near the alternatives "
        f"(topic block: {deployment_block!r})"
    )

    alternative_items = [
        item
        for item in _bullet_items(deployment_block)
        if "blue-green" in item.casefold() or "canary" in item.casefold()
    ]
    assert alternative_items
    decision_markers = ("decided", "agreed")
    for item in alternative_items:
        assert not _contains_unnegated_decision_language(item, decision_markers), (
            f"decision language found on the same bullet as the alternatives: {item!r}"
        )

    # Report, never prescribe (see `_PRESCRIPTIVE_MARKERS` above).
    for marker in _PRESCRIPTIVE_MARKERS:
        assert marker not in lowered, (
            f"prescriptive marker {marker!r} found in real model output: {combined!r}"
        )

    print("meeting_summary:", response.meeting_summary)
    print("discussion_notes:", notes)


# --- slow: grounding - no invented dates/owners, no misattribution ----------
#
# The acceptance review's F4/F5/F19 findings: a real run stated an
# unresolved date as confirmed fact, invented an action item for the person
# it was *asked of* rather than the person who actually committed to it,
# and reported a fragmentary guess as a settled conclusion. This fixture
# plants both shapes deliberately: a genuinely ambiguous date (the raw
# dialogue itself gives two candidates and says nobody is sure), and an
# action item whose owner (Rob Miller, who says "I'll ...") is a different
# person from who it was addressed to (Michael Bryan, who only asked).

_INVOICE_CHAPTER = _chapter(
    "Supplier invoice payment date",
    summary=(
        "The group was unsure whether a supplier invoice was paid before "
        "or after quarter end."
    ),
    turns=[
        (
            "Michael Bryan",
            "Did we pay that supplier invoice before the end of the "
            "quarter, or just after?",
        ),
        (
            "Rob Miller",
            "Honestly I'm not certain - the portal shows the twelfth, but "
            "it might have actually gone through on the fourteenth, the "
            "finance portal was glitching that day.",
        ),
        ("Michael Bryan", "Okay, let's check the bank statement to be sure."),
    ],
)
_FOLLOWUP_CHAPTER = _chapter(
    "Bank statement follow-up",
    summary="Rob agreed to pull the bank statement to confirm the payment date.",
    turns=[
        (
            "Michael Bryan",
            "Can you pull the bank statement and confirm the exact date?",
        ),
        ("Rob Miller", "Yeah, I'll pull the statement and confirm by Friday."),
    ],
)


def test_the_invoice_fixtures_ambiguity_and_ownership_are_honest() -> None:
    """Guards the slow test's premise: the two candidate dates really are
    both in the raw dialogue (so a check for both surviving is checking
    genuine preservation, not something the model would say unprompted),
    and it's Rob, not Michael, who actually commits to the follow-up action
    (so a check that the action lands on Rob is checking real attribution,
    not something trivially true regardless of who said what)."""

    invoice_text = " ".join(turn.text for turn in _INVOICE_CHAPTER.turns).casefold()
    assert "twelfth" in invoice_text
    assert "fourteenth" in invoice_text

    committing_turn = _FOLLOWUP_CHAPTER.turns[1]
    assert committing_turn.speaker == "Rob Miller"
    assert "i'll" in committing_turn.text.casefold()


@pytest.mark.slow
async def test_live_minutes_does_not_invent_a_confirmed_date_or_misattribute_the_action() -> (
    None
):
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))

    response = await generate_minutes(
        [_INVOICE_CHAPTER, _FOLLOWUP_CHAPTER],
        ["Michael Bryan", "Rob Miller"],
        ["Michael Bryan", "Rob Miller"],
        agent,
    )

    notes = response.discussion_notes
    assert notes.strip() != ""

    # Grounding on the ambiguous date: the raw dialogue gives two
    # candidates and says nobody is sure which. The minutes must not
    # assert exactly one of them as confirmed fact - either both survive
    # (the ambiguity reported honestly) or neither does (omitted), but not
    # exactly one, which would mean the model silently resolved an
    # unresolved fact.
    invoice_block = _topic_block(notes, "invoice")
    day_mentions = {
        match.lower()
        for match in re.findall(
            r"\b(twelfth|fourteenth|12th|14th)\b", invoice_block, re.IGNORECASE
        )
    }
    assert len(day_mentions) != 1, (
        f"minutes asserted exactly one of two candidate dates as fact when "
        f"the transcript left it ambiguous: {invoice_block!r}"
    )

    # Grounding on ownership: the bank-statement action was Rob's ("I'll
    # pull the statement"), not Michael's (who only asked for it, and who
    # separately - and legitimately - proposed checking the statement
    # himself back in the first chapter; that's a different, real bullet
    # this check must not collide with). "pull" is the verb only Rob's
    # actual commitment uses in the raw dialogue, so it isolates the
    # delegated action from Michael's own, unrelated proposal.
    items = _bullet_items(notes)
    pull_items = [item for item in items if _contains_marker(item, "pull")]
    assert pull_items, f"no bullet describes pulling the bank statement: {notes!r}"
    assert all(_contains_marker(item, "Rob") for item in pull_items), (
        f"a bullet about pulling the bank statement doesn't name Rob - the "
        f"action looks misattributed away from who actually committed to "
        f"it: {pull_items!r}"
    )

    print("meeting_summary:", response.meeting_summary)
    print("discussion_notes:", notes)
