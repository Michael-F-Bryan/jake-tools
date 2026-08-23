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
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from click.testing import CliRunner

from jake_tools.claude import AgentSpec, ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.minutes import (
    MINUTES_CACHE_NAME,
    MinutesResult,
    MissingPolishedChaptersError,
    generate_minutes,
    run_minutes,
)
from jake_tools.transcription.models import PolishedChapter, PolishedTurn
from jake_tools.transcription.note import NoteSection, ParsedNote
from jake_tools.transcription.obsidian import VaultClient
from jake_tools.transcription.polish import POLISHED_CACHE_NAME, PolishedChapterList

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

    # The as-said action item is attributed: who, and what they took on.
    assert "venue" in lowered
    assert "rob" in lowered

    # The open question is reported as open, with both alternatives named -
    # never resolved one way.
    assert "blue-green" in lowered
    assert "canary" in lowered

    # Report, never prescribe (see `_PRESCRIPTIVE_MARKERS` above).
    for marker in _PRESCRIPTIVE_MARKERS:
        assert marker not in lowered, (
            f"prescriptive marker {marker!r} found in real model output: {combined!r}"
        )

    print("meeting_summary:", response.meeting_summary)
    print("discussion_notes:", notes)
