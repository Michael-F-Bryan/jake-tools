"""Behaviour of per-chapter polish + adversarial fix (`transcription/polish.py`)
and the `jake-tools transcript polish` CLI command.

This module protects the product's existential quality bar (plan
008-polish-and-fix.md), and it deliberately tests two different things at
two different layers - conflating them would either hide a plumbing
regression behind LLM noise, or let a genuinely bad prompt hide behind
green plumbing tests:

- **Plumbing/contract tests (the bulk of this file, fast, run by default).**
  A fake `ClaudeAgent` (patterns: `ScriptedQuery`/`ConcurrencyTrackingQuery`,
  both following `tests/test_transcription_chapters.py`'s `ScriptedQuery`)
  proves the *shape* of the orchestration is right: polish and fix are two
  *separate* `run_structured` calls per chapter, the fixer's prompt actually
  contains both the raw utterances and the polisher's output (never a
  "check your work" turn on one conversation), the fixer's corrected output
  - not the polisher's - is what ends up in the `PolishedChapter`, the
  semaphore bounds concurrency, `--chapter` merges correctly, and a failing
  chapter names itself in the error. None of this proves the *prompts*
  produce good polish - a fake only ever replays what the test already
  wrote down.
- **`@pytest.mark.slow` integration tests (real Claude Sonnet calls, opt-in
  via `--slow`, skipped by default - see `pyproject.toml`'s `slow` marker).**
  These exercise the real prompts against a deliberately messy raw-ASR
  fixture and assert on *observable properties* of the real model's reply
  (fragments gone, speakers constrained, lexicon term corrected, content
  words survive) rather than exact strings, since LLM output is
  nondeterministic. This is the only layer that actually tests whether the
  prompts teach the model to polish transcripts well; the fake-based tests
  above cannot substitute for it.

Per the CLI-options memo (rule 4), the primary test surface is the library
seam (`polish_chapters`, `build_lexicon`, `run_polish`). CLI tests stay
thin: flag parsing -> delegation, mirroring `test_transcript_cli.py`.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from click.testing import CliRunner

from jake_tools.claude import AgentSpec, ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache, stable_hash
from jake_tools.transcription.chapters import ChapterList
from jake_tools.transcription.models import (
    ChapterSpan,
    PolishedChapter,
    PolishedTurn,
    RawTranscript,
    Utterance,
)
from jake_tools.transcription.note import (
    NoteSection,
    ParsedNote,
    human_owned_note_context,
    parse_note,
)
from jake_tools.transcription.polish import (
    ChapterFixPrompt,
    ChapterFixResponse,
    ChapterPolishError,
    InvalidChapterIndexError,
    MissingChaptersError,
    MissingPolishedChaptersError,
    MissingResolvedTranscriptError,
    PolishedChapterList,
    PolishedChapterResponse,
    PolishIssueList,
    PolishPrompt,
    _render_turns,
    _render_utterances,
    build_lexicon,
    polish_chapters,
    run_polish,
)

FIXTURES = Path(__file__).parent / "fixtures"

# `jake_tools.cli`'s __init__ rebinds the name `transcript` to the Click
# group, shadowing the submodule (see `test_transcript_cli.py`) — fetch the
# actual module via importlib to monkeypatch its `run_polish` binding.
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


class SelectiveFailQuery:
    """A fake `run_query` that returns no structured output for one chapter.

    Any prompt containing `fail_marker` gets `structured_output=None`
    (which `ClaudeAgent.run_structured` turns into a `ClaudeAgentError`);
    every other prompt gets `ok_payload`.
    """

    def __init__(self, fail_marker: str, ok_payload: object) -> None:
        self._fail_marker = fail_marker
        self._ok_payload = ok_payload
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))
        payload = None if self._fail_marker in prompt else self._ok_payload

        async def stream() -> AsyncIterator[Message]:
            yield _structured_result(payload)

        return stream()


class ConcurrencyTrackingQuery:
    """A fake `run_query` that always replays the same payload, tracking
    the maximum number of concurrently in-flight calls (a short real sleep
    per call gives overlapping tasks a chance to actually overlap)."""

    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []
        self._in_flight = 0
        self.max_in_flight = 0

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))

        async def stream() -> AsyncIterator[Message]:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            try:
                await asyncio.sleep(0.02)
                yield _structured_result(self._payload)
            finally:
                self._in_flight -= 1

        return stream()


class FakeVaultClient:
    """Fake `VaultClient`: `vault_root` points at a real tmp_path tree so
    `build_lexicon`'s glob has real files to find; `resolve_embed` is
    unused by polish.py and left unimplemented."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def vault_root(self) -> Path:
        return self._root

    def resolve_embed(self, target: str) -> Path:  # pragma: no cover - unused here
        raise NotImplementedError


def _utterance(index: int, speaker: str, text: str) -> Utterance:
    return Utterance(
        start=float(index), end=float(index) + 0.9, speaker=speaker, text=text
    )


def _transcript(utterances: list[Utterance]) -> RawTranscript:
    return RawTranscript(clips=[], utterances=utterances, audio_sha256=None)


def _note(
    *,
    attendees: list[str],
    body: str = "",
) -> ParsedNote:
    return ParsedNote(
        path="fake-note.md",
        frontmatter={},
        context="meeting",
        attendees=attendees,
        diarisation_hints=[],
        embeds=[],
        sections=[NoteSection(heading=None, level=0, body=body)],
    )


def _empty_vault(tmp_path: Path) -> FakeVaultClient:
    root = tmp_path / "empty-vault"
    root.mkdir(exist_ok=True)
    return FakeVaultClient(root)


# --- Step 1: build_lexicon ----------------------------------------------------


def test_build_lexicon_combines_attendees_wikilinks_and_vault_titles_deduped(
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    (vault_root / "Ada Lovelace.md").write_text("# Ada Lovelace\n")
    (vault_root / "Autonomy Stack.md").write_text("# Autonomy Stack\n")
    sub = vault_root / "sub"
    sub.mkdir()
    (sub / "Rob Miller.md").write_text("# Rob Miller\n")

    note = _note(
        attendees=["Ada Lovelace", "Grace Hopper"],
        body=(
            "Some intro mentioning [[Autonomy Stack]] and an embed "
            "![[Recording.m4a]] that must not count as a wikilink.\n"
        ),
    )
    vault = FakeVaultClient(vault_root)

    lexicon = build_lexicon(note, vault)

    assert "Ada Lovelace" in lexicon
    assert "Grace Hopper" in lexicon
    assert "Autonomy Stack" in lexicon
    assert "Rob Miller" in lexicon
    assert "Recording.m4a" not in lexicon  # the embed target, not a wikilink
    # Ada Lovelace is both an attendee and a vault-note title: deduped once.
    assert lexicon.count("Ada Lovelace") == 1


def test_build_lexicon_caps_the_list(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    for i in range(500):
        (vault_root / f"Note {i}.md").write_text("x")

    note = _note(attendees=[])
    vault = FakeVaultClient(vault_root)

    lexicon = build_lexicon(note, vault)

    assert len(lexicon) <= 300


# --- Step 2/3: polish_chapters -----------------------------------------------


async def test_polish_then_fix_are_two_separate_calls_and_fixer_sees_raw_and_polished(
    tmp_path: Path,
) -> None:
    transcript = _transcript(
        [
            _utterance(0, "Ada Lovelace", "Let's tal- talk about vel- ocity."),
            _utterance(1, "Grace Hopper", "Sounds good."),
        ]
    )
    span = ChapterSpan(
        title="Opening", start_utterance=0, end_utterance=1, start_seconds=0.0
    )
    fake = ScriptedQuery(
        {
            "summary": "Draft summary",
            "turns": [
                {"speaker": "Ada Lovelace", "text": "Let's talk about vel- ocity."}
            ],
        },
        {
            "summary": "Fixed summary",
            "turns": [
                {"speaker": "Ada Lovelace", "text": "Let's talk about velocity."},
                {"speaker": "Grace Hopper", "text": "Sounds good."},
            ],
            "issues": ["Reattached a split word: 'vel- ocity' became 'velocity'."],
        },
    )
    agent = ClaudeAgent(run_query=fake)
    note = _note(attendees=["Ada Lovelace", "Grace Hopper"])
    vault = _empty_vault(tmp_path)

    chapters, issues = await polish_chapters(transcript, [span], note, vault, agent)

    assert len(fake.calls) == 2  # a fresh call for polish, a fresh call for fix
    polish_prompt, fix_prompt = (call[0] for call in fake.calls)

    # The fixer's prompt carries BOTH the raw utterances and the polisher's
    # output - the guard rail against collapsing this into one call.
    assert "Let's tal- talk about vel- ocity." in fix_prompt  # raw
    assert "Draft summary" in fix_prompt  # polished summary
    assert "Let's talk about vel- ocity." in fix_prompt  # polished turn text
    # The polish prompt never saw a "fixed" version - it's a fresh call.
    assert "Fixed summary" not in polish_prompt

    # The fixer's corrected output wins over the polisher's.
    assert len(chapters) == 1
    assert chapters[0].title == "Opening"
    assert chapters[0].start_seconds == 0.0
    assert chapters[0].summary == "Fixed summary"
    assert chapters[0].turns == [
        PolishedTurn(speaker="Ada Lovelace", text="Let's talk about velocity."),
        PolishedTurn(speaker="Grace Hopper", text="Sounds good."),
    ]
    assert issues == [
        "Opening: Reattached a split word: 'vel- ocity' became 'velocity'."
    ]


async def test_polish_library_rejects_non_positive_max_concurrency(
    tmp_path: Path,
) -> None:
    transcript = _transcript([_utterance(0, "Ada Lovelace", "Hello.")])
    span = ChapterSpan(
        title="Opening", start_utterance=0, end_utterance=0, start_seconds=0.0
    )

    with pytest.raises(ValueError, match="max_concurrency"):
        await polish_chapters(
            transcript,
            [span],
            _note(attendees=["Ada Lovelace"]),
            _empty_vault(tmp_path),
            ClaudeAgent(run_query=ScriptedQuery({}, {})),
            max_concurrency=0,
        )


async def test_fixer_meta_commentary_never_ships_as_the_chapter_summary_when_nothing_was_fixed(
    tmp_path: Path,
) -> None:
    """Regression for the acceptance run's chapter-11 leak: the fixer found
    nothing to fix (`issues == []`) but still wrote review commentary about
    itself into `summary` - "Reviewed the polish against the raw
    transcript... so the chapter is returned unchanged" shipped verbatim as
    a user-facing chapter summary. When the fixer reports no issues, its
    `summary` must never be trusted over the polish's own - this is exactly
    the shape a real model produced.
    """
    transcript = _transcript(
        [_utterance(0, "Ada Lovelace", "Let's talk about crosstalk handling.")]
    )
    span = ChapterSpan(
        title="Crosstalk", start_utterance=0, end_utterance=0, start_seconds=0.0
    )
    leaked_commentary = (
        "Reviewed the polish against the raw transcript. The raw text "
        "contains heavily interleaved crosstalk that the lexicon does not "
        "resolve cleanly, so there is a risk of meaning drift, but the "
        "polish captures every substantive statement, so the chapter is "
        "returned unchanged."
    )
    fake = ScriptedQuery(
        {
            "summary": "The team discussed how to handle crosstalk.",
            "turns": [
                {
                    "speaker": "Ada Lovelace",
                    "text": "Let's talk about crosstalk handling.",
                }
            ],
        },
        {
            "summary": leaked_commentary,
            "turns": [
                {
                    "speaker": "Ada Lovelace",
                    "text": "Let's talk about crosstalk handling.",
                }
            ],
            "issues": [],  # nothing flagged as needing a fix
        },
    )
    agent = ClaudeAgent(run_query=fake)
    note = _note(attendees=["Ada Lovelace"])
    vault = _empty_vault(tmp_path)

    chapters, issues = await polish_chapters(transcript, [span], note, vault, agent)

    assert issues == []
    assert chapters[0].summary == "The team discussed how to handle crosstalk."
    assert leaked_commentary not in chapters[0].summary


def test_fixer_prompt_states_the_lexicon_is_authoritative_over_reverting_a_correction() -> (
    None
):
    """Pins the grounding rule the acceptance review's F2/F3/F22 findings
    need: a real fixer reverted a lexicon-supported correction ("Xero")
    back to a raw ASR artefact ("Jane"), and separately left an unrelated
    real company name ("Suncorp") uncorrected despite the meeting's own
    company ("Sunfish") sitting in the lexicon. The rendered prompt must
    say, imperatively, not to do either."""

    prompt = ChapterFixPrompt(
        chapter_title="Opening",
        lexicon=["Sunfish"],
        raw_lines="Ada Lovelace: hello",
        polished_summary="s",
        polished_lines="Ada Lovelace: hello",
    ).render()

    normalised = " ".join(prompt.split())
    assert "do NOT revert it back to the raw wording" in normalised
    assert "never introduce, restore, or invent a proper noun" in normalised


async def test_polish_chapters_passes_the_lexicon_into_the_polish_prompt(
    tmp_path: Path,
) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    transcript = _transcript([_utterance(0, "Ada Lovelace", "hello")])
    span = ChapterSpan(
        title="Opening", start_utterance=0, end_utterance=0, start_seconds=0.0
    )
    fake = ScriptedQuery(
        {"summary": "s", "turns": [{"speaker": "Ada Lovelace", "text": "hello"}]},
        {
            "summary": "s",
            "turns": [{"speaker": "Ada Lovelace", "text": "hello"}],
            "issues": [],
        },
    )
    agent = ClaudeAgent(run_query=fake)
    note = _note(attendees=["Ada Lovelace"], body="Mentions [[Autonomy Stack]].")
    vault = FakeVaultClient(vault_root)

    await polish_chapters(transcript, [span], note, vault, agent)

    polish_prompt = fake.calls[0][0]
    assert "Autonomy Stack" in polish_prompt
    assert "Ada Lovelace" in polish_prompt


async def test_polish_chapters_runs_concurrently_bounded_by_the_semaphore(
    tmp_path: Path,
) -> None:
    chapter_count = 5
    max_concurrency = 2
    utterances = [
        _utterance(i, "SPEAKER_00", f"utterance {i}") for i in range(chapter_count)
    ]
    transcript = _transcript(utterances)
    spans = [
        ChapterSpan(
            title=f"Chapter {i}",
            start_utterance=i,
            end_utterance=i,
            start_seconds=float(i),
        )
        for i in range(chapter_count)
    ]
    fake = ConcurrencyTrackingQuery({"summary": "s", "turns": [], "issues": []})
    agent = ClaudeAgent(run_query=fake)
    note = _note(attendees=[])
    vault = _empty_vault(tmp_path)

    chapters, _issues = await polish_chapters(
        transcript, spans, note, vault, agent, max_concurrency=max_concurrency
    )

    assert len(chapters) == chapter_count
    assert len(fake.calls) == 2 * chapter_count  # polish + fix, per chapter
    assert fake.max_in_flight <= max_concurrency  # the semaphore was respected
    assert fake.max_in_flight >= 2  # and it actually ran concurrently, not serially


async def test_polish_chapters_names_the_failing_chapter_in_the_error(
    tmp_path: Path,
) -> None:
    transcript = _transcript(
        [_utterance(0, "SPEAKER_00", "fine"), _utterance(1, "SPEAKER_00", "trouble")]
    )
    good_span = ChapterSpan(
        title="Good Chapter", start_utterance=0, end_utterance=0, start_seconds=0.0
    )
    bad_span = ChapterSpan(
        title="Bad Chapter", start_utterance=1, end_utterance=1, start_seconds=1.0
    )
    fake = SelectiveFailQuery(
        fail_marker="Bad Chapter",
        ok_payload={"summary": "s", "turns": [], "issues": []},
    )
    agent = ClaudeAgent(run_query=fake)
    note = _note(attendees=[])
    vault = _empty_vault(tmp_path)

    with pytest.raises(ChapterPolishError) as excinfo:
        await polish_chapters(transcript, [good_span, bad_span], note, vault, agent)

    assert "Bad Chapter" in str(excinfo.value)


# --- slow: real-LLM integration tests -----------------------------------------
#
# Everything above proves the plumbing calls the model correctly; it cannot
# prove the *prompts* teach the model to polish transcripts well - a fake
# only ever replays what the test already wrote down. These two tests make
# real `claude-sonnet-5` calls (via the same `claude` CLI auth this
# environment already uses) against a deliberately messy fixture, and check
# observable properties of the real reply rather than exact strings, since
# LLM output is nondeterministic. Skipped by default (pytest-skip-slow); run
# with `uv run pytest --slow -k slow tests/test_transcription_polish.py`.

_MESSY_UTTERANCES = [
    _utterance(
        0,
        "SPEAKER_00",
        "So, um, the Otonomy Stack team wants to, uh, ship the pilot deployment in Q3.",
    ),
    _utterance(1, "SPEAKER_01", "Mm-hm."),
    _utterance(2, "SPEAKER_00", "We just need to hit the target vel-"),
    _utterance(
        3,
        "SPEAKER_00",
        "-ocity of twenty units a day, otherwise the beachhead use case falls apart.",
    ),
    _utterance(4, "SPEAKER_01", "Wait, twenty units a day?"),
    _utterance(
        5,
        "SPEAKER_00",
        "Yeah, twenty units a day. The safety certification review is what's blocking us right now.",
    ),
    _utterance(6, "SPEAKER_01", "Okay, uh, who owns that piece?"),
    _utterance(7, "SPEAKER_00", "That's on Jake, he's coordinating with the vendor."),
]
# "Otonomy Stack" is a plausible ASR mishearing of "Autonomy Stack" (the
# lexicon's correct term, via the wikilink below); "vel-"/"-ocity" is a word
# split by a diarisation/ASR boundary glitch; "Mm-hm." is a filler
# interjection. Distinctive content words we expect to survive polish
# verbatim-ish: Jake, vendor, certification, beachhead, deployment.
_MESSY_SPAN = ChapterSpan(
    title="Autonomy Stack pilot planning",
    start_utterance=0,
    end_utterance=len(_MESSY_UTTERANCES) - 1,
    start_seconds=0.0,
)


def _chapter_text(chapter: PolishedChapter) -> str:
    return (chapter.summary + " " + _render_turns(chapter.turns)).casefold()


@pytest.mark.slow
async def test_live_polish_repairs_fragments_and_mishearings_and_preserves_meaning(
    tmp_path: Path,
) -> None:
    transcript = _transcript(list(_MESSY_UTTERANCES))
    note = _note(
        attendees=["Jake"],
        body="Discussing the [[Autonomy Stack]] pilot.",
    )
    vault = _empty_vault(tmp_path)
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))

    chapters, issues = await polish_chapters(
        transcript, [_MESSY_SPAN], note, vault, agent, max_concurrency=1
    )

    assert len(chapters) == 1
    chapter = chapters[0]

    assert chapter.summary.strip() != ""
    assert len(chapter.turns) > 0
    for turn in chapter.turns:
        assert turn.text.strip() != ""
        assert turn.speaker in {"SPEAKER_00", "SPEAKER_01", "Unknown"}

    text = _chapter_text(chapter)

    # ASR fragmentation repaired: the split word is reattached, not left
    # dangling either side of the break.
    assert "vel-" not in text
    assert "-ocity" not in text
    assert "velocity" in text

    # The mishearing is corrected against the lexicon (the wikilink target).
    assert "otonomy" not in text
    assert "autonomy stack" in text

    # Meaning preservation: distinctive content words from the raw text
    # survive polish (not summarised away or dropped).
    for content_word in ("jake", "vendor", "certification", "beachhead", "deployment"):
        assert content_word in text, (
            f"{content_word!r} missing from polished text: {text!r}"
        )

    print("polish issues:", issues)
    print("polished chapter:", chapter.model_dump_json(indent=2))


@pytest.mark.slow
async def test_live_fixer_repairs_a_planted_defect_in_someone_elses_polish() -> None:
    """The fixer, given a deliberately broken "polish", must actually fix it.

    Plants three defects a real fixer should catch: an unrepaired split
    word, a fabricated statement the raw text does not support, and a
    dropped substantive statement (the certification review) - exactly the
    categories the fixer prompt is told to hunt for.
    """

    raw_lines = _render_utterances(_MESSY_UTTERANCES)
    broken_turns = [
        PolishedTurn(
            speaker="SPEAKER_00",
            text=(
                "So the Autonomy Stack team wants to ship the pilot "
                "deployment in Q3. We need to hit the target vel- ocity of "
                "twenty units a day, otherwise the beachhead use case falls "
                "apart."
            ),
        ),
        PolishedTurn(
            speaker="SPEAKER_00",
            text=(
                "That's on Jake, he's coordinating with the vendor. The "
                "launch has been cancelled."
            ),
        ),
    ]
    broken_summary = (
        "The team discussed the pilot deployment timeline; the launch has "
        "been cancelled."
    )
    lexicon = ["Autonomy Stack", "Jake"]
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))

    fix_response, _reply = await agent.run_structured(
        ChapterFixPrompt(
            chapter_title=_MESSY_SPAN.title,
            lexicon=lexicon,
            raw_lines=raw_lines,
            polished_summary=broken_summary,
            polished_lines=_render_turns(broken_turns),
        )
    )

    text = (fix_response.summary + " " + _render_turns(fix_response.turns)).casefold()

    # The fixer found *something* to fix - it did not just rubber-stamp it.
    assert len(fix_response.issues) > 0

    # The unrepaired fragment is gone.
    assert "vel-" not in text
    assert "-ocity" not in text
    assert "velocity" in text

    # The fabricated statement (meaning drift the raw text does not
    # support) is not repeated as fact in the corrected output.
    assert "cancelled" not in text

    # The dropped substantive statement is restored.
    assert "certification" in text

    print("fixer issues:", fix_response.issues)
    print("fixed chapter:", fix_response.model_dump_json(indent=2))


@pytest.mark.slow
async def test_live_fixer_keeps_a_lexicon_supported_correction_instead_of_reverting_it() -> (
    None
):
    """The opposite failure to the planted-defect test above: the polish
    under review is already CORRECT (it already matched a raw ASR word
    against the lexicon and fixed it), and the fixer's job here is to
    leave it alone rather than "restore" the raw wording.

    Regression for the acceptance review's F3/F22 shape: a real fixer
    reverted the polish's correct "Xero" repair back to a raw ASR artefact
    ("Jane") that occurred exactly once with no corroboration, on the
    grounds that it was "unsupported" - even though a lexicon term
    supported the correction it just undid. This plants the same shape
    with fictional names so it doesn't reproduce the acceptance run's own
    transcript content: a raw utterance names an accounting product,
    mis-transcribed once as a plausible-looking name, that the polish
    already corrected against the lexicon.
    """

    raw_lines = "SPEAKER_00: Once you're synced up in Karen Ledger, we should be able to lodge without any issues."
    already_correct_turns = [
        PolishedTurn(
            speaker="SPEAKER_00",
            text=(
                "Once you're synced up in Beacon Ledger, we should be able "
                "to lodge without any issues."
            ),
        )
    ]
    polished_summary = "Confirmed everything is synced in Beacon Ledger before lodging."
    lexicon = ["Beacon Ledger"]
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))

    fix_response, _reply = await agent.run_structured(
        ChapterFixPrompt(
            chapter_title="Bookkeeping sync",
            lexicon=lexicon,
            raw_lines=raw_lines,
            polished_summary=polished_summary,
            polished_lines=_render_turns(already_correct_turns),
        )
    )

    text = (fix_response.summary + " " + _render_turns(fix_response.turns)).casefold()

    # The lexicon-supported correction must survive - not be reverted back
    # to the raw ASR wording just because that wording is right there in
    # the raw utterances.
    assert "beacon ledger" in text
    assert "karen ledger" not in text

    print("fixer issues:", fix_response.issues)
    print("fixed chapter:", fix_response.model_dump_json(indent=2))


# --- Step 4: run_polish orchestration ----------------------------------------


def _cache_with_prerequisites(
    cache: RunCache, run_id: str, *, chapter_count: int = 2
) -> list[ChapterSpan]:
    utterances = [
        _utterance(i, "SPEAKER_00", f"utterance {i}") for i in range(chapter_count)
    ]
    cache.store(run_id, "resolved_transcript", _transcript(utterances))
    spans = [
        ChapterSpan(
            title=f"Chapter {i}",
            start_utterance=i,
            end_utterance=i,
            start_seconds=float(i),
        )
        for i in range(chapter_count)
    ]
    cache.store(run_id, "chapters", ChapterList(chapters=spans))
    return spans


async def test_run_polish_raises_without_a_cached_resolved_transcript(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    agent = ClaudeAgent(run_query=ScriptedQuery())
    vault = _empty_vault(tmp_path)

    with pytest.raises(MissingResolvedTranscriptError) as excinfo:
        await run_polish(
            FIXTURES / "meeting_note.md",
            "run-1",
            chapter=None,
            max_concurrency=4,
            agent=agent,
            vault=vault,
            cache=cache,
        )

    assert "run-1" in str(excinfo.value)
    assert "transcript speakers" in str(excinfo.value)


async def test_run_polish_raises_without_cached_chapters(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.store("run-1", "resolved_transcript", _transcript([_utterance(0, "S", "hi")]))
    agent = ClaudeAgent(run_query=ScriptedQuery())
    vault = _empty_vault(tmp_path)

    with pytest.raises(MissingChaptersError) as excinfo:
        await run_polish(
            FIXTURES / "meeting_note.md",
            "run-1",
            chapter=None,
            max_concurrency=4,
            agent=agent,
            vault=vault,
            cache=cache,
        )

    assert "run-1" in str(excinfo.value)
    assert "transcript chapterise" in str(excinfo.value)


async def test_run_polish_stores_polished_json_and_polish_issues_json_on_disk(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    _cache_with_prerequisites(cache, "run-1", chapter_count=1)
    fake = ScriptedQuery(
        {"summary": "s", "turns": [{"speaker": "SPEAKER_00", "text": "utterance 0"}]},
        {
            "summary": "s",
            "turns": [{"speaker": "SPEAKER_00", "text": "utterance 0"}],
            "issues": [],
        },
    )
    agent = ClaudeAgent(run_query=fake)
    vault = _empty_vault(tmp_path)

    result = await run_polish(
        FIXTURES / "meeting_note.md",
        "run-1",
        chapter=None,
        max_concurrency=4,
        agent=agent,
        vault=vault,
        cache=cache,
    )

    run_dir = cache.run_dir("run-1")
    # The literal on-disk filenames the typed cache path produces - plan 007
    # shipped `chapters.txt` by accident via `store_text`; this guards
    # against repeating that here.
    assert (run_dir / "polished.json").exists()
    assert (run_dir / "polish_issues.json").exists()
    assert not (run_dir / "polished.txt").exists()
    assert not (run_dir / "polish_issues.txt").exists()

    loaded_chapters = cache.load("run-1", "polished", PolishedChapterList)
    assert loaded_chapters is not None
    assert loaded_chapters.chapters[0].summary == "s"

    loaded_issues = cache.load("run-1", "polish_issues", PolishIssueList)
    assert loaded_issues is not None
    assert loaded_issues.issues == []

    assert result.chapter_count == 1
    assert result.issue_count == 0


async def test_run_polish_with_chapter_flag_merges_into_an_existing_polished_json(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    _cache_with_prerequisites(cache, "run-1", chapter_count=2)

    original_chapters = [
        PolishedChapter(
            title="Chapter 0",
            start_seconds=0.0,
            summary="Original 0",
            turns=[PolishedTurn(speaker="SPEAKER_00", text="original zero")],
        ),
        PolishedChapter(
            title="Chapter 1",
            start_seconds=1.0,
            summary="Original 1 (bad)",
            turns=[PolishedTurn(speaker="SPEAKER_00", text="original one, garbled")],
        ),
    ]
    cache.store("run-1", "polished", PolishedChapterList(chapters=original_chapters))
    cache.store(
        "run-1",
        "polish_issues",
        PolishIssueList(
            issues=[
                "Chapter 0: minor filler removed.",
                "Chapter 1: garbled crosstalk left unresolved.",
            ]
        ),
    )

    fake = ScriptedQuery(
        {
            "summary": "Redone 1",
            "turns": [{"speaker": "SPEAKER_00", "text": "fixed one"}],
        },
        {
            "summary": "Redone 1, checked",
            "turns": [{"speaker": "SPEAKER_00", "text": "fixed one, checked"}],
            "issues": ["Reattached a split word."],
        },
    )
    agent = ClaudeAgent(run_query=fake)
    vault = _empty_vault(tmp_path)
    note = parse_note(FIXTURES / "meeting_note.md")
    transcript = cache.load("run-1", "resolved_transcript", RawTranscript)
    chapters = cache.load("run-1", "chapters", ChapterList)
    assert transcript is not None and chapters is not None
    lexicon = build_lexicon(note, vault)
    cache.store_manifest(
        "run-1",
        cache.stage_manifest(
            "polish",
            inputs={
                "resolved_transcript": transcript.model_dump(mode="json"),
                "chapters": chapters.model_dump(mode="json"),
                "human_context": human_owned_note_context(note),
                "lexicon": lexicon,
            },
            input_hashes={
                "resolved_transcript": stable_hash(transcript.model_dump(mode="json")),
                "chapters": stable_hash(chapters.model_dump(mode="json")),
                "human_context": stable_hash(human_owned_note_context(note)),
                "lexicon": stable_hash(lexicon),
            },
            config={
                "agent": agent.defaults.model_dump(mode="json"),
                "polish_prompt": PolishPrompt.template,
                "review_prompt": ChapterFixPrompt.template,
                "polish_schema": PolishedChapterResponse.model_json_schema(),
                "review_schema": ChapterFixResponse.model_json_schema(),
            },
        ),
    )
    current_manifest = cache.load_manifest("run-1", "polish")
    assert current_manifest is not None
    cache.store_manifest(
        "run-1",
        current_manifest.model_copy(
            update={
                "output_hash": stable_hash(
                    PolishedChapterList(chapters=original_chapters).model_dump(
                        mode="json"
                    )
                )
            }
        ),
    )

    result = await run_polish(
        FIXTURES / "meeting_note.md",
        "run-1",
        chapter=1,
        max_concurrency=4,
        agent=agent,
        vault=vault,
        cache=cache,
    )

    assert len(fake.calls) == 2  # only the re-polished chapter costs new calls

    loaded = cache.load("run-1", "polished", PolishedChapterList)
    assert loaded is not None
    # Chapter 0 is untouched...
    assert loaded.chapters[0].summary == "Original 0"
    # ...chapter 1 is replaced with the freshly polished-and-fixed result.
    assert loaded.chapters[1].summary == "Redone 1, checked"
    assert loaded.chapters[1].turns == [
        PolishedTurn(speaker="SPEAKER_00", text="fixed one, checked")
    ]

    loaded_issues = cache.load("run-1", "polish_issues", PolishIssueList)
    assert loaded_issues is not None
    # Chapter 0's old issue survives; chapter 1's old issue is replaced by
    # the new fixer run's issues, not accumulated alongside it.
    assert "Chapter 0: minor filler removed." in loaded_issues.issues
    assert "Chapter 1: garbled crosstalk left unresolved." not in loaded_issues.issues
    assert "Chapter 1: Reattached a split word." in loaded_issues.issues

    assert result.chapter_count == 2
    assert result.issue_count == 2


async def test_run_polish_with_chapter_flag_requires_a_prior_full_run(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    _cache_with_prerequisites(cache, "run-1", chapter_count=2)
    agent = ClaudeAgent(run_query=ScriptedQuery())
    vault = _empty_vault(tmp_path)

    with pytest.raises(MissingPolishedChaptersError):
        await run_polish(
            FIXTURES / "meeting_note.md",
            "run-1",
            chapter=0,
            max_concurrency=4,
            agent=agent,
            vault=vault,
            cache=cache,
        )


async def test_run_polish_with_out_of_range_chapter_flag_raises(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "cache")
    _cache_with_prerequisites(cache, "run-1", chapter_count=2)
    agent = ClaudeAgent(run_query=ScriptedQuery())
    vault = _empty_vault(tmp_path)

    with pytest.raises(InvalidChapterIndexError):
        await run_polish(
            FIXTURES / "meeting_note.md",
            "run-1",
            chapter=5,
            max_concurrency=4,
            agent=agent,
            vault=vault,
            cache=cache,
        )


# --- CLI: delegation, error mapping, help -------------------------------------


def test_polish_cli_delegates_to_run_polish_and_prints_its_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jake_tools.transcription.polish import PolishResponse

    fake_result = PolishResponse(
        run_id="run-1", chapter_count=2, issue_count=1, issues=["Chapter 0: fixed."]
    )
    captured: dict[str, object] = {}

    async def fake_run_polish(
        note_path: Path,
        run_id: str,
        *,
        chapter: int | None,
        max_concurrency: int,
        agent: object,
        vault: object,
        cache: object,
    ) -> object:
        captured["note_path"] = note_path
        captured["run_id"] = run_id
        captured["chapter"] = chapter
        captured["max_concurrency"] = max_concurrency
        return fake_result

    monkeypatch.setattr(transcript_cli, "run_polish", fake_run_polish)
    note_path = FIXTURES / "meeting_note.md"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "polish",
            str(note_path),
            "--run-id",
            "run-1",
            "--chapter",
            "1",
            "--max-concurrency",
            "8",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(fake_result.model_dump_json())
    assert captured["run_id"] == "run-1"
    assert captured["chapter"] == 1
    assert captured["max_concurrency"] == 8


def test_polish_cli_defaults_chapter_to_none_and_max_concurrency_to_four(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jake_tools.transcription.polish import PolishResponse

    captured: dict[str, object] = {}

    async def fake_run_polish(
        note_path: Path,
        run_id: str,
        *,
        chapter: int | None,
        max_concurrency: int,
        agent: object,
        vault: object,
        cache: object,
    ) -> object:
        captured["chapter"] = chapter
        captured["max_concurrency"] = max_concurrency
        return PolishResponse(run_id=run_id, chapter_count=0, issue_count=0, issues=[])

    monkeypatch.setattr(transcript_cli, "run_polish", fake_run_polish)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "polish",
            str(FIXTURES / "meeting_note.md"),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["chapter"] is None
    assert captured["max_concurrency"] == 4


@pytest.mark.parametrize("value", ["0", "-1"])
def test_polish_cli_rejects_non_positive_max_concurrency(
    value: str, tmp_path: Path
) -> None:
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "polish",
            str(FIXTURES / "meeting_note.md"),
            "--run-id",
            "run-1",
            "--max-concurrency",
            value,
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--max-concurrency'" in result.output


def test_polish_cli_reports_domain_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def raising_run_polish(
        note_path: Path,
        run_id: str,
        *,
        chapter: int | None,
        max_concurrency: int,
        agent: object,
        vault: object,
        cache: object,
    ) -> object:
        raise MissingResolvedTranscriptError(run_id)

    monkeypatch.setattr(transcript_cli, "run_polish", raising_run_polish)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "polish",
            str(FIXTURES / "meeting_note.md"),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "run-1" in result.output
    assert "transcript speakers" in result.output


def test_polish_cli_reports_claude_agent_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def raising_run_polish(
        note_path: Path,
        run_id: str,
        *,
        chapter: int | None,
        max_concurrency: int,
        agent: object,
        vault: object,
        cache: object,
    ) -> object:
        raise ClaudeAgentError("agent call failed")

    monkeypatch.setattr(transcript_cli, "run_polish", raising_run_polish)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "polish",
            str(FIXTURES / "meeting_note.md"),
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "agent call failed" in result.output


def test_polish_cli_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["transcript", "polish", "--help"])

    assert result.exit_code == 0
    assert "polish" in result.output.lower()
