"""Behaviour of chapterisation (`transcription/chapters.py`) and the
`jake-tools transcript chapterise` CLI command.

Per the CLI-options memo (rule 4), the primary test surface is the library
seam: `chapterise` (and its private post-processing helpers,
`_repair_boundaries`/`_spans_from_boundaries`) are tested directly with a
fake `ClaudeAgent` (pattern: `RecordingQuery`,
`tests/test_claude_agent.py:58-79`; a scripted multi-call variant,
`ScriptedQuery`, follows the pattern used for the retry test in
`tests/test_transcription_speakers.py`). CLI tests stay thin: flag parsing
-> delegation, plus the one bit of logic the handler itself owns (the
`effort="low"` default), mirroring `test_transcript_cli.py`.

The partition invariant - every utterance belongs to exactly one chapter -
is the load-bearing property this module protects. It is enforced entirely
in `_repair_boundaries`/`_spans_from_boundaries`, deterministically, never
trusting the model's arithmetic; `test_repair_and_spans_partition_utterances_property`
exercises that against many randomly generated (and deliberately malformed)
boundary lists.
"""

from __future__ import annotations

import importlib
import json
import random
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from click.testing import CliRunner

from jake_tools.claude import ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.chapters import (
    CHAPTERS_ADAPTER,
    ChapterBoundary,
    ChapterList,
    DegenerateChaptersError,
    MissingResolvedTranscriptError,
    _repair_boundaries,
    _spans_from_boundaries,
    chapterise,
    run_chapterisation,
)
from jake_tools.transcription.models import ChapterSpan, RawTranscript, Utterance

# `jake_tools.cli`'s __init__ rebinds the name `transcript` to the Click
# group, shadowing the submodule (see `test_transcript_cli.py`) — fetch the
# actual module via importlib to monkeypatch its `run_chapterisation`.
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


class RecordingQuery:
    """A fake `run_query` that always replays the same structured payload."""

    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))

        async def stream() -> AsyncIterator[Message]:
            yield _structured_result(self._payload)

        return stream()


class ScriptedQuery:
    """A fake `run_query` that replays a different structured payload per call.

    Mirrors the multi-call fake needed for `speakers.py`'s LLM/snippet
    round trips (`tests/test_transcription_speakers.py`), specialised here
    for chapterisation's "propose, maybe retry" shape: one payload per
    expected call, consumed in order.
    """

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


def _utterances(count: int, *, gap: float = 30.0) -> list[Utterance]:
    return [
        Utterance(
            start=index * gap,
            end=index * gap + gap - 1,
            speaker="SPEAKER_00",
            text=f"utterance {index}",
        )
        for index in range(count)
    ]


def _transcript(count: int) -> RawTranscript:
    return RawTranscript(clips=[], utterances=_utterances(count), audio_sha256=None)


# --- _repair_boundaries / _spans_from_boundaries: example-based -------------


def test_repair_and_spans_give_correct_inclusive_ends_and_start_seconds() -> None:
    utterances = _utterances(6)
    boundaries = [
        ChapterBoundary(title="Intro", start_utterance=0),
        ChapterBoundary(title="Deep dive", start_utterance=2),
        ChapterBoundary(title="Wrap up", start_utterance=4),
    ]

    repaired = _repair_boundaries(boundaries, len(utterances))
    spans = _spans_from_boundaries(repaired, utterances)

    assert spans == [
        ChapterSpan(
            title="Intro", start_utterance=0, end_utterance=1, start_seconds=0.0
        ),
        ChapterSpan(
            title="Deep dive", start_utterance=2, end_utterance=3, start_seconds=60.0
        ),
        ChapterSpan(
            title="Wrap up", start_utterance=4, end_utterance=5, start_seconds=120.0
        ),
    ]


def test_repair_boundaries_sorts_clamps_and_dedupes() -> None:
    # Unsorted, duplicate, and out-of-range (negative and too-large) inputs.
    boundaries = [
        ChapterBoundary(title="C (too large)", start_utterance=100),
        ChapterBoundary(title="B", start_utterance=2),
        ChapterBoundary(title="B (duplicate, dropped)", start_utterance=2),
        ChapterBoundary(title="A (negative)", start_utterance=-5),
    ]

    repaired = _repair_boundaries(boundaries, utterance_count=5)

    assert [(b.title, b.start_utterance) for b in repaired] == [
        ("A (negative)", 0),
        ("B", 2),
        ("C (too large)", 4),
    ]


def test_repair_boundaries_forces_the_first_boundary_to_start_at_zero() -> None:
    boundaries = [ChapterBoundary(title="Late start", start_utterance=3)]

    repaired = _repair_boundaries(boundaries, utterance_count=5)

    assert repaired == [ChapterBoundary(title="Late start", start_utterance=0)]


def test_repair_boundaries_of_empty_transcript_is_empty() -> None:
    assert _repair_boundaries([ChapterBoundary(title="X", start_utterance=0)], 0) == []


# --- partition invariant: property-style over generated transcripts --------


def test_repair_and_spans_partition_utterances_property() -> None:
    """For many random (and deliberately malformed) boundary lists, the
    repaired spans must partition every utterance index exactly once:
    contiguous, non-overlapping, and covering the whole transcript.
    """

    rng = random.Random(20260823)

    for utterance_count in (1, 2, 3, 5, 10, 30):
        utterances = _utterances(utterance_count)
        for _trial in range(200):
            raw_boundary_count = rng.randint(1, utterance_count + 5)
            boundaries = [
                ChapterBoundary(
                    title=f"chapter-{i}",
                    # Deliberately out of range in both directions, and
                    # likely to collide with another boundary's index.
                    start_utterance=rng.randint(-5, utterance_count + 10),
                )
                for i in range(raw_boundary_count)
            ]

            repaired = _repair_boundaries(boundaries, utterance_count)
            spans = _spans_from_boundaries(repaired, utterances)

            assert spans, "a non-empty boundary list must never repair to no spans"

            covered: list[int] = []
            for span in spans:
                assert span.start_utterance <= span.end_utterance
                covered.extend(range(span.start_utterance, span.end_utterance + 1))

            # Exactly one chapter per utterance: no gaps, no overlaps, no
            # utterance outside the transcript's valid index range.
            assert covered == list(range(utterance_count))


# --- chapterise(): happy path and retry/raise behaviour ----------------------


async def test_chapterise_happy_path_returns_spans_from_boundaries() -> None:
    utterances = _utterances(4)
    transcript = RawTranscript(clips=[], utterances=utterances, audio_sha256=None)
    fake = RecordingQuery(
        {
            "chapters": [
                {"title": "Opening", "start_utterance": 0},
                {"title": "Closing", "start_utterance": 2},
            ]
        }
    )
    agent = ClaudeAgent(run_query=fake)

    chapters = await chapterise(transcript, agent=agent)

    assert chapters == [
        ChapterSpan(
            title="Opening", start_utterance=0, end_utterance=1, start_seconds=0.0
        ),
        ChapterSpan(
            title="Closing", start_utterance=2, end_utterance=3, start_seconds=60.0
        ),
    ]
    assert len(fake.calls) == 1
    # The prompt renders utterance index/speaker/text; timestamps are not
    # the model's job — indexes are the contract.
    prompt_text = fake.calls[0][0]
    assert "0 | SPEAKER_00: utterance 0" in prompt_text


async def test_chapterise_retries_once_on_a_degenerate_reply_then_succeeds() -> None:
    transcript = _transcript(4)
    fake = ScriptedQuery(
        {"chapters": []},  # degenerate: no boundaries at all
        {
            "chapters": [
                {"title": "First", "start_utterance": 0},
                {"title": "Second", "start_utterance": 2},
            ]
        },
    )
    agent = ClaudeAgent(run_query=fake)

    chapters = await chapterise(transcript, agent=agent)

    assert [c.title for c in chapters] == ["First", "Second"]
    assert len(fake.calls) == 2
    # The retry carries a corrective note the first prompt didn't have.
    first_prompt, second_prompt = (call[0] for call in fake.calls)
    assert "collapsed the whole meeting" not in first_prompt
    assert "collapsed the whole meeting" in second_prompt


async def test_chapterise_raises_when_still_degenerate_after_one_retry() -> None:
    transcript = _transcript(4)
    # Both calls collapse to a single chapter (one boundary covering
    # everything) - degenerate both times.
    fake = ScriptedQuery(
        {"chapters": [{"title": "Everything", "start_utterance": 0}]},
        {"chapters": [{"title": "Still everything", "start_utterance": 0}]},
    )
    agent = ClaudeAgent(run_query=fake)

    with pytest.raises(DegenerateChaptersError):
        await chapterise(transcript, agent=agent)

    # Exactly one retry - not fabricated boundaries, not an unbounded loop.
    assert len(fake.calls) == 2


# --- run_chapterisation: cache orchestration ---------------------------------


async def test_run_chapterisation_raises_without_a_cached_resolved_transcript(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    fake = RecordingQuery({"chapters": []})
    agent = ClaudeAgent(run_query=fake)

    with pytest.raises(MissingResolvedTranscriptError) as excinfo:
        await run_chapterisation("run-1", agent=agent, cache=cache)

    assert "run-1" in str(excinfo.value)
    assert "transcript speakers" in str(excinfo.value)
    assert fake.calls == []  # never even asked the model


async def test_run_chapterisation_stores_chapters_json_and_returns_them(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.store("run-1", "resolved_transcript", _transcript(4))
    fake = RecordingQuery(
        {
            "chapters": [
                {"title": "Opening", "start_utterance": 0},
                {"title": "Closing", "start_utterance": 2},
            ]
        }
    )
    agent = ClaudeAgent(run_query=fake)

    chapters = await run_chapterisation("run-1", agent=agent, cache=cache)

    run_dir = cache.run_dir("run-1")
    # The artefact on disk must actually be `chapters.json`, stored via the
    # typed cache path like every sibling stage - not `chapters.txt` via
    # `store_text` (the bug this test now guards against).
    assert (run_dir / "chapters.json").exists()
    assert not (run_dir / "chapters.txt").exists()
    loaded = cache.load("run-1", "chapters", ChapterList)
    assert loaded is not None
    assert loaded.chapters == chapters


# --- CLI: delegation, prerequisite error, and the effort="low" default -----


def _cache_with_resolved_transcript(tmp_path: Path, count: int = 4) -> Path:
    cache_root = tmp_path / "cache"
    RunCache(cache_root).store("run-1", "resolved_transcript", _transcript(count))
    return cache_root


def test_chapterise_cli_delegates_to_run_chapterisation_and_prints_its_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_chapters = [
        ChapterSpan(
            title="Opening", start_utterance=0, end_utterance=1, start_seconds=0.0
        )
    ]
    captured: dict[str, object] = {}

    async def fake_run_chapterisation(
        run_id: str, *, agent: object, cache: object
    ) -> list[ChapterSpan]:
        captured["run_id"] = run_id
        captured["agent"] = agent
        captured["cache"] = cache
        return fake_chapters

    monkeypatch.setattr(transcript_cli, "run_chapterisation", fake_run_chapterisation)
    cache_root = tmp_path / "cache"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "chapterise",
            "--run-id",
            "run-1",
            "--cache-root",
            str(cache_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(
        CHAPTERS_ADAPTER.dump_json(fake_chapters)
    )
    assert captured["run_id"] == "run-1"


def test_chapterise_cli_reports_domain_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def raising_run_chapterisation(
        run_id: str, *, agent: object, cache: object
    ) -> list[ChapterSpan]:
        raise MissingResolvedTranscriptError(run_id)

    monkeypatch.setattr(
        transcript_cli, "run_chapterisation", raising_run_chapterisation
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "chapterise",
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "run-1" in result.output
    assert "transcript speakers" in result.output


def test_chapterise_cli_reports_claude_agent_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def raising_run_chapterisation(
        run_id: str, *, agent: object, cache: object
    ) -> list[ChapterSpan]:
        raise ClaudeAgentError("agent call failed")

    monkeypatch.setattr(
        transcript_cli, "run_chapterisation", raising_run_chapterisation
    )

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "chapterise",
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "agent call failed" in result.output


def test_chapterise_cli_exits_nonzero_with_a_prerequisite_hint_for_a_missing_cache(
    tmp_path: Path,
) -> None:
    # No `resolved_transcript.json` stored for this run id: exercise the
    # real `run_chapterisation`/`RunCache` path end to end (no monkeypatch),
    # so this proves the CLI surfaces the *real* missing-prerequisite error,
    # naming `transcript speakers` as the command to run first. The agent
    # is never consulted on this path, so a real (unused) `ClaudeAgent` is
    # safe to construct.
    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "chapterise",
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path / "cache"),
        ],
    )

    assert result.exit_code != 0
    assert "run-1" in result.output
    assert "transcript speakers" in result.output


def test_chapterise_cli_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["transcript", "chapterise", "--help"])

    assert result.exit_code == 0
    assert "chapter" in result.output.lower()


def test_chapterise_cli_defaults_effort_to_low_when_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_chapterisation(
        run_id: str, *, agent: ClaudeAgent, cache: object
    ) -> list[ChapterSpan]:
        captured["effort"] = agent.defaults.effort
        captured["model"] = agent.defaults.model
        return []

    monkeypatch.setattr(transcript_cli, "run_chapterisation", fake_run_chapterisation)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "chapterise",
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["effort"] == "low"


def test_chapterise_cli_respects_an_explicit_effort_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_chapterisation(
        run_id: str, *, agent: ClaudeAgent, cache: object
    ) -> list[ChapterSpan]:
        captured["effort"] = agent.defaults.effort
        return []

    monkeypatch.setattr(transcript_cli, "run_chapterisation", fake_run_chapterisation)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "chapterise",
            "--run-id",
            "run-1",
            "--cache-root",
            str(tmp_path),
            "--effort",
            "high",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["effort"] == "high"
