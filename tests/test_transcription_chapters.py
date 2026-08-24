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

from jake_tools.claude import AgentSpec, ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache, StageManifest
from jake_tools.transcription.chapters import (
    CHAPTERS_ADAPTER,
    MIN_CHAPTER_SECONDS,
    ChapterBoundary,
    ChapterList,
    DegenerateChaptersError,
    MissingResolvedTranscriptError,
    _merge_short_spans,
    _render_utterances,
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


# --- _render_utterances: the duration signal ---------------------------------


def test_render_utterances_includes_elapsed_time_not_just_index() -> None:
    utterances = [
        Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="hi"),
        Utterance(start=125.0, end=126.0, speaker="SPEAKER_01", text="two minutes in"),
        Utterance(start=3661.0, end=3662.0, speaker="SPEAKER_00", text="an hour plus"),
    ]

    rendered = _render_utterances(utterances)

    assert "0 | 00:00 | SPEAKER_00: hi" in rendered
    assert "1 | 02:05 | SPEAKER_01: two minutes in" in rendered
    assert "2 | 1:01:01 | SPEAKER_00: an hour plus" in rendered


# --- _merge_short_spans: the too-fine mechanical backstop --------------------


def _utterances_with_last_end(count: int, last_end: float) -> list[Utterance]:
    """`count` filler utterances; only the last one's `.end` is load-bearing
    for `_span_duration` (used for the final span's duration)."""

    return [
        Utterance(
            start=float(i), end=float(i) + 0.5, speaker="SPEAKER_00", text=f"u{i}"
        )
        for i in range(count - 1)
    ] + [
        Utterance(
            start=float(count - 1), end=last_end, speaker="SPEAKER_00", text="last"
        )
    ]


def test_merge_short_spans_merges_a_short_chapter_into_its_shorter_neighbour() -> None:
    # Durations: A=100s, B=30s (short), C=130s, D=140s (via last_end=400).
    # B's neighbours are A (100s) and C (130s) - the shorter one, A, wins.
    spans = [
        ChapterSpan(title="A", start_utterance=0, end_utterance=1, start_seconds=0.0),
        ChapterSpan(title="B", start_utterance=2, end_utterance=2, start_seconds=100.0),
        ChapterSpan(title="C", start_utterance=3, end_utterance=4, start_seconds=130.0),
        ChapterSpan(title="D", start_utterance=5, end_utterance=5, start_seconds=260.0),
    ]
    utterances = _utterances_with_last_end(6, last_end=400.0)

    merged = _merge_short_spans(spans, utterances)

    assert [span.title for span in merged] == ["A", "C", "D"]
    # B dissolved into A: A's span now covers B's utterances too, keeping
    # A's own title and start rather than fabricating a new one.
    assert merged[0] == ChapterSpan(
        title="A", start_utterance=0, end_utterance=2, start_seconds=0.0
    )
    covered: list[int] = []
    for span in merged:
        covered.extend(range(span.start_utterance, span.end_utterance + 1))
    assert covered == list(range(6))


def test_merge_short_spans_never_merges_below_two_chapters() -> None:
    # Both chapters are short (well under MIN_CHAPTER_SECONDS), but there
    # are only two - the floor that mirrors `_RETRY_NOTE`'s "at least two
    # boundaries" guard on the too-coarse side.
    spans = [
        ChapterSpan(title="A", start_utterance=0, end_utterance=0, start_seconds=0.0),
        ChapterSpan(title="B", start_utterance=1, end_utterance=1, start_seconds=10.0),
    ]
    utterances = _utterances_with_last_end(2, last_end=15.0)

    merged = _merge_short_spans(spans, utterances)

    assert merged == spans


def test_merge_short_spans_leaves_spans_at_or_above_the_floor_untouched() -> None:
    spans = [
        ChapterSpan(title="A", start_utterance=0, end_utterance=0, start_seconds=0.0),
        ChapterSpan(
            title="B",
            start_utterance=1,
            end_utterance=1,
            start_seconds=MIN_CHAPTER_SECONDS,
        ),
        ChapterSpan(
            title="C",
            start_utterance=2,
            end_utterance=2,
            start_seconds=MIN_CHAPTER_SECONDS * 2,
        ),
    ]
    utterances = _utterances_with_last_end(3, last_end=MIN_CHAPTER_SECONDS * 3)

    merged = _merge_short_spans(spans, utterances)

    assert merged == spans


def test_merge_short_spans_collapses_a_chain_of_short_chapters_to_the_two_chapter_floor() -> (
    None
):
    utterances = [
        Utterance(
            start=float(i) * 10.0,
            end=float(i) * 10.0 + 5.0,
            speaker="SPEAKER_00",
            text=f"u{i}",
        )
        for i in range(6)
    ]
    # Six chapters, each 10s apart - every one shorter than the 60s floor.
    spans = [
        ChapterSpan(
            title=f"C{i}",
            start_utterance=i,
            end_utterance=i,
            start_seconds=float(i) * 10.0,
        )
        for i in range(6)
    ]

    merged = _merge_short_spans(spans, utterances)

    assert len(merged) == 2  # the floor, not a duration guarantee here
    covered: list[int] = []
    for span in merged:
        assert span.start_utterance <= span.end_utterance
        covered.extend(range(span.start_utterance, span.end_utterance + 1))
    assert covered == list(range(6))  # partition invariant survives the merge


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
    # The prompt renders utterance index/elapsed-time/speaker/text - the
    # model is not asked to compute timestamps, but it does need to see
    # them to judge chapter length (the duration-blind rendering bug the
    # adversarial review found).
    prompt_text = fake.calls[0][0]
    assert "0 | 00:00 | SPEAKER_00: utterance 0" in prompt_text


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


async def test_chapterise_merges_llm_proposed_boundaries_shorter_than_the_minimum_span() -> (
    None
):
    """End-to-end: a model reply that over-splits (two chapters a few
    seconds apart) gets mechanically merged by `chapterise`, the same
    `_merge_short_spans` pass unit-tested above - the too-fine failure mode
    the adversarial review found nothing guarding against."""

    utterances = [
        Utterance(start=0.0, end=4.0, speaker="SPEAKER_00", text="quick a"),
        Utterance(start=5.0, end=9.0, speaker="SPEAKER_00", text="quick b"),
        Utterance(start=10.0, end=65.0, speaker="SPEAKER_00", text="middle"),
        Utterance(start=70.0, end=75.0, speaker="SPEAKER_00", text="tail 1"),
        Utterance(start=140.0, end=145.0, speaker="SPEAKER_00", text="tail 2"),
        Utterance(start=210.0, end=300.0, speaker="SPEAKER_00", text="tail 3"),
    ]
    transcript = RawTranscript(clips=[], utterances=utterances, audio_sha256=None)
    fake = RecordingQuery(
        {
            "chapters": [
                {"title": "Quick note A", "start_utterance": 0},
                {"title": "Quick note B", "start_utterance": 1},
                {"title": "Middle chunk", "start_utterance": 2},
                {"title": "Long tail", "start_utterance": 3},
            ]
        }
    )
    agent = ClaudeAgent(run_query=fake)

    chapters = await chapterise(transcript, agent=agent)

    # Four proposed chapters, two of them 5s long - merged down to two,
    # both clearing MIN_CHAPTER_SECONDS.
    assert chapters == [
        ChapterSpan(
            title="Middle chunk", start_utterance=0, end_utterance=2, start_seconds=0.0
        ),
        ChapterSpan(
            title="Long tail", start_utterance=3, end_utterance=5, start_seconds=70.0
        ),
    ]


# --- slow: real-LLM integration test ------------------------------------------
#
# Everything above proves the deterministic post-processing is right
# (`_repair_boundaries`/`_spans_from_boundaries` guarantee the partition
# invariant regardless of what the model returns) using a fake agent that
# only ever replays what the test already wrote down. None of it proves the
# chapterisation prompt actually finds real topic boundaries. This makes
# real `claude-sonnet-5` calls (`--slow`, skipped by default - see
# `pyproject.toml`'s `slow` marker, `effort="low"` to bound cost) against a
# deliberately messy raw-ASR fixture with three clearly distinct topics, and
# checks observable properties of the real reply rather than exact
# boundaries, since LLM output is nondeterministic. Run with
# `uv run pytest --slow -k slow tests/test_transcription_chapters.py`.

_THREE_TOPIC_UTTERANCES = [
    # Topic 1 (indexes 0-9): Q3 budget review.
    "Okay, so, um, let's kick off with the budget review for Q3, uh, spend is tracking about eight percent over plan.",
    "Eight percent, that's, uh, mostly the vendor contract right?",
    "Yeah, the ven- the vendor contract renewal came in higher than, uh, we forecast.",
    "Okay. Do we need to, um, flag that to finance this week?",
    "I think so, yeah. I'll send the, uh, variance report over today.",
    "Cool. Anything else on the budget side before we move on?",
    "Just that the marketing line item is under spend, so it kind of nets out.",
    "Good, good. Okay, uh, I think that covers the budget stuff.",
    "Yep, agreed. Q3 numbers are basically on track once you net it out.",
    "Great, let's, uh, move on then.",
    # Topic 2 (indexes 10-19): backend hiring plan.
    "So, uh, next up, hiring. We've got two open reqs for backend engineers.",
    "Right, and, um, one of those is backfilling Sam's role, yeah?",
    "Correct, and the other is, uh, a net-new headcount for the platform team.",
    "Okay. How's the, uh, the candidate pipeline looking so far?",
    "We've got three candidates in final rounds, should have offers out by, uh, next Friday.",
    "Nice. Do we, um, need budget approval for the net-new one still?",
    "No, that was already approved back in, uh, the Q2 planning cycle.",
    "Perfect, that makes it easier. Any concerns from the, uh, hiring managers?",
    "Not really, just the usual, uh, timeline pressure to close before the holidays.",
    "Makes sense. Okay, I think hiring's in good shape.",
    # Topic 3 (indexes 20-29): the office move.
    "Last thing, uh, the office move. Facilities confirmed the new floor is ready December first.",
    "December first, okay. Is that, um, still the date we told everyone?",
    "Yeah, that lines up with what went out in the, uh, all-hands email.",
    "Good. Do we need to, uh, coordinate the IT setup separately?",
    "IT's already scheduled to do the network drops the week before, so we're covered.",
    "Great, and what about, um, the parking situation at the new building?",
    "There's a, uh, dedicated garage, should be more spots than we had before, honestly.",
    "That's a relief, honestly. Okay, uh, anything else on the move?",
    "Just that we should, uh, send a reminder email the week before the move.",
    "Agreed, I'll draft that. Okay, uh, I think that's everything for today.",
]


def _three_topic_transcript() -> RawTranscript:
    utterances = [
        Utterance(
            start=float(index) * 20.0,
            end=float(index) * 20.0 + 18.0,
            speaker="SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
            text=text,
        )
        for index, text in enumerate(_THREE_TOPIC_UTTERANCES)
    ]
    return RawTranscript(clips=[], utterances=utterances, audio_sha256=None)


_GENERIC_TITLES = {
    "discussion",
    "introduction",
    "general discussion",
    "meeting",
    "chapter",
    "untitled",
    "misc",
    "miscellaneous",
}


@pytest.mark.slow
async def test_live_chapterise_splits_three_distinct_topics_into_different_chapters() -> (
    None
):
    transcript = _three_topic_transcript()
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))

    chapters = await chapterise(transcript, agent=agent)

    assert 2 <= len(chapters) <= 5

    # Partition invariant: already enforced deterministically in code
    # (`_repair_boundaries`/`_spans_from_boundaries`), but assert it holds
    # for this real reply too.
    covered: list[int] = []
    for span in chapters:
        assert span.start_utterance <= span.end_utterance
        covered.extend(range(span.start_utterance, span.end_utterance + 1))
    assert covered == list(range(len(_THREE_TOPIC_UTTERANCES)))

    def chapter_index_of(utterance_index: int) -> int:
        for chapter_index, span in enumerate(chapters):
            if span.start_utterance <= utterance_index <= span.end_utterance:
                return chapter_index
        raise AssertionError(f"utterance {utterance_index} not covered by any chapter")

    # A representative utterance from the middle of each topic - robust to
    # the model landing a boundary a few utterances early or late, but still
    # requires the three topics to end up in genuinely different chapters.
    topic_chapters = {
        chapter_index_of(4),  # mid budget review
        chapter_index_of(14),  # mid hiring plan
        chapter_index_of(24),  # mid office move
    }
    assert len(topic_chapters) == 3, (
        f"the three distinct topics did not land in three distinct chapters: "
        f"{[(span.title, span.start_utterance, span.end_utterance) for span in chapters]!r}"
    )

    for span in chapters:
        title = span.title.strip()
        assert title != ""
        assert title.casefold() not in _GENERIC_TITLES

    print("chapters:", [span.model_dump() for span in chapters])


# A 20-utterance, ~20-second crosstalk burst about one narrow topic
# (clarifying a budget number), immediately followed by a five-utterance,
# ~220-second monologue about a completely different topic (a hiring
# plan), then a short three-utterance wrap-up. Before the elapsed-time
# rendering fix, the burst's utterance *count* (20, more than the
# monologue's 5) was the model's only proxy for "how big a topic is" -
# exactly backwards from its actual ~20s duration versus the monologue's
# ~220s. This is the acceptance run's own failure shape (chapter 6: 29
# utterances/73s vs chapter 10: 195 utterances/608s), reproduced small.
_BURST_LINES = [
    "Wait, that's how much?",
    "Eight percent.",
    "Over plan?",
    "Over plan, yeah.",
    "Eight percent over plan then.",
    "Right.",
    "That's rough.",
    "Yeah, it is.",
    "Is that the vendor contract thing?",
    "The vendor contract thing.",
    "Same as last time?",
    "Basically the same.",
    "Okay, noted.",
    "Yep.",
    "Moving on then?",
    "Moving on.",
    "Sure, go ahead.",
    "Go ahead.",
    "Okay.",
    "Right, so, noted.",
]
_MONOLOGUE_LINES = [
    "So switching to hiring - we've got two open reqs for backend "
    "engineers, one of which is backfilling Sam's role and the other is a "
    "net-new headcount for the platform team that finance already "
    "approved back in the Q2 planning cycle.",
    "We've got three candidates through the first round of interviews, "
    "two of whom look strong on embedded systems work and one who's more "
    "of a generalist, and we should have final offers ready to go out by "
    "the end of next week.",
    "The main risk right now is timeline pressure to close everything out "
    "before the holidays, since a lot of candidates tend to go quiet once "
    "December hits and start dates get pushed into the new year.",
    "Facilities also confirmed separately that they can onboard two new "
    "desks on the current floor without needing to expand into the annex, "
    "so seating isn't going to be a blocker for whichever candidates we "
    "bring on.",
    "I'll send round the updated headcount tracker after this call so "
    "everyone can see exactly where the two open reqs stand and who's "
    "covering each interview loop this week.",
]
_WRAP_LINES = ["Okay, anything else?", "Nothing from me.", "Great, let's wrap here."]

_BURST_COUNT = len(_BURST_LINES)
_MONOLOGUE_START = _BURST_COUNT
_WRAP_START = _MONOLOGUE_START + len(_MONOLOGUE_LINES)
_TOTAL_UTTERANCES = _WRAP_START + len(_WRAP_LINES)


def _burst_then_monologue_transcript() -> RawTranscript:
    utterances: list[Utterance] = []
    for i, text in enumerate(_BURST_LINES):
        start = float(i)
        utterances.append(
            Utterance(
                start=start,
                end=start + 0.9,
                speaker="SPEAKER_00" if i % 2 == 0 else "SPEAKER_01",
                text=text,
            )
        )
    monologue_starts = [25.0, 70.0, 115.0, 160.0, 205.0]
    for start, text in zip(monologue_starts, _MONOLOGUE_LINES, strict=True):
        utterances.append(
            Utterance(start=start, end=start + 40.0, speaker="SPEAKER_00", text=text)
        )
    wrap_starts = [250.0, 255.0, 260.0]
    for i, (start, text) in enumerate(zip(wrap_starts, _WRAP_LINES, strict=True)):
        utterances.append(
            Utterance(
                start=start,
                end=start + 3.0,
                speaker="SPEAKER_00" if i % 2 == 0 else "SPEAKER_01",
                text=text,
            )
        )
    return RawTranscript(clips=[], utterances=utterances, audio_sha256=None)


@pytest.mark.slow
async def test_live_chapterise_keeps_a_short_crosstalk_burst_in_one_chapter() -> None:
    transcript = _burst_then_monologue_transcript()
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))

    chapters = await chapterise(transcript, agent=agent)

    # Partition invariant, for this real reply too.
    covered: list[int] = []
    for span in chapters:
        assert span.start_utterance <= span.end_utterance
        covered.extend(range(span.start_utterance, span.end_utterance + 1))
    assert covered == list(range(_TOTAL_UTTERANCES))

    def chapter_index_of(utterance_index: int) -> int:
        for index, span in enumerate(chapters):
            if span.start_utterance <= utterance_index <= span.end_utterance:
                return index
        raise AssertionError(f"utterance {utterance_index} not covered by any chapter")

    # The short crosstalk burst must land in exactly one chapter, not be
    # fragmented because it happens to have more utterances than the much
    # longer monologue that follows it.
    burst_chapters = {chapter_index_of(i) for i in range(_BURST_COUNT)}
    assert len(burst_chapters) == 1, (
        f"the ~20s crosstalk burst was split across multiple chapters: "
        f"{[(s.title, s.start_utterance, s.end_utterance, s.start_seconds) for s in chapters]!r}"
    )

    # The hiring monologue is its own chapter, distinct from the burst -
    # not merged away as if it were the small one.
    monologue_chapter = chapter_index_of(_MONOLOGUE_START + 2)
    assert monologue_chapter not in burst_chapters

    print("chapters:", [span.model_dump() for span in chapters])


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


async def test_run_chapterisation_removes_stale_output_before_failed_regeneration(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.store("run-1", "resolved_transcript", _transcript(4))
    cache.store("run-1", "chapters", ChapterList(chapters=[]))
    cache.store_manifest(
        "run-1", StageManifest(stage="chapterise", input_hash="old", config_hash="old")
    )
    fake = ScriptedQuery(None, None)
    agent = ClaudeAgent(run_query=fake)

    with pytest.raises(ClaudeAgentError):
        await run_chapterisation("run-1", agent=agent, cache=cache)

    assert cache.load("run-1", "chapters", ChapterList) is None
    assert cache.load_manifest("run-1", "chapterise") is None


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
