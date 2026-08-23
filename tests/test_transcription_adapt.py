"""Behaviour of the pre-diarised transcript adapter (`transcription/adapt.py`)
and the `jake-tools transcript adapt` CLI command.

Per the plan, the two deterministic parsers (`parse_vtt`, `parse_named_lines`)
carry the real test weight as pure functions. `adapt_transcript`'s routing is
then tested with an injected `ClaudeAgent` built on a fake `run_query`
(pattern: `RecordingQuery`, `tests/test_claude_agent.py:58-79`) — asserting
the deterministic parsers are preferred whenever they match (the LLM fake is
never called), and that the LLM fallback is used, and used correctly, only
when neither parser recognises the document. The CLI test stays thin: flag
parsing -> options object -> delegation, monkeypatching `adapt_transcript`,
mirroring `test_transcript_cli.py` and `test_transcription_asr.py`.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    TextBlock,
)
from click.testing import CliRunner

from jake_tools.claude import AgentSpec, ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.adapt import (
    NoFallbackAgentError,
    parse_named_lines,
    parse_vtt,
)
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.models import RawTranscript, Utterance

# `jake_tools.cli`'s __init__ rebinds the name `transcript` to the Click
# group, shadowing the submodule (see `test_transcript_cli.py`) — fetch the
# actual module via importlib to monkeypatch its `adapt_transcript` binding.
transcript_cli = importlib.import_module("jake_tools.cli.transcript")
adapt_module = importlib.import_module("jake_tools.transcription.adapt")


# --- parse_vtt: deterministic WebVTT parsing --------------------------------


def test_parse_vtt_extracts_speakers_and_second_resolution_timestamps() -> None:
    text = (
        "WEBVTT\n"
        "\n"
        "1\n"
        "00:00:01.500 --> 00:00:04.000\n"
        "<v Jane Doe>Hello everyone, thanks for joining.</v>\n"
        "\n"
        "2\n"
        "00:00:04.000 --> 00:00:07.250\n"
        "<v John Smith>Happy to be here.</v>\n"
    )

    utterances = parse_vtt(text)

    assert utterances == [
        Utterance(
            start=1.5,
            end=4.0,
            speaker="Jane Doe",
            text="Hello everyone, thanks for joining.",
        ),
        Utterance(start=4.0, end=7.25, speaker="John Smith", text="Happy to be here."),
    ]


def test_parse_vtt_handles_hour_component_and_no_cue_identifier() -> None:
    text = (
        "WEBVTT\n"
        "\n"
        "01:02:03.000 --> 01:02:05.000\n"
        "<v Speaker 1>a long meeting indeed</v>\n"
    )

    utterances = parse_vtt(text)

    assert utterances == [
        Utterance(
            start=3723.0, end=3725.0, speaker="Speaker 1", text="a long meeting indeed"
        )
    ]


def test_parse_vtt_defaults_a_cue_with_no_voice_tag_to_unknown_speaker() -> None:
    text = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nno voice tag here\n"

    utterances = parse_vtt(text)

    assert utterances == [
        Utterance(start=0.0, end=1.0, speaker="Unknown", text="no voice tag here")
    ]


def test_parse_vtt_returns_none_for_text_with_no_webvtt_header() -> None:
    assert parse_vtt("Jane Doe: hello\nJohn Smith: hi\n") is None
    assert parse_vtt("just some prose, no structure at all.\n") is None


# A synthetic fixture faithful to the structure of a real Teams `.vtt` export
# Jake (Michael's Hermes agent) retrieved via `hermes chat` and described
# during this plan's implementation, 2026-08-23: opaque UUID cue identifiers
# before the timing line, cue text wrapped across multiple physical lines,
# and a long utterance split across several consecutive same-speaker cues
# (`/53-0` etc. — not merged back together, same as ASR's segment-per-cue
# philosophy in `asr.align`). Structure verified against the real sample;
# the UUID and dialogue content here are synthesised, not copied from it.
_TEAMS_VTT_SAMPLE = """WEBVTT

00000000-0000-4000-8000-000000000000/5-0
00:00:03.456 --> 00:00:03.976
<v Speaker A>Yeah, there we go.</v>

00000000-0000-4000-8000-000000000000/6-0
00:00:05.656 --> 00:00:06.056
<v Speaker B>It'll break.</v>

00000000-0000-4000-8000-000000000000/20-0
00:00:19.310 --> 00:00:22.750
<v Speaker A>Yeah. Good. How are you? Yep.
You're coming from this weekend.</v>

00000000-0000-4000-8000-000000000000/53-0
00:00:23.830 --> 00:00:29.882
<v Speaker B>Oh God, it was a long weekend.
I'm gonna give you that. Like, yeah, like,</v>

00000000-0000-4000-8000-000000000000/24-0
00:00:25.990 --> 00:00:26.110
<v Speaker A>Yeah.</v>
"""


def test_parse_vtt_handles_a_teams_export_shaped_sample() -> None:
    utterances = parse_vtt(_TEAMS_VTT_SAMPLE)
    assert utterances is not None

    assert utterances == [
        Utterance(
            start=3.456, end=3.976, speaker="Speaker A", text="Yeah, there we go."
        ),
        Utterance(start=5.656, end=6.056, speaker="Speaker B", text="It'll break."),
        Utterance(
            start=19.31,
            end=22.75,
            speaker="Speaker A",
            text="Yeah. Good. How are you? Yep. You're coming from this weekend.",
        ),
        Utterance(
            start=23.83,
            end=29.882,
            speaker="Speaker B",
            text="Oh God, it was a long weekend. I'm gonna give you that. Like, yeah, like,",
        ),
        Utterance(start=25.99, end=26.11, speaker="Speaker A", text="Yeah."),
    ]
    # The two trailing cues genuinely overlap in time (23.83-29.882 and
    # 25.99-26.11) — preserved, not clamped or merged away.
    assert utterances[3].end > utterances[4].start


# --- parse_named_lines: deterministic plain-text parsing --------------------


def test_parse_named_lines_returns_utterances_in_order_with_zero_timestamps() -> None:
    text = (
        "Jane Doe: Hello everyone, thanks for joining.\n"
        "John Smith: Happy to be here.\n"
        "Jane Doe: Great, let's get started.\n"
    )

    utterances = parse_named_lines(text)

    assert utterances == [
        Utterance(
            start=0.0,
            end=0.0,
            speaker="Jane Doe",
            text="Hello everyone, thanks for joining.",
        ),
        Utterance(start=0.0, end=0.0, speaker="John Smith", text="Happy to be here."),
        Utterance(
            start=0.0, end=0.0, speaker="Jane Doe", text="Great, let's get started."
        ),
    ]


def test_parse_named_lines_parses_an_inline_timestamp() -> None:
    text = "Jane Doe (00:00:05): Hello.\nJohn Smith (00:01:10): Hi there.\n"

    utterances = parse_named_lines(text)

    assert utterances == [
        Utterance(start=5.0, end=5.0, speaker="Jane Doe", text="Hello."),
        Utterance(start=70.0, end=70.0, speaker="John Smith", text="Hi there."),
    ]


def test_parse_named_lines_joins_wrapped_continuation_lines() -> None:
    text = "Jane Doe: Hello everyone,\nthanks for joining.\nJohn Smith: Happy to be here.\n"

    utterances = parse_named_lines(text)

    assert utterances == [
        Utterance(
            start=0.0,
            end=0.0,
            speaker="Jane Doe",
            text="Hello everyone, thanks for joining.",
        ),
        Utterance(start=0.0, end=0.0, speaker="John Smith", text="Happy to be here."),
    ]


def test_parse_named_lines_returns_none_when_the_first_line_has_no_speaker_prefix() -> (
    None
):
    text = "Meeting notes for 2026-01-01\nJane Doe: Hello everyone.\n"

    assert parse_named_lines(text) is None


def test_parse_named_lines_returns_none_for_text_with_no_colon_lines() -> None:
    text = "This is just a paragraph of prose with no speaker markers at all here.\n"

    assert parse_named_lines(text) is None


# --- adapt_transcript: routing -----------------------------------------------


class _RecordingQuery:
    """A fake `run_query` that records calls and replays fixed messages."""

    def __init__(self, *messages: Message) -> None:
        self.messages = messages
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))

        async def stream() -> AsyncIterator[Message]:
            for message in self.messages:
                yield message

        return stream()


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


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model="claude-sonnet-5")


async def test_adapt_transcript_prefers_the_vtt_parser_over_the_llm(
    tmp_path: Path,
) -> None:
    fake = _RecordingQuery(
        _structured_result(
            {"utterances": [{"start": 0.0, "end": 0.0, "speaker": "x", "text": "y"}]}
        )
    )
    agent = ClaudeAgent(run_query=fake)
    path = tmp_path / "transcript.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<v Jane Doe>hi there</v>\n"
    )

    result = await adapt_module.adapt_transcript(path, agent=agent)

    assert result.utterances == [
        Utterance(start=0.0, end=1.0, speaker="Jane Doe", text="hi there")
    ]
    assert fake.calls == []  # the LLM fallback must never be reached


async def test_adapt_transcript_prefers_the_named_lines_parser_over_the_llm(
    tmp_path: Path,
) -> None:
    fake = _RecordingQuery(
        _structured_result(
            {"utterances": [{"start": 0.0, "end": 0.0, "speaker": "x", "text": "y"}]}
        )
    )
    agent = ClaudeAgent(run_query=fake)
    path = tmp_path / "transcript.txt"
    path.write_text("Jane Doe: hi there\nJohn Smith: hello\n")

    result = await adapt_module.adapt_transcript(path, agent=agent)

    assert result.utterances == [
        Utterance(start=0.0, end=0.0, speaker="Jane Doe", text="hi there"),
        Utterance(start=0.0, end=0.0, speaker="John Smith", text="hello"),
    ]
    assert fake.calls == []  # the LLM fallback must never be reached


async def test_adapt_transcript_falls_back_to_the_llm_for_an_unparseable_document(
    tmp_path: Path,
) -> None:
    payload = {
        "utterances": [
            {
                "start": 0.0,
                "end": 3.0,
                "speaker": "Speaker 1",
                "text": "a rambling intro",
            },
            {
                "start": 3.0,
                "end": 6.0,
                "speaker": "Speaker 2",
                "text": "a rambling reply",
            },
        ]
    }
    fake = _RecordingQuery(_assistant("..."), _structured_result(payload))
    agent = ClaudeAgent(run_query=fake)
    path = tmp_path / "transcript.txt"
    path.write_text(
        "This is just a paragraph of prose with no speaker markers at all, "
        "spanning several sentences and never once using a colon to introduce "
        "anyone by name.\n"
    )

    result = await adapt_module.adapt_transcript(path, agent=agent)

    assert result.utterances == [
        Utterance(start=0.0, end=3.0, speaker="Speaker 1", text="a rambling intro"),
        Utterance(start=3.0, end=6.0, speaker="Speaker 2", text="a rambling reply"),
    ]
    assert result.clips == []
    assert result.audio_sha256 is None
    assert len(fake.calls) == 1
    prompt_text = fake.calls[0][0]
    assert "restructure" in prompt_text.lower()
    assert (
        "a paragraph of prose" in prompt_text
    )  # the raw document is embedded verbatim


# A synthetic fixture faithful to the structure of a real Google Meet "Notes
# by Gemini" transcript document Jake retrieved via `hermes chat` and
# described during this plan's implementation, 2026-08-23. It's *not*
# one-line-per-turn like the brief hypothesised: a `### HH:MM:SS` heading
# covers a whole block of dialogue, and multiple speakers' bold `**Name:**`
# turns appear inline in one paragraph rather than one per line. Both
# deterministic parsers correctly refuse it (`parse_vtt`: no `WEBVTT`
# header; `parse_named_lines`: the first non-blank line is a markdown
# heading, not a speaker line) — this is exactly the shape the LLM fallback
# exists for: real structure, verbatim words, just not fixed-pattern-
# parseable. Structure verified against the real sample; the date/time and
# dialogue content here are synthesised, not copied from it.
_GEMINI_EXCERPT_SAMPLE = (
    "# \U0001f4d6 Transcript\n"
    "\n"
    "### Jan 1, 2031\n"
    "\n"
    "## Meeting Jan 1, 2031 at 12:00 UTC-Transcript\n"
    "\n"
    "### 00:00:01\n"
    "\n"
    "**Speaker A:** Hey, Speaker B. How are you? **Speaker B:** Yeah, I'm "
    "not too bad. What about you? **Speaker A:** I'm good. **Speaker B:** "
    "Good to hear. **Speaker A:** Yeah. **Speaker B:** Um, ...\n"
)


def test_gemini_excerpt_sample_is_rejected_by_both_deterministic_parsers() -> None:
    assert parse_vtt(_GEMINI_EXCERPT_SAMPLE) is None
    assert parse_named_lines(_GEMINI_EXCERPT_SAMPLE) is None


async def test_adapt_transcript_sends_the_gemini_excerpt_sample_verbatim_to_the_llm(
    tmp_path: Path,
) -> None:
    payload = {
        "utterances": [
            {
                "start": 1.0,
                "end": 1.0,
                "speaker": "Speaker A",
                "text": "Hey, Speaker B. How are you?",
            },
            {
                "start": 1.0,
                "end": 1.0,
                "speaker": "Speaker B",
                "text": "Yeah, I'm not too bad. What about you?",
            },
        ]
    }
    fake = _RecordingQuery(_structured_result(payload))
    agent = ClaudeAgent(run_query=fake)
    path = tmp_path / "gemini_notes.md"
    path.write_text(_GEMINI_EXCERPT_SAMPLE)

    result = await adapt_module.adapt_transcript(path, agent=agent)

    assert len(fake.calls) == 1
    prompt_text = fake.calls[0][0]
    # The raw inline dialogue reaches the model unmodified — no re-wrapping,
    # no stripping of the bold markers, no truncation.
    assert "**Speaker A:** Hey, Speaker B. How are you?" in prompt_text
    assert "**Speaker B:** Good to hear." in prompt_text
    assert result.utterances == [
        Utterance(
            start=1.0, end=1.0, speaker="Speaker A", text="Hey, Speaker B. How are you?"
        ),
        Utterance(
            start=1.0,
            end=1.0,
            speaker="Speaker B",
            text="Yeah, I'm not too bad. What about you?",
        ),
    ]


async def test_adapt_transcript_raises_without_an_agent_when_nothing_parses(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transcript.txt"
    path.write_text("no colons, no WEBVTT header, nothing structured at all here.\n")

    with pytest.raises(NoFallbackAgentError):
        await adapt_module.adapt_transcript(path, agent=None)


# --- slow: real-LLM integration test ------------------------------------------
#
# Everything above proves the *routing* is right (deterministic parsers
# preferred, LLM fallback reached only when neither matches, the raw
# document embedded verbatim in the prompt). None of it proves the LLM
# fallback's prompt actually teaches the model to restructure without
# paraphrasing - a fake only ever replays what the test already wrote down.
# This makes one real `claude-sonnet-5` call (`--slow`, skipped by default -
# see `pyproject.toml`'s `slow` marker) against the Gemini-shaped sample
# above and checks observable properties of the real reply: non-empty
# utterances, the exact speaker-label set from the source, non-decreasing
# ordering, and - the load-bearing rule for this stage - that a handful of
# distinctive phrases from the source survive as utterance text verbatim,
# since a paraphrasing adapter would poison the pipeline's factual record.
# Run with `uv run pytest --slow -k slow tests/test_transcription_adapt.py`.


@pytest.mark.slow
async def test_live_adapt_transcript_restructures_the_gemini_excerpt_preserving_wording(
    tmp_path: Path,
) -> None:
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))
    path = tmp_path / "gemini_notes.md"
    path.write_text(_GEMINI_EXCERPT_SAMPLE)

    result = await adapt_module.adapt_transcript(path, agent=agent)

    assert result.clips == []
    assert result.audio_sha256 is None
    assert len(result.utterances) > 0

    # Speaker labels pass through verbatim: exactly the label set present in
    # the source, no invented or dropped speakers.
    speakers = {u.speaker for u in result.utterances}
    assert speakers == {"Speaker A", "Speaker B"}

    # Non-decreasing start ordering, per the module's documented contract.
    starts = [u.start for u in result.utterances]
    assert starts == sorted(starts)

    # Wording preserved verbatim - restructuring must never paraphrase. Pick
    # a handful of distinctive phrases straight from the source and require
    # each to survive, unaltered, as some utterance's text.
    all_text = " ".join(u.text for u in result.utterances)
    for phrase in (
        "Hey, Speaker B. How are you?",
        "I'm not too bad",
        "I'm good",
        "Good to hear",
    ):
        assert phrase in all_text, (
            f"{phrase!r} missing from adapted text (paraphrased or dropped): "
            f"{all_text!r}"
        )

    print("adapted transcript:", result.model_dump_json(indent=2))


# --- CLI: `jake-tools transcript adapt` -------------------------------------


def test_adapt_cli_delegates_to_adapt_transcript_and_prints_its_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    fake_result = RawTranscript(
        clips=[],
        utterances=[Utterance(start=0.0, end=1.0, speaker="Jane Doe", text="hi there")],
        audio_sha256=None,
    )

    async def fake_adapt_transcript(path: Path, *, agent: object) -> RawTranscript:
        captured["path"] = path
        captured["agent"] = agent
        return fake_result

    monkeypatch.setattr(transcript_cli, "adapt_transcript", fake_adapt_transcript)
    transcript_path = tmp_path / "transcript.vtt"
    transcript_path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<v Jane Doe>hi there</v>\n"
    )

    result = CliRunner().invoke(main, ["transcript", "adapt", str(transcript_path)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(fake_result.model_dump_json())
    assert captured["path"] == transcript_path
    assert isinstance(captured["agent"], ClaudeAgent)


def test_adapt_cli_stores_result_in_run_cache_when_run_id_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_result = RawTranscript(
        clips=[],
        utterances=[Utterance(start=0.0, end=0.0, speaker="Jane Doe", text="hi")],
        audio_sha256=None,
    )

    async def fake_adapt_transcript(path: Path, *, agent: object) -> RawTranscript:
        return fake_result

    monkeypatch.setattr(transcript_cli, "adapt_transcript", fake_adapt_transcript)
    transcript_path = tmp_path / "transcript.vtt"
    transcript_path.write_text("WEBVTT\n")
    cache_root = tmp_path / "cache"

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "adapt",
            str(transcript_path),
            "--run-id",
            "meeting-1",
            "--cache-root",
            str(cache_root),
        ],
    )

    assert result.exit_code == 0, result.output
    cached = RunCache(root=cache_root).load(
        "meeting-1", "raw_transcript", RawTranscript
    )
    assert cached == fake_result


def test_adapt_cli_does_not_write_a_cache_entry_without_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_result = RawTranscript(clips=[], utterances=[], audio_sha256=None)

    async def fake_adapt_transcript(path: Path, *, agent: object) -> RawTranscript:
        return fake_result

    monkeypatch.setattr(transcript_cli, "adapt_transcript", fake_adapt_transcript)
    transcript_path = tmp_path / "transcript.vtt"
    transcript_path.write_text("WEBVTT\n")
    cache_root = tmp_path / "cache"

    result = CliRunner().invoke(
        main,
        ["transcript", "adapt", str(transcript_path), "--cache-root", str(cache_root)],
    )

    assert result.exit_code == 0, result.output
    assert not cache_root.exists()


def test_adapt_cli_reports_adapt_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def raising_adapt_transcript(path: Path, *, agent: object) -> RawTranscript:
        raise adapt_module.NoFallbackAgentError(path)

    monkeypatch.setattr(transcript_cli, "adapt_transcript", raising_adapt_transcript)
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text("no structure here at all\n")

    result = CliRunner().invoke(main, ["transcript", "adapt", str(transcript_path)])

    assert result.exit_code != 0
    assert str(transcript_path) in result.output


def test_adapt_cli_reports_claude_agent_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every other LLM stage's CLI command catches `ClaudeAgentError`
    alongside its own domain error - `adapt` must too, since its fallback
    path makes the same kind of agent call."""

    async def raising_adapt_transcript(path: Path, *, agent: object) -> RawTranscript:
        raise ClaudeAgentError("agent call failed")

    monkeypatch.setattr(transcript_cli, "adapt_transcript", raising_adapt_transcript)
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text("no structure here at all\n")

    result = CliRunner().invoke(main, ["transcript", "adapt", str(transcript_path)])

    assert result.exit_code != 0
    assert "agent call failed" in result.output


def test_transcript_adapt_help_exits_zero() -> None:
    result = CliRunner().invoke(main, ["transcript", "adapt", "--help"])

    assert result.exit_code == 0
    assert "adapt" in result.output.lower()
