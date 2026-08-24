"""Behaviour of the composed pipeline (`transcription/pipeline.py`) and the
`jake-tools transcribe` porcelain.

Per the CLI-options memo (rule 4), the primary test surface is the library
seam: `run_pipeline` is driven directly with fakes for every dependency
(`FakeTranscriber`, `FakeAudioTool`, `FakeVaultClient`, and `RoutingQuery` -
a fake `run_query` that dispatches a payload by the requested response
model's JSON-schema title rather than call order, because `polish_chapters`
runs its per-chapter calls concurrently under `asyncio.gather`, which makes
a strict call-order fake racy). CLI tests stay thin: flag parsing ->
delegation, monkeypatching `run_pipeline` itself, mirroring
`test_transcript_cli.py`/`test_transcription_speakers.py`.

One `@pytest.mark.slow` test (E22) drives the real pipeline, real model
included, over a small pre-diarised-transcript fixture - skipped by default
(pytest-skip-slow; run with `uv run pytest --slow -k live`).
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
import yaml
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from click.testing import CliRunner

from jake_tools.claude import AgentSpec, ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.asr import (
    DEFAULT_ASR_MODEL,
    DEFAULT_DIARISATION_MODEL,
    Transcriber,
    TranscriberError,
)
from jake_tools.transcription.audio import (
    AudioEmbedResolutionError,
    AudioTool,
    NoAudioEmbedsError,
)
from jake_tools.transcription.cache import RunCache, sha256_of
from jake_tools.transcription.integrate import IntegrationReport
from jake_tools.transcription.models import (
    RawTranscript,
    SnippetRequest,
    SourceClip,
    Utterance,
)
from jake_tools.transcription.note import NoteParseError
from jake_tools.transcription.obsidian import ObsidianCliError, VaultClient
from jake_tools.transcription.pipeline import (
    NoEntryRampError,
    PipelineFactories,
    PipelineOutcome,
    RunReport,
    run_pipeline,
)
from jake_tools.transcription.speakers import SpeakersError, SpeakersResponse

# `jake_tools.cli`'s __init__ rebinds the names `transcribe`/`transcript` to
# their Click commands, shadowing the submodules (see `test_transcript_cli.py`)
# - fetch the actual modules via importlib to monkeypatch their bindings.
transcribe_cli = importlib.import_module("jake_tools.cli.transcribe")
transcript_cli = importlib.import_module("jake_tools.cli.transcript")

_ATTENDEES = ["Ada Lovelace", "Grace Hopper"]


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


class RoutingQuery:
    """A fake `run_query` that dispatches a payload by the requested response
    model's JSON-schema `title` (its class name), not by call order.

    A composed pipeline run makes several different kinds of structured
    calls (speaker proposals, chapter boundaries, per-chapter polish/fix,
    minutes) whose relative order and count depend on the scenario (a
    degenerate-chapters retry, `asyncio.gather`-scheduled concurrent chapter
    polishing) - keying off `output_format`'s schema title, which
    `ClaudeAgent.run_structured` always sets from the prompt's
    `response_model`, sidesteps all of that.
    """

    def __init__(self, payloads: dict[str, object]) -> None:
        self._payloads = payloads
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))
        schema = options.output_format["schema"] if options.output_format else {}
        title = schema.get("title", "")
        payload = self._payloads[title]

        async def stream() -> AsyncIterator[Message]:
            yield _structured_result(payload)

        return stream()


class FakeVaultClient:
    """Fake `VaultClient`: `vault_root` points at a real (possibly empty)
    tmp_path tree (for `build_lexicon`'s glob), `resolve_embed` looks up a
    pre-registered mapping."""

    def __init__(self, root: Path, files: dict[str, Path]) -> None:
        self._root = root
        self._files = files

    def vault_root(self) -> Path:
        return self._root

    def resolve_embed(self, target: str) -> Path:
        try:
            return self._files[target]
        except KeyError:
            raise ObsidianCliError(f"no fake mapping for {target!r}") from None


class FakeAudioTool:
    """Fake `AudioTool`: records `merge` calls, writes placeholder bytes
    instead of shelling out to ffmpeg."""

    def __init__(self) -> None:
        self.merge_calls: list[tuple[list[Path], Path]] = []
        self.cut_calls: list[tuple[Path, float, float, Path]] = []

    def duration_seconds(self, path: Path) -> float:
        return 1.0

    def merge(self, clips: Sequence[Path], out: Path) -> list[SourceClip]:
        self.merge_calls.append((list(clips), out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake merged audio")
        offset = 0.0
        result: list[SourceClip] = []
        for clip in clips:
            result.append(
                SourceClip(path=str(clip), offset_seconds=offset, duration_seconds=1.0)
            )
            offset += 1.0
        return result

    def cut(self, source: Path, start: float, end: float, out: Path) -> Path:
        self.cut_calls.append((source, start, end, out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake cut audio")
        return out


class FakeTranscriber:
    """Fake `Transcriber`: records call count, replays a canned transcript
    whose `clips` deliberately does NOT match the real merge clips - proving
    the pipeline splices the real ones back in."""

    def __init__(self, result: RawTranscript) -> None:
        self._result = result
        self.calls: list[Path] = []
        self.speaker_counts: list[int | None] = []

    def transcribe(
        self, audio: Path, *, num_speakers: int | None = None
    ) -> RawTranscript:
        self.calls.append(audio)
        self.speaker_counts.append(num_speakers)
        return self._result.model_copy(
            update={
                "audio_sha256": sha256_of(audio),
                "asr_model": DEFAULT_ASR_MODEL,
                "diarisation_model": DEFAULT_DIARISATION_MODEL,
                "diarisation_device": "cpu",
                "num_speakers": 2,
            }
        )


def _unexpected_transcriber() -> Transcriber:
    raise AssertionError("a text-ramp run must never construct a Transcriber")


# --- note fixtures -------------------------------------------------------------


def _write_note(tmp_path: Path, *, name: str, embed_block: str) -> Path:
    note_path = tmp_path / name
    note_path.write_text(
        "---\n"
        'Date: "[[August 3, 2026]]"\n'
        "Attendees:\n"
        '  - "[[Ada Lovelace]]"\n'
        '  - "[[Grace Hopper]]"\n'
        "tags:\n"
        "  - note/meeting\n"
        "---\n"
        "\n"
        "## Meeting Prep\n"
        "\n"
        "- Agenda: quarterly check-in\n"
        "\n"
        f"{embed_block}\n"
    )
    return note_path


def _audio_note(tmp_path: Path) -> Path:
    return _write_note(
        tmp_path,
        name="audio-note.md",
        embed_block=(
            "![[Recording 20260803090000.m4a]]\n\n![[Recording 20260803094500.m4a]]"
        ),
    )


def _transcript_note(tmp_path: Path, *, filename: str = "transcript.txt") -> Path:
    return _write_note(tmp_path, name="text-note.md", embed_block=f"![[{filename}]]")


def _live_transcript_note(tmp_path: Path, *, filename: str) -> Path:
    """Like `_transcript_note`, but with an explicit diarisation hint.

    `parse_named_lines` labels each utterance's speaker with the literal
    name already in the source text (`"Ada Lovelace: ..."`), so the
    "cluster" `resolve()` asks about is already spelled exactly like an
    attendee - about as unambiguous as evidence gets. A real run against the
    real model still needs a nudge to answer at `high` confidence rather
    than hedge, so this spells that out as a Meeting Prep hint too, the same
    way Michael would for a real pre-diarised transcript.
    """
    note_path = tmp_path / "live-note.md"
    note_path.write_text(
        "---\n"
        'Date: "[[August 3, 2026]]"\n'
        "Attendees:\n"
        '  - "[[Ada Lovelace]]"\n'
        '  - "[[Grace Hopper]]"\n'
        "tags:\n"
        "  - note/meeting\n"
        "---\n"
        "\n"
        "## Meeting Prep\n"
        "\n"
        "- Agenda: quarterly check-in\n"
        "- Diarisation hints:\n"
        "\t- The transcript's speaker labels are already Ada Lovelace and "
        "Grace Hopper's real names, verbatim - no further identification "
        "needed.\n"
        "\n"
        f"![[{filename}]]\n"
    )
    return note_path


def _raw_transcript() -> RawTranscript:
    return RawTranscript(
        clips=[
            SourceClip(
                path="whole-file-pseudo-clip",
                offset_seconds=0.0,
                duration_seconds=999.0,
            )
        ],
        utterances=[
            Utterance(
                start=0.0,
                end=5.0,
                speaker="SPEAKER_00",
                text="Let's get started with the budget review.",
            ),
            Utterance(
                start=5.0,
                end=10.0,
                speaker="SPEAKER_01",
                text="Sounds good, I have the numbers ready.",
            ),
            Utterance(
                start=60.0, end=65.0, speaker="SPEAKER_00", text="Let's wrap up now."
            ),
            Utterance(
                start=65.0, end=70.0, speaker="SPEAKER_01", text="Thanks everyone."
            ),
        ],
        audio_sha256="deadbeef",
    )


_HIGH_CONFIDENCE_PROPOSALS = {
    "SpeakerProposals": {
        "proposals": [
            {
                "cluster": "SPEAKER_00",
                "name": "Ada Lovelace",
                "confidence": "high",
                "reasoning": "matches hint",
            },
            {
                "cluster": "SPEAKER_01",
                "name": "Grace Hopper",
                "confidence": "high",
                "reasoning": "matches hint",
            },
        ]
    }
}

_TWO_CHAPTER_BOUNDARIES = {
    "ChapterisationResponse": {
        "chapters": [
            {"title": "Opening", "start_utterance": 0},
            {"title": "Wrap-up", "start_utterance": 2},
        ]
    }
}

_DEFAULT_POLISH = {
    "PolishedChapterResponse": {
        "summary": "A short chapter summary.",
        "turns": [{"speaker": "Ada Lovelace", "text": "Hello there."}],
    },
    "ChapterFixResponse": {
        "summary": "A short, fixed chapter summary.",
        "turns": [{"speaker": "Ada Lovelace", "text": "Hello there, fixed."}],
        "issues": [],
    },
}

_DEFAULT_MINUTES = {
    "MinutesResponse": {
        "meeting_summary": "A short fictional summary of the meeting.",
        "discussion_notes": "- Talked about the budget\n\t- Came in under budget\n",
    }
}


def _factories(
    *,
    vault: VaultClient,
    audio_tool: AudioTool,
    transcriber: Transcriber | None,
    agent: ClaudeAgent,
    cache: RunCache,
) -> PipelineFactories:
    return PipelineFactories(
        vault=lambda: vault,
        audio_tool=lambda: audio_tool,
        transcriber=(lambda: transcriber)
        if transcriber is not None
        else _unexpected_transcriber,
        agent=lambda: agent,
        cache=lambda: cache,
    )


# --- happy path: audio ramp, exemplar order, clips provenance ---------------


async def test_run_pipeline_happy_path_audio_ramp_writes_all_four_sections_in_order(
    tmp_path: Path,
) -> None:
    note_path = _audio_note(tmp_path)
    clip_a = tmp_path / "clip_a.m4a"
    clip_b = tmp_path / "clip_b.m4a"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    vault = FakeVaultClient(
        tmp_path / "vault",
        {
            "Recording 20260803090000.m4a": clip_a,
            "Recording 20260803094500.m4a": clip_b,
        },
    )
    (tmp_path / "vault").mkdir()
    audio_tool = FakeAudioTool()
    transcriber = FakeTranscriber(_raw_transcript())
    query = RoutingQuery(
        {
            **_HIGH_CONFIDENCE_PROPOSALS,
            **_TWO_CHAPTER_BOUNDARIES,
            **_DEFAULT_POLISH,
            **_DEFAULT_MINUTES,
        }
    )
    agent = ClaudeAgent(run_query=query)
    cache = RunCache(tmp_path / "cache")
    factories = _factories(
        vault=vault,
        audio_tool=audio_tool,
        transcriber=transcriber,
        agent=agent,
        cache=cache,
    )

    outcome = await run_pipeline(note_path, factories, AgentSpec())

    assert outcome.status == "complete"
    assert outcome.report is not None
    assert outcome.report.chapters == 2
    assert outcome.report.integration.sections
    assert [timing.stage for timing in outcome.report.timings] == ["merge"]
    assert transcriber.speaker_counts == [2]

    # Clips provenance: the real per-source clips from `merge_note_audio`
    # replaced the transcriber's fabricated whole-file pseudo-clip.
    stored = cache.load(outcome.run_id or "", "raw_transcript", RawTranscript)
    assert stored is not None
    assert [clip.path for clip in stored.clips] == [str(clip_a), str(clip_b)]

    text = note_path.read_text()
    assert "[!summary]" in text
    assert "## Discussion Notes" in text
    assert "## Chapters" in text
    assert "## Transcript" in text
    # Exemplar order: preamble summary -> Meeting Prep -> Discussion Notes ->
    # Chapters -> Transcript.
    assert (
        text.index("[!summary]")
        < text.index("## Meeting Prep")
        < text.index("## Discussion Notes")
        < text.index("## Chapters")
        < text.index("## Transcript")
    )


# --- needs-input: CLI contract equality with `transcript speakers` ----------


def test_transcribe_needs_input_contract_matches_transcript_speakers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    request = SnippetRequest(cluster="SPEAKER_02", clip_paths=["/x.m4a"], context="hi")
    note_path = tmp_path / "note.md"
    note_path.write_text("placeholder")

    async def fake_run_pipeline(*args: object, **kwargs: object) -> PipelineOutcome:
        return PipelineOutcome(status="needs_input", run_id="run-1", requests=[request])

    async def fake_run_speaker_resolution(
        *args: object, **kwargs: object
    ) -> SpeakersResponse:
        return SpeakersResponse(
            status="needs_input", run_id="run-1", requests=[request]
        )

    monkeypatch.setattr(transcribe_cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        transcript_cli, "run_speaker_resolution", fake_run_speaker_resolution
    )
    runner = CliRunner()

    porcelain_result = runner.invoke(main, ["transcribe", str(note_path)])
    plumbing_result = runner.invoke(
        main, ["transcript", "speakers", str(note_path), "--run-id", "run-1"]
    )

    assert porcelain_result.exit_code == transcribe_cli.NEEDS_INPUT_EXIT_CODE
    assert plumbing_result.exit_code == transcript_cli.NEEDS_INPUT_EXIT_CODE
    assert porcelain_result.exit_code not in (0, 1)
    assert json.loads(porcelain_result.output) == json.loads(plumbing_result.output)


# --- resume: fake transcriber called once across two invocations ------------


async def test_run_pipeline_resume_after_assign_does_not_repeat_asr(
    tmp_path: Path,
) -> None:
    note_path = _audio_note(tmp_path)
    clip_a = tmp_path / "clip_a.m4a"
    clip_b = tmp_path / "clip_b.m4a"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    (tmp_path / "vault").mkdir()
    vault = FakeVaultClient(
        tmp_path / "vault",
        {
            "Recording 20260803090000.m4a": clip_a,
            "Recording 20260803094500.m4a": clip_b,
        },
    )
    audio_tool = FakeAudioTool()
    transcriber = FakeTranscriber(_raw_transcript())
    # SPEAKER_00 confidently resolves; SPEAKER_01 stays unresolved until
    # --assign fills it in on the second call.
    query = RoutingQuery(
        {
            "SpeakerProposals": {
                "proposals": [
                    {
                        "cluster": "SPEAKER_00",
                        "name": "Ada Lovelace",
                        "confidence": "high",
                        "reasoning": "clear",
                    },
                    {
                        "cluster": "SPEAKER_01",
                        "name": None,
                        "confidence": "low",
                        "reasoning": "unsure",
                    },
                ]
            },
            **_TWO_CHAPTER_BOUNDARIES,
            **_DEFAULT_POLISH,
            **_DEFAULT_MINUTES,
        }
    )
    agent = ClaudeAgent(run_query=query)
    cache = RunCache(tmp_path / "cache")
    factories = _factories(
        vault=vault,
        audio_tool=audio_tool,
        transcriber=transcriber,
        agent=agent,
        cache=cache,
    )

    first = await run_pipeline(note_path, factories, AgentSpec())
    assert first.status == "needs_input"
    assert len(transcriber.calls) == 1

    second = await run_pipeline(
        note_path, factories, AgentSpec(), assignments=("SPEAKER_01=Grace Hopper",)
    )
    assert second.status == "complete"
    assert len(transcriber.calls) == 1  # still one: ASR was never repeated
    assert first.run_id == second.run_id


# --- ramp selection: audio vs transcript vs neither -------------------------


async def test_run_pipeline_transcript_ramp_never_constructs_a_transcriber(
    tmp_path: Path,
) -> None:
    note_path = _transcript_note(tmp_path)
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        "Ada Lovelace: Let's start with the budget review.\n"
        "Grace Hopper: Sounds good, I have the numbers ready.\n"
        "Ada Lovelace: Great, let's begin.\n"
        "Grace Hopper: The total comes to twelve thousand.\n"
    )
    (tmp_path / "vault").mkdir()
    vault = FakeVaultClient(tmp_path / "vault", {"transcript.txt": transcript_path})
    query = RoutingQuery(
        {
            "SpeakerProposals": {
                "proposals": [
                    {
                        "cluster": "Ada Lovelace",
                        "name": "Ada Lovelace",
                        "confidence": "high",
                        "reasoning": "self-identifying label",
                    },
                    {
                        "cluster": "Grace Hopper",
                        "name": "Grace Hopper",
                        "confidence": "high",
                        "reasoning": "self-identifying label",
                    },
                ]
            },
            **_TWO_CHAPTER_BOUNDARIES,
            **_DEFAULT_POLISH,
            **_DEFAULT_MINUTES,
        }
    )
    agent = ClaudeAgent(run_query=query)
    cache = RunCache(tmp_path / "cache")
    factories = _factories(
        vault=vault,
        audio_tool=FakeAudioTool(),
        transcriber=None,
        agent=agent,
        cache=cache,
    )

    outcome = await run_pipeline(note_path, factories, AgentSpec())

    assert outcome.status == "complete"
    stored = cache.load(outcome.run_id or "", "raw_transcript", RawTranscript)
    assert stored is not None
    assert stored.audio_sha256 is None  # a text source, never a hashed audio one
    assert stored.clips == []


async def test_run_pipeline_raises_when_note_has_no_recognisable_embed(
    tmp_path: Path,
) -> None:
    note_path = _write_note(
        tmp_path, name="empty-note.md", embed_block="No embed here."
    )
    (tmp_path / "vault").mkdir()
    vault = FakeVaultClient(tmp_path / "vault", {})
    factories = _factories(
        vault=vault,
        audio_tool=FakeAudioTool(),
        transcriber=None,
        agent=ClaudeAgent(run_query=RoutingQuery({})),
        cache=RunCache(tmp_path / "cache"),
    )

    with pytest.raises(NoEntryRampError):
        await run_pipeline(note_path, factories, AgentSpec())


async def test_run_pipeline_prefers_the_audio_ramp_when_both_embeds_present(
    tmp_path: Path,
) -> None:
    note_path = _write_note(
        tmp_path,
        name="both-note.md",
        embed_block="![[Recording 20260803090000.m4a]]\n\n![[transcript.txt]]",
    )
    clip_a = tmp_path / "clip_a.m4a"
    clip_a.write_bytes(b"a")
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text("Ada Lovelace: hi\nGrace Hopper: hi\n")
    (tmp_path / "vault").mkdir()
    vault = FakeVaultClient(
        tmp_path / "vault",
        {"Recording 20260803090000.m4a": clip_a, "transcript.txt": transcript_path},
    )
    audio_tool = FakeAudioTool()
    transcriber = FakeTranscriber(_raw_transcript())
    query = RoutingQuery(
        {
            **_HIGH_CONFIDENCE_PROPOSALS,
            **_TWO_CHAPTER_BOUNDARIES,
            **_DEFAULT_POLISH,
            **_DEFAULT_MINUTES,
        }
    )
    agent = ClaudeAgent(run_query=query)
    cache = RunCache(tmp_path / "cache")
    factories = _factories(
        vault=vault,
        audio_tool=audio_tool,
        transcriber=transcriber,
        agent=agent,
        cache=cache,
    )

    outcome = await run_pipeline(note_path, factories, AgentSpec())

    assert outcome.status == "complete"
    assert len(transcriber.calls) == 1
    stored = cache.load(outcome.run_id or "", "raw_transcript", RawTranscript)
    assert stored is not None
    assert stored.audio_sha256 is not None  # the audio ramp ran, not the text one


# --- unknown_turn_ratio, from a scripted resolution --------------------------


async def test_run_pipeline_unknown_turn_ratio_reflects_the_polished_turns(
    tmp_path: Path,
) -> None:
    note_path = _audio_note(tmp_path)
    clip_a = tmp_path / "clip_a.m4a"
    clip_b = tmp_path / "clip_b.m4a"
    clip_a.write_bytes(b"a")
    clip_b.write_bytes(b"b")
    (tmp_path / "vault").mkdir()
    vault = FakeVaultClient(
        tmp_path / "vault",
        {
            "Recording 20260803090000.m4a": clip_a,
            "Recording 20260803094500.m4a": clip_b,
        },
    )
    audio_tool = FakeAudioTool()
    transcriber = FakeTranscriber(_raw_transcript())
    # Two chapters, three polished turns each (from the shared canned
    # response below): one "Unknown" turn and two named turns per chapter -
    # an unambiguous, hand-computed ratio of 2/6.
    query = RoutingQuery(
        {
            **_HIGH_CONFIDENCE_PROPOSALS,
            **_TWO_CHAPTER_BOUNDARIES,
            "PolishedChapterResponse": {
                "summary": "A short chapter summary.",
                "turns": [{"speaker": "Ada Lovelace", "text": "placeholder"}],
            },
            "ChapterFixResponse": {
                "summary": "A short, fixed chapter summary.",
                "turns": [
                    {"speaker": "Ada Lovelace", "text": "The budget looks fine."},
                    {"speaker": "Unknown", "text": "Something unclear was said."},
                    {"speaker": "Grace Hopper", "text": "Agreed, let's continue."},
                ],
                "issues": [],
            },
            **_DEFAULT_MINUTES,
        }
    )
    agent = ClaudeAgent(run_query=query)
    cache = RunCache(tmp_path / "cache")
    factories = _factories(
        vault=vault,
        audio_tool=audio_tool,
        transcriber=transcriber,
        agent=agent,
        cache=cache,
    )

    outcome = await run_pipeline(
        note_path,
        factories,
        AgentSpec(),
        assignments=("SPEAKER_01=Unknown",),
        finalise=True,
    )

    assert outcome.status == "complete"
    assert outcome.report is not None
    assert outcome.report.unknown_turn_ratio == pytest.approx(2 / 6)


# --- CLI layer: thin flag-parsing/delegation tests ---------------------------


def test_transcribe_cli_delegates_to_run_pipeline_and_prints_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("placeholder")
    fake_report = RunReport(
        run_id="run-1",
        chapters=2,
        unknown_turn_ratio=0.25,
        fixer_issues=["Opening: fixed a typo"],
        integration=IntegrationReport(
            run_id="run-1", note_path=str(note_path), sections=[]
        ),
    )
    captured: dict[str, object] = {}

    async def fake_run_pipeline(
        note_path_arg: Path,
        factories: object,
        spec: object,
        *,
        assignments: tuple[str, ...],
        finalise: bool,
        max_concurrency: int,
    ) -> PipelineOutcome:
        captured["note_path"] = note_path_arg
        captured["assignments"] = assignments
        captured["finalise"] = finalise
        captured["max_concurrency"] = max_concurrency
        return PipelineOutcome(status="complete", run_id="run-1", report=fake_report)

    monkeypatch.setattr(transcribe_cli, "run_pipeline", fake_run_pipeline)

    result = CliRunner().invoke(
        main,
        [
            "transcribe",
            str(note_path),
            "--assign",
            "SPEAKER_00=Ada Lovelace",
            "--finalise",
            "--max-concurrency",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(fake_report.model_dump_json())
    assert captured["note_path"] == note_path
    assert captured["assignments"] == ("SPEAKER_00=Ada Lovelace",)
    assert captured["finalise"] is True
    assert captured["max_concurrency"] == 2


# One representative instance per error family `_STAGE_ERRORS` maps to a
# clean `ClickException` - including the two `audio.py` raises directly as
# `RuntimeError` (never through `AudioToolError`), the specific gap a
# moved/renamed audio file exercises in production (a previous version of
# this test only covered `NoEntryRampError`, which stayed green even after
# `AudioEmbedResolutionError`/`NoAudioEmbedsError` were dropped from
# `_STAGE_ERRORS` - a raw, unwrapped traceback would not have failed it).
_ERROR_CASES: list[tuple[str, Exception]] = [
    ("pipeline (no entry ramp)", NoEntryRampError(Path("note.md"))),
    ("audio embed resolution failure", AudioEmbedResolutionError("no such clip")),
    ("no audio embeds at all", NoAudioEmbedsError("nothing to merge")),
    ("speaker resolution", SpeakersError("no cached raw transcript")),
    ("transcriber", TranscriberError("HF_TOKEN missing")),
    ("obsidian CLI", ObsidianCliError("vault not found")),
    ("claude agent", ClaudeAgentError("agent call failed")),
    (
        "note parse (broken frontmatter YAML)",
        NoteParseError(Path("note.md"), yaml.YAMLError("bad yaml")),
    ),
]


@pytest.mark.parametrize("case", _ERROR_CASES, ids=[label for label, _ in _ERROR_CASES])
def test_transcribe_cli_reports_each_stage_error_family_as_a_clean_click_exception(
    case: tuple[str, Exception], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _label, error = case
    note_path = tmp_path / "note.md"
    note_path.write_text("placeholder")

    async def raising_run_pipeline(*args: object, **kwargs: object) -> PipelineOutcome:
        raise error

    monkeypatch.setattr(transcribe_cli, "run_pipeline", raising_run_pipeline)

    result = CliRunner().invoke(main, ["transcribe", str(note_path)])

    assert result.exit_code == 1
    assert str(error) in result.output


def test_transcribe_help_exits_zero_and_documents_the_needs_input_contract() -> None:
    result = CliRunner().invoke(main, ["transcribe", "--help"])

    assert result.exit_code == 0
    assert "needs_input" in result.output
    # Whitespace-normalised so Click's own paragraph rewrapping (which can
    # split "exits with code 3" across a line break) doesn't defeat an
    # anchored substring check - a bare "3" would also match "1", "--max-
    # concurrency 4", etc. and prove nothing about the documented exit code.
    normalised = " ".join(result.output.split())
    assert "exits with code 3" in normalised


# --- Step 3a: real-LLM end-to-end slow test (E22) ---------------------------

_RAW_MARKERS = ("budget", "spot instance", "embedded", "firmware")

_TRANSCRIPT_LINES = "\n".join(
    [
        "Ada Lovelace: Let's start with the budget review for this quarter.",
        "Grace Hopper: Sure, we came in about ten percent under budget on the "
        "compute cluster.",
        "Ada Lovelace: That's good news. Where did the savings come from?",
        "Grace Hopper: Mostly from switching two of the training jobs to the "
        "cheaper spot instance pool.",
        "Ada Lovelace: Good, let's keep an eye on spot availability though, we "
        "don't want a job to get evicted mid run.",
        "Grace Hopper: Agreed, I'll add an alert for that.",
        "Ada Lovelace: Let's move on to hiring. Where are we with the new "
        "firmware engineer role?",
        "Grace Hopper: We have three candidates through the first round. Two "
        "look strong on embedded C, one is more of a generalist.",
        "Ada Lovelace: Let's prioritise the two embedded candidates for the "
        "next round.",
        "Grace Hopper: Will do, I'll schedule those for next week.",
        "Ada Lovelace: Anything else on hiring?",
        "Grace Hopper: Not from me, that's everything.",
    ]
)


def test_the_fixtures_own_raw_marker_check_is_honest() -> None:
    """Guards the slow test's premise: `_RAW_MARKERS` really do come from the
    raw dialogue fed to the model, so the slow test's assertion that they
    survive into the polished transcript checks genuine derivation, not
    something the fixture merely repeats by coincidence."""
    lowered = _TRANSCRIPT_LINES.casefold()
    for marker in _RAW_MARKERS:
        assert marker in lowered


@pytest.mark.slow
async def test_live_transcribe_produces_all_four_sections_with_attendee_only_speakers(
    tmp_path: Path,
) -> None:
    note_path = _live_transcript_note(tmp_path, filename="live-transcript.txt")
    transcript_path = tmp_path / "live-transcript.txt"
    transcript_path.write_text(_TRANSCRIPT_LINES + "\n")
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    vault = FakeVaultClient(vault_root, {"live-transcript.txt": transcript_path})
    # Only exercised if the real model stays unresolved on a cluster despite
    # the hint above - `--finalise` then maps it to "Unknown" rather than
    # stopping for input, so this slow test doesn't hinge on the model
    # reaching `high` confidence every run.
    audio_tool = FakeAudioTool()
    agent = ClaudeAgent(defaults=AgentSpec(effort="low"))
    cache = RunCache(tmp_path / "cache")
    factories = _factories(
        vault=vault, audio_tool=audio_tool, transcriber=None, agent=agent, cache=cache
    )

    outcome = await run_pipeline(
        note_path, factories, AgentSpec(effort="low"), finalise=True
    )

    assert outcome.status == "complete", outcome.model_dump()
    assert outcome.report is not None
    print("run report:", outcome.report.model_dump_json(indent=2))

    text = note_path.read_text()
    assert (
        text.index("[!summary]")
        < text.index("## Meeting Prep")
        < text.index("## Discussion Notes")
        < text.index("## Chapters")
        < text.index("## Transcript")
    )

    # All four product sections are populated, not just present.
    summary_start = text.index("[!summary]")
    meeting_prep_start = text.index("## Meeting Prep")
    summary_body = text[summary_start:meeting_prep_start]
    summary_content_lines = [
        line.lstrip(">").strip()
        for line in summary_body.splitlines()
        if "[!summary]" not in line
    ]
    assert any(line for line in summary_content_lines)

    discussion_notes_start = text.index("## Discussion Notes")
    chapters_start = text.index("## Chapters")
    discussion_notes_body = text[discussion_notes_start:chapters_start]
    assert any(
        line.lstrip().startswith(("-", "*"))
        for line in discussion_notes_body.splitlines()
    )

    transcript_start = text.index("## Transcript")
    chapters_body = text[chapters_start:transcript_start]
    assert "- [[#" in chapters_body

    transcript_body = text[transcript_start:]
    assert transcript_body.strip() != ""

    # Speakers confined to the attendee set (or "Unknown").
    speakers = set(re.findall(r"\*\*([^:*]+):\*\*", transcript_body))
    assert speakers, f"no speaker turns found in transcript: {transcript_body!r}"
    assert speakers <= {*_ATTENDEES, "Unknown"}, speakers

    # Wording recognisably derived from the input, not fabricated.
    lowered_transcript = transcript_body.casefold()
    found = [marker for marker in _RAW_MARKERS if marker in lowered_transcript]
    assert len(found) >= 2, (
        f"expected at least 2 of {_RAW_MARKERS} in the polished transcript, "
        f"found {found}: {transcript_body!r}"
    )

    print("note text:\n", text)
