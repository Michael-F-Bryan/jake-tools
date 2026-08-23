"""Behaviour of speaker resolution (`transcription/speakers.py`), the
sanctioned Meeting Prep append helper (`transcription/note.py`), and the
`jake-tools transcript speakers` CLI command.

Per the CLI-options memo (rule 4), the primary test surface is the library
seam: `resolve` is tested directly with a fake `ClaudeAgent` (pattern:
`RecordingQuery`, `tests/test_claude_agent.py:58-79`) and a fake `AudioTool`
(pattern: `FakeAudioTool`, `tests/test_transcription_audio.py:93-118`), and
`run_speaker_resolution` is tested with a real `RunCache` pointed at
`tmp_path`. CLI tests stay thin: flag parsing -> delegation, monkeypatching
`run_speaker_resolution`, mirroring `test_transcript_cli.py`.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from click.testing import CliRunner

from jake_tools.claude import ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.models import (
    RawTranscript,
    SnippetRequest,
    SourceClip,
    SpeakerAssignment,
    Utterance,
)
from jake_tools.transcription.note import (
    NoteSection,
    ParsedNote,
    append_diarisation_hints,
    parse_note,
    render_body,
)
from jake_tools.transcription.speakers import (
    AssignmentSet,
    MissingRawTranscriptError,
    SpeakersError,
    SpeakersResponse,
    resolve,
    run_speaker_resolution,
)

FIXTURES = Path(__file__).parent / "fixtures"

# `jake_tools.cli`'s __init__ rebinds the name `transcript` to the Click
# group, shadowing the submodule (see `test_transcript_cli.py`) — fetch the
# actual module via importlib to monkeypatch its `run_speaker_resolution`.
transcript_cli = importlib.import_module("jake_tools.cli.transcript")


# --- fakes -------------------------------------------------------------------


class RecordingQuery:
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


def _fake_agent(
    proposals: list[dict[str, object]],
) -> tuple[ClaudeAgent, RecordingQuery]:
    fake = RecordingQuery(_structured_result({"proposals": proposals}))
    return ClaudeAgent(run_query=fake), fake


class FakeAudioTool:
    """Fake AudioTool that records `cut` calls instead of shelling out."""

    def __init__(self) -> None:
        self.cut_calls: list[tuple[Path, float, float, Path]] = []

    def duration_seconds(self, path: Path) -> float:  # pragma: no cover - unused here
        return 1.0

    def merge(
        self, clips: Sequence[Path], out: Path
    ) -> list[SourceClip]:  # pragma: no cover - unused here
        raise NotImplementedError

    def cut(self, source: Path, start: float, end: float, out: Path) -> Path:
        self.cut_calls.append((source, start, end, out))
        out.write_bytes(b"fake cut audio")
        return out


def _note(
    *,
    attendees: list[str],
    diarisation_hints: list[str] | None = None,
    meeting_prep_body: str = "",
) -> ParsedNote:
    return ParsedNote(
        path="fake-note.md",
        frontmatter={"Date": "[[August 3, 2026]]"},
        context="meeting",
        attendees=attendees,
        diarisation_hints=diarisation_hints or [],
        embeds=[],
        sections=[
            NoteSection(heading=None, level=0, body=""),
            NoteSection(heading="Meeting Prep", level=2, body=meeting_prep_body),
        ],
    )


# --- resolve(): explicit assignments beat the LLM ----------------------------


async def test_assignments_win_over_contradictory_llm_proposals(tmp_path: Path) -> None:
    note = _note(attendees=["Ada Lovelace", "Grace Hopper"])
    transcript = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="Good morning."),
            Utterance(
                start=1.0, end=2.0, speaker="SPEAKER_01", text="Morning to you too."
            ),
        ],
        audio_sha256=None,
    )
    # The LLM contradicts the explicit assignment for SPEAKER_00, and
    # confidently resolves SPEAKER_01 — only the latter should win.
    agent, fake = _fake_agent(
        [
            {
                "cluster": "SPEAKER_00",
                "name": "Grace Hopper",
                "confidence": "high",
                "reasoning": "should never be consulted",
            },
            {
                "cluster": "SPEAKER_01",
                "name": "Grace Hopper",
                "confidence": "high",
                "reasoning": "clear",
            },
        ]
    )

    resolved, requests = await resolve(
        note,
        transcript,
        agent=agent,
        assignments=[SpeakerAssignment(cluster="SPEAKER_00", name="Ada Lovelace")],
        audio_tool=FakeAudioTool(),
        merged_audio=tmp_path / "merged.m4a",
        snippet_dir=tmp_path / "snippets",
    )

    assert [u.speaker for u in resolved.utterances] == ["Ada Lovelace", "Grace Hopper"]
    assert requests == []
    # The assigned cluster is never even put in front of the LLM.
    prompt_text = fake.calls[0][0]
    assert '"cluster": "SPEAKER_00"' not in prompt_text
    assert '"cluster": "SPEAKER_01"' in prompt_text


# --- resolve(): low confidence never guesses, and cuts snippets -------------


async def test_low_confidence_proposal_becomes_a_snippet_request_with_cut_clips(
    tmp_path: Path,
) -> None:
    note = _note(attendees=["Ada Lovelace", "Grace Hopper"])
    transcript = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0.0, end=1.0, speaker="SPEAKER_02", text="Hello there."),
            Utterance(
                start=40.0, end=42.0, speaker="SPEAKER_02", text="Anyway, moving on."
            ),
        ],
        audio_sha256=None,
    )
    agent, _fake = _fake_agent(
        [
            {
                "cluster": "SPEAKER_02",
                "name": None,
                "confidence": "low",
                "reasoning": "not enough to go on",
            }
        ]
    )
    audio_tool = FakeAudioTool()
    merged_audio = tmp_path / "merged.m4a"
    snippet_dir = tmp_path / "run" / "snippets"

    resolved, requests = await resolve(
        note,
        transcript,
        agent=agent,
        assignments=[],
        audio_tool=audio_tool,
        merged_audio=merged_audio,
        snippet_dir=snippet_dir,
    )

    # Never guesses: the cluster id survives untouched, not "Unknown" either.
    assert [u.speaker for u in resolved.utterances] == ["SPEAKER_02", "SPEAKER_02"]
    assert len(requests) == 1
    request = requests[0]
    assert request.cluster == "SPEAKER_02"
    assert request.clip_paths == [
        str(snippet_dir / "SPEAKER_02-1.m4a"),
        str(snippet_dir / "SPEAKER_02-2.m4a"),
    ]
    assert [call[3] for call in audio_tool.cut_calls] == [
        snippet_dir / "SPEAKER_02-1.m4a",
        snippet_dir / "SPEAKER_02-2.m4a",
    ]
    assert all(call[0] == merged_audio for call in audio_tool.cut_calls)
    for path in request.clip_paths:
        assert Path(path).exists()


async def test_no_proposal_at_all_for_a_cluster_also_becomes_a_snippet_request(
    tmp_path: Path,
) -> None:
    """The LLM must answer once per cluster; if it silently drops one, that's
    still an unresolved cluster, never a guess."""
    note = _note(attendees=["Ada Lovelace"])
    transcript = RawTranscript(
        clips=[],
        utterances=[Utterance(start=0.0, end=1.0, speaker="SPEAKER_05", text="Hi.")],
        audio_sha256=None,
    )
    agent, _fake = _fake_agent([])  # no proposals at all

    resolved, requests = await resolve(
        note,
        transcript,
        agent=agent,
        assignments=[],
        audio_tool=FakeAudioTool(),
        merged_audio=tmp_path / "merged.m4a",
        snippet_dir=tmp_path / "snippets",
    )

    assert resolved.utterances[0].speaker == "SPEAKER_05"
    assert [r.cluster for r in requests] == ["SPEAKER_05"]


# --- resolve(): silent attendees are never force-assigned --------------------


async def test_silent_attendee_never_force_assigned(tmp_path: Path) -> None:
    note = _note(attendees=["Ada Lovelace", "Grace Hopper", "Charles Babbage"])
    transcript = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="Hi Grace."),
        ],
        audio_sha256=None,
    )
    agent, _fake = _fake_agent(
        [
            {
                "cluster": "SPEAKER_00",
                "name": "Ada Lovelace",
                "confidence": "high",
                "reasoning": "clear",
            }
        ]
    )

    resolved, requests = await resolve(
        note,
        transcript,
        agent=agent,
        assignments=[],
        audio_tool=FakeAudioTool(),
        merged_audio=tmp_path / "merged.m4a",
        snippet_dir=tmp_path / "snippets",
    )

    speakers = {u.speaker for u in resolved.utterances}
    assert "Charles Babbage" not in speakers
    assert requests == []


async def test_medium_confidence_requires_hint_corroboration(tmp_path: Path) -> None:
    note = _note(
        attendees=["Ada Lovelace", "Grace Hopper"],
        diarisation_hints=["Grace did most of the talking"],
    )
    transcript = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="Let's begin."),
            Utterance(start=1.0, end=2.0, speaker="SPEAKER_01", text="Sounds good."),
        ],
        audio_sha256=None,
    )
    agent, _fake = _fake_agent(
        [
            {
                "cluster": "SPEAKER_00",
                "name": "Grace Hopper",
                "confidence": "medium",
                "reasoning": "corroborated by hint",
            },
            {
                "cluster": "SPEAKER_01",
                "name": "Ada Lovelace",
                "confidence": "medium",
                "reasoning": "uncorroborated guess",
            },
        ]
    )

    resolved, requests = await resolve(
        note,
        transcript,
        agent=agent,
        assignments=[],
        audio_tool=FakeAudioTool(),
        merged_audio=tmp_path / "merged.m4a",
        snippet_dir=tmp_path / "snippets",
    )

    assert resolved.utterances[0].speaker == "Grace Hopper"  # corroborated by a hint
    assert resolved.utterances[1].speaker == "SPEAKER_01"  # uncorroborated: unresolved
    assert [r.cluster for r in requests] == ["SPEAKER_01"]


async def test_medium_confidence_corroboration_rejects_unrelated_substring_matches(
    tmp_path: Path,
) -> None:
    """Reviewer repro: a short first name that happens to be a substring of
    an unrelated word ("ada" in "Canada", "ed" in "needed") must NOT count
    as corroboration - that's the exact misattribution-poisons-the-record
    scenario the never-guess rule exists to prevent."""
    note = _note(
        attendees=["Ada Lovelace", "Ed Chen"],
        diarisation_hints=["The vendor is based in Canada", "We needed more time"],
    )
    transcript = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="Let's begin."),
            Utterance(start=1.0, end=2.0, speaker="SPEAKER_01", text="Sounds good."),
        ],
        audio_sha256=None,
    )
    agent, _fake = _fake_agent(
        [
            {
                "cluster": "SPEAKER_00",
                "name": "Ada Lovelace",
                "confidence": "medium",
                "reasoning": "would spuriously match 'Canada'",
            },
            {
                "cluster": "SPEAKER_01",
                "name": "Ed Chen",
                "confidence": "medium",
                "reasoning": "would spuriously match 'needed'",
            },
        ]
    )

    resolved, requests = await resolve(
        note,
        transcript,
        agent=agent,
        assignments=[],
        audio_tool=FakeAudioTool(),
        merged_audio=tmp_path / "merged.m4a",
        snippet_dir=tmp_path / "snippets",
    )

    assert resolved.utterances[0].speaker == "SPEAKER_00"
    assert resolved.utterances[1].speaker == "SPEAKER_01"
    assert {r.cluster for r in requests} == {"SPEAKER_00", "SPEAKER_01"}


async def test_medium_confidence_corroboration_accepts_a_genuine_word_boundary_match(
    tmp_path: Path,
) -> None:
    note = _note(attendees=["Ed Chen"], diarisation_hints=["Ed was very quiet today"])
    transcript = RawTranscript(
        clips=[],
        utterances=[Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="Hi.")],
        audio_sha256=None,
    )
    agent, _fake = _fake_agent(
        [
            {
                "cluster": "SPEAKER_00",
                "name": "Ed Chen",
                "confidence": "medium",
                "reasoning": "hint names Ed directly",
            }
        ]
    )

    resolved, requests = await resolve(
        note,
        transcript,
        agent=agent,
        assignments=[],
        audio_tool=FakeAudioTool(),
        merged_audio=tmp_path / "merged.m4a",
        snippet_dir=tmp_path / "snippets",
    )

    assert resolved.utterances[0].speaker == "Ed Chen"
    assert requests == []


# --- note.append_diarisation_hints: the sanctioned Meeting Prep write -------


def _read(path: Path) -> str:
    return path.read_text()


def test_append_diarisation_hints_adds_a_bullet_under_existing_hints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.md"
    path.write_text((FIXTURES / "meeting_note.md").read_text())
    original = _read(path)

    changed = append_diarisation_hints(
        path, ["Nikki Staltari was SPEAKER_03 in the 2026-08-11 recording"]
    )

    assert changed is True
    text = _read(path)
    assert "Nikki Staltari was SPEAKER_03 in the 2026-08-11 recording" in text
    assert "Ada mostly asked questions" in text  # pre-existing hint untouched
    assert "Grace did most of the talking" in text
    # Everything before and after Meeting Prep is untouched.
    assert (
        text.split("## Discussion Notes")[1] == original.split("## Discussion Notes")[1]
    )
    assert text.split("---\n", 2)[0] == original.split("---\n", 2)[0]


def test_append_diarisation_hints_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text((FIXTURES / "meeting_note.md").read_text())
    line = "Nikki Staltari was SPEAKER_03 in the 2026-08-11 recording"

    first = append_diarisation_hints(path, [line])
    after_first = _read(path)
    second = append_diarisation_hints(path, [line])
    after_second = _read(path)

    assert first is True
    assert second is False
    assert after_first == after_second


def test_append_diarisation_hints_creates_the_bullet_when_absent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.md"
    path.write_text(
        "---\n"
        'Attendees:\n  - "[[Ada Lovelace]]"\n'
        "tags:\n  - note/meeting\n"
        "---\n\n"
        "## Meeting Prep\n\n"
        "- Agenda link: <https://example.test/agenda>\n\n"
        "## Discussion Notes\n\n"
        "- placeholder\n"
    )

    changed = append_diarisation_hints(
        path, ["Ada Lovelace was SPEAKER_00 in the 2026-08-11 recording"]
    )

    assert changed is True
    text = _read(path)
    assert "- Diarisation hints:\n" in text
    assert "\t- Ada Lovelace was SPEAKER_00 in the 2026-08-11 recording\n" in text
    # Reconstruction property: nothing outside the Meeting Prep body changed.
    note = parse_note(path)
    discussion = next(s for s in note.sections if s.heading == "Discussion Notes")
    assert discussion.body == "\n- placeholder\n"


def test_append_diarisation_hints_is_a_no_op_without_a_meeting_prep_section(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.md"
    path.write_text((FIXTURES / "ops_log.md").read_text())
    original = _read(path)

    changed = append_diarisation_hints(path, ["Anyone was SPEAKER_00 in the recording"])

    assert changed is False
    assert _read(path) == original


def test_append_diarisation_hints_reconstructs_byte_identical_sections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.md"
    path.write_text((FIXTURES / "meeting_note.md").read_text())

    append_diarisation_hints(path, ["Nikki Staltari was SPEAKER_03 in this recording"])

    before = parse_note(FIXTURES / "meeting_note.md")
    after = parse_note(path)
    before_by_heading = {s.heading: s.body for s in before.sections}
    after_by_heading = {s.heading: s.body for s in after.sections}
    for heading in (
        "Chapters",
        "Discussion Notes",
        "00:00 — Opening",
        "05:00 — Wrap-up",
    ):
        assert after_by_heading[heading] == before_by_heading[heading]
    assert after.frontmatter == before.frontmatter
    # render_body's reconstruction property still holds for the mutated note.
    assert render_body(after.sections) == path.read_text().split("---\n", 2)[2]


# --- run_speaker_resolution: cache orchestration -----------------------------


def _cache_note(tmp_path: Path) -> Path:
    path = tmp_path / "note.md"
    path.write_text((FIXTURES / "meeting_note.md").read_text())
    return path


async def test_run_speaker_resolution_raises_without_a_cached_raw_transcript(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    agent, _fake = _fake_agent([])

    with pytest.raises(MissingRawTranscriptError):
        await run_speaker_resolution(
            _cache_note(tmp_path),
            "run-1",
            assign=(),
            finalise=False,
            agent=agent,
            audio_tool=FakeAudioTool(),
            cache=cache,
        )


async def test_run_speaker_resolution_needs_input_when_unresolved_and_not_finalised(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.store(
        "run-1",
        "raw_transcript",
        RawTranscript(
            clips=[],
            utterances=[
                Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="hi"),
            ],
            audio_sha256=None,
        ),
    )
    agent, _fake = _fake_agent(
        [{"cluster": "SPEAKER_00", "name": None, "confidence": "low", "reasoning": "?"}]
    )

    result = await run_speaker_resolution(
        _cache_note(tmp_path),
        "run-1",
        assign=(),
        finalise=False,
        agent=agent,
        audio_tool=FakeAudioTool(),
        cache=cache,
    )

    assert result.status == "needs_input"
    assert result.run_id == "run-1"
    assert len(result.requests) == 1
    assert result.requests[0].cluster == "SPEAKER_00"
    assert cache.load("run-1", "resolved_transcript", RawTranscript) is None


async def test_run_speaker_resolution_assign_unknown_and_finalise_resolves(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.store(
        "run-1",
        "raw_transcript",
        RawTranscript(
            clips=[],
            utterances=[
                Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="hi"),
                Utterance(start=1.0, end=2.0, speaker="SPEAKER_01", text="there"),
            ],
            audio_sha256=None,
        ),
    )
    # SPEAKER_00 gets a real name via LLM; SPEAKER_01 is deliberately given up
    # on via `--assign SPEAKER_01=Unknown` before `--finalise` mops up.
    agent, _fake = _fake_agent(
        [
            {
                "cluster": "SPEAKER_00",
                "name": "Ada Lovelace",
                "confidence": "high",
                "reasoning": "clear",
            }
        ]
    )
    note_path = _cache_note(tmp_path)

    result = await run_speaker_resolution(
        note_path,
        "run-1",
        assign=("SPEAKER_01=Unknown",),
        finalise=True,
        agent=agent,
        audio_tool=FakeAudioTool(),
        cache=cache,
    )

    assert result.status == "resolved"
    assert result.requests == []
    resolved = cache.load("run-1", "resolved_transcript", RawTranscript)
    assert resolved is not None
    assert [u.speaker for u in resolved.utterances] == ["Ada Lovelace", "Unknown"]
    stored_assignments = cache.load("run-1", "assignments", AssignmentSet)
    assert stored_assignments is not None
    assert SpeakerAssignment(cluster="SPEAKER_01", name="Unknown") in (
        stored_assignments.assignments
    )
    # The Meeting Prep write is scoped to human-confirmed (--assign)
    # mappings only, per the E15 ruling: SPEAKER_00 was resolved by the LLM
    # alone (never --assign'd), so it must NOT be written, even though it's
    # in resolved_transcript.json. "Unknown" is never written either.
    note_text = note_path.read_text()
    assert "Ada Lovelace was SPEAKER_00" not in note_text
    assert "Unknown was SPEAKER_01" not in note_text


async def test_run_speaker_resolution_writes_hints_only_for_assigned_clusters(
    tmp_path: Path,
) -> None:
    """The other half of the E15 scoping: an --assign'd mapping IS written
    to Meeting Prep, in the same run where an LLM-only mapping is not."""
    cache = RunCache(tmp_path / "cache")
    cache.store(
        "run-1",
        "raw_transcript",
        RawTranscript(
            clips=[],
            utterances=[
                Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="hi"),
                Utterance(start=1.0, end=2.0, speaker="SPEAKER_01", text="there"),
            ],
            audio_sha256=None,
        ),
    )
    # SPEAKER_00 is confirmed by Michael via --assign; SPEAKER_01 is resolved
    # by the LLM alone.
    agent, _fake = _fake_agent(
        [
            {
                "cluster": "SPEAKER_01",
                "name": "Grace Hopper",
                "confidence": "high",
                "reasoning": "clear",
            }
        ]
    )
    note_path = _cache_note(tmp_path)

    result = await run_speaker_resolution(
        note_path,
        "run-1",
        assign=("SPEAKER_00=Ada Lovelace",),
        finalise=False,
        agent=agent,
        audio_tool=FakeAudioTool(),
        cache=cache,
    )

    assert result.status == "resolved"
    resolved = cache.load("run-1", "resolved_transcript", RawTranscript)
    assert resolved is not None
    assert [u.speaker for u in resolved.utterances] == ["Ada Lovelace", "Grace Hopper"]
    note_text = note_path.read_text()
    assert "Ada Lovelace was SPEAKER_00" in note_text  # --assign'd: written
    assert "Grace Hopper was SPEAKER_01" not in note_text  # LLM-only: not written


async def test_run_speaker_resolution_persists_assignments_across_calls(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.store(
        "run-1",
        "raw_transcript",
        RawTranscript(
            clips=[],
            utterances=[Utterance(start=0.0, end=1.0, speaker="SPEAKER_00", text="hi")],
            audio_sha256=None,
        ),
    )
    agent, _fake = _fake_agent([])
    note_path = _cache_note(tmp_path)

    await run_speaker_resolution(
        note_path,
        "run-1",
        assign=("SPEAKER_00=Ada Lovelace",),
        finalise=False,
        agent=agent,
        audio_tool=FakeAudioTool(),
        cache=cache,
    )
    stored = cache.load("run-1", "assignments", AssignmentSet)
    assert stored is not None
    assert stored.assignments == [
        SpeakerAssignment(cluster="SPEAKER_00", name="Ada Lovelace")
    ]

    # A second call with no new --assign should still see the persisted one
    # and resolve quietly without needing input again.
    result = await run_speaker_resolution(
        note_path,
        "run-1",
        assign=(),
        finalise=False,
        agent=agent,
        audio_tool=FakeAudioTool(),
        cache=cache,
    )
    assert result.status == "resolved"


# --- CLI: `jake-tools transcript speakers` -----------------------------------


def test_speakers_cli_delegates_and_exits_zero_on_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    fake_result = SpeakersResponse(status="resolved", run_id="run-1", requests=[])

    async def fake_run_speaker_resolution(
        note_path: Path,
        run_id: str,
        *,
        assign: tuple[str, ...],
        finalise: bool,
        agent: object,
        audio_tool: object,
        cache: object,
    ) -> SpeakersResponse:
        captured["note_path"] = note_path
        captured["run_id"] = run_id
        captured["assign"] = assign
        captured["finalise"] = finalise
        return fake_result

    monkeypatch.setattr(
        transcript_cli, "run_speaker_resolution", fake_run_speaker_resolution
    )
    note_path = _cache_note(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "transcript",
            "speakers",
            str(note_path),
            "--run-id",
            "run-1",
            "--assign",
            "SPEAKER_00=Ada Lovelace",
            "--assign",
            "SPEAKER_01=Unknown",
            "--finalise",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == json.loads(fake_result.model_dump_json())
    assert captured["note_path"] == note_path
    assert captured["run_id"] == "run-1"
    assert captured["assign"] == ("SPEAKER_00=Ada Lovelace", "SPEAKER_01=Unknown")
    assert captured["finalise"] is True


def test_speakers_cli_exits_with_the_needs_input_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_result = SpeakersResponse(
        status="needs_input",
        run_id="run-1",
        requests=[
            SnippetRequest(cluster="SPEAKER_02", clip_paths=["/x.m4a"], context="hi")
        ],
    )

    async def fake_run_speaker_resolution(
        *args: object, **kwargs: object
    ) -> SpeakersResponse:
        return fake_result

    monkeypatch.setattr(
        transcript_cli, "run_speaker_resolution", fake_run_speaker_resolution
    )

    result = CliRunner().invoke(
        main,
        ["transcript", "speakers", str(_cache_note(tmp_path)), "--run-id", "run-1"],
    )

    assert result.exit_code == transcript_cli.NEEDS_INPUT_EXIT_CODE
    assert result.exit_code not in (0, 1)
    payload = json.loads(result.output)
    assert payload["status"] == "needs_input"
    assert payload["requests"][0]["cluster"] == "SPEAKER_02"


def test_speakers_cli_reports_domain_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def raising_run_speaker_resolution(
        *args: object, **kwargs: object
    ) -> SpeakersResponse:
        raise MissingRawTranscriptError("run-1")

    monkeypatch.setattr(
        transcript_cli, "run_speaker_resolution", raising_run_speaker_resolution
    )

    result = CliRunner().invoke(
        main,
        ["transcript", "speakers", str(_cache_note(tmp_path)), "--run-id", "run-1"],
    )

    assert result.exit_code == 1
    assert "run-1" in result.output


def test_speakers_cli_reports_claude_agent_errors_as_a_clean_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def raising_run_speaker_resolution(
        *args: object, **kwargs: object
    ) -> SpeakersResponse:
        raise ClaudeAgentError("agent call failed")

    monkeypatch.setattr(
        transcript_cli, "run_speaker_resolution", raising_run_speaker_resolution
    )

    result = CliRunner().invoke(
        main,
        ["transcript", "speakers", str(_cache_note(tmp_path)), "--run-id", "run-1"],
    )

    assert result.exit_code == 1
    assert "agent call failed" in result.output


def test_speakers_cli_help_documents_the_needs_input_exit_code() -> None:
    result = CliRunner().invoke(main, ["transcript", "speakers", "--help"])

    assert result.exit_code == 0
    assert "3" in result.output
    assert "needs_input" in result.output


async def test_invalid_assign_value_is_a_speakers_error(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "cache")
    cache.store(
        "run-1",
        "raw_transcript",
        RawTranscript(clips=[], utterances=[], audio_sha256=None),
    )
    agent, _fake = _fake_agent([])

    with pytest.raises(SpeakersError):
        await run_speaker_resolution(
            _cache_note(tmp_path),
            "run-1",
            assign=("not-a-valid-pair",),
            finalise=False,
            agent=agent,
            audio_tool=FakeAudioTool(),
            cache=cache,
        )
