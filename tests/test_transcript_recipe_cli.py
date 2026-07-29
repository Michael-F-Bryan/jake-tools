from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from jake_tools.claude import Reply
from jake_tools.prompting import StructuredPrompt
from jake_tools.transcripts.models import (
    SourceArtifact,
    TranscriptArtifact,
    TranscriptSourceRef,
    TranscriptTurn,
)
from jake_tools.transcripts.recipe_primitives import (
    RecipePrimitiveError,
    _polish_youtube_chunks,
    run_teams_meeting_recipe,
    run_youtube_source_notes_recipe,
)
from jake_tools.transcripts.teams_graph import (
    GraphCalendarEvent,
    GraphCallTranscript,
    GraphOnlineMeeting,
    TeamsMeetingSourceResult,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcript"
FIXTURE = FIXTURES_DIR / "youtube-sample.json3"


class FakeAgent:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.prompts: list[Any] = []

    async def run_structured[TModel: BaseModel](
        self, prompt: StructuredPrompt[TModel]
    ) -> tuple[TModel, Reply]:
        self.prompts.append(prompt)
        payload = self.responses.pop(0)
        result = prompt.response_model.model_validate(payload)
        return cast(TModel, result), Reply(text=json.dumps(payload))


def _source_fetcher(url: str, *, out_dir: Path, language: str) -> SourceArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "video-info.json"
    info_path.write_text('{"id": "video-123"}\n', encoding="utf-8")
    return SourceArtifact(
        kind="youtube",
        source_path=info_path,
        source_url=url,
        message_id="youtube:video-123",
        title="MAVLink tools - Patrick Pereira, Blue Robotics",
        date=dt.date(2025, 11, 28),
        organisation="PX4 Autopilot",
        raw_text_path=FIXTURE,
        metadata={
            "video_id": "video-123",
            "channel": "PX4 Autopilot",
            "duration_seconds": 9,
            "subtitle_track": language,
            "subtitle_kind": "automatic",
            "capture_method": "yt-dlp",
        },
        warnings=["Using automatic captions; review technical terminology."],
    )


def _recipe_responses() -> list[dict[str, Any]]:
    return [
        {
            "turns": [
                {
                    "start": 1.2,
                    "end": 3.4,
                    "speaker": "Speaker",
                    "text": "MAVLink tools solve integration problems.",
                },
                {
                    "start": 3.4,
                    "end": 5.2,
                    "speaker": "Speaker",
                    "text": "The protocol stays interoperable.",
                },
                {
                    "start": 6.2,
                    "end": 7.6,
                    "speaker": "Speaker",
                    "text": "The protocol stays interoperable.",
                },
                {
                    "start": 7.8,
                    "end": 9.4,
                    "speaker": "Speaker",
                    "text": "Blue Robotics uses the same messages.",
                },
            ],
        },
        {
            "chapters": {
                "chapters": [
                    {
                        "start": 1.2,
                        "end": 9.4,
                        "title": "MAVLink interoperability",
                        "summary": "The source describes MAVLink integration, interoperability, and shared messages.",
                    }
                ],
                "boundary_source": "llm",
            },
            "overview": {
                "summary": "The source introduces MAVLink tools for integration and protocol interoperability.",
                "key_points": [
                    "The tools address integration problems.",
                    "Blue Robotics uses the same interoperable messages.",
                ],
            },
        },
    ]


async def test_youtube_chunk_polish_preserves_turns_without_writing_chunk_files(
    tmp_path: Path,
) -> None:
    transcript = TranscriptArtifact(
        turns=[
            TranscriptTurn(
                start=0.0, end=4.0, speaker="Speaker", text="First chunk text."
            ),
            TranscriptTurn(
                start=301.0,
                end=305.0,
                speaker="Speaker",
                text="Second chunk text.",
            ),
        ],
        source_refs=[
            TranscriptSourceRef(turn_index=0, source_ref="captions:events[1]"),
            TranscriptSourceRef(turn_index=1, source_ref="captions:events[2]"),
        ],
    )
    agent = FakeAgent(
        [
            {
                "turns": [transcript.turns[0].model_dump(mode="json")],
            },
            {
                "turns": [transcript.turns[1].model_dump(mode="json")],
            },
        ]
    )

    polished, replies = await _polish_youtube_chunks(
        agent,
        transcript,
        context="Video title: test",
    )

    assert polished.turns == transcript.turns
    assert polished.source_refs == transcript.source_refs
    assert len(replies) == 2
    assert list(tmp_path.iterdir()) == []


async def test_youtube_chunk_polish_rejects_unfaithful_rewrite() -> None:
    transcript = TranscriptArtifact(
        turns=[
            TranscriptTurn(
                start=0.0,
                end=4.0,
                speaker="Speaker",
                text="MAVLink camera tools preserve low latency video streams.",
            )
        ]
    )
    agent = FakeAgent(
        [
            {
                "turns": [
                    {
                        "start": 0.0,
                        "end": 4.0,
                        "speaker": "Speaker",
                        "text": "The presenter recommends replacing every vehicle network.",
                    }
                ],
            }
        ]
    )

    with pytest.raises(RecipePrimitiveError, match="source fidelity"):
        await _polish_youtube_chunks(agent, transcript, context="Video title: test")


async def test_run_youtube_source_notes_recipe_writes_only_useful_artefacts(
    tmp_path: Path,
) -> None:
    agent = FakeAgent(_recipe_responses())
    out_dir = tmp_path / "run"
    vault_note = tmp_path / "Vault" / "MAVLink tools.md"

    result = await run_youtube_source_notes_recipe(
        agent,
        "https://www.youtube.com/watch?v=video-123",
        out_dir=out_dir,
        language="en-orig",
        vault_note=vault_note,
        dry_run=False,
        source_fetcher=_source_fetcher,
    )

    assert result["updated"] is True
    assert vault_note.exists()
    note = vault_note.read_text(encoding="utf-8")
    assert "> [!summary]" in note
    assert "## Key points" in note
    assert "https://www.youtube.com/watch?v=video-123&t=1s" in note
    assert "WEBVTT" not in note

    assert {path.name for path in out_dir.iterdir()} == {
        "manifest.json",
        "source-note.md",
        "transcript-polished.json",
        "transcript-raw.json",
        "video-info.json",
    }
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [(stage["stage"], stage["status"]) for stage in manifest["stages"]] == [
        ("source-note", "pass"),
        ("vault.write", "pass"),
    ]


async def test_youtube_recipe_marks_dry_run_write_as_skipped(tmp_path: Path) -> None:
    agent = FakeAgent(_recipe_responses())
    out_dir = tmp_path / "run"
    vault_note = tmp_path / "Vault" / "MAVLink tools.md"

    result = await run_youtube_source_notes_recipe(
        agent,
        "https://www.youtube.com/watch?v=video-123",
        out_dir=out_dir,
        language="en-orig",
        vault_note=vault_note,
        dry_run=True,
        source_fetcher=_source_fetcher,
    )

    assert result["updated"] is False
    assert not vault_note.exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"][-1] == {
        "stage": "vault.write",
        "status": "skipped",
        "artefacts": [],
    }


def _teams_source_fetcher(
    *, out_dir: Path, **_kwargs: object
) -> TeamsMeetingSourceResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_vtt_path = out_dir / "transcript.vtt"
    raw_vtt_path.write_text(
        (FIXTURES_DIR / "teams-sample.vtt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    event = GraphCalendarEvent.model_validate_json(
        (FIXTURES_DIR / "teams-calendar-event.json").read_text(encoding="utf-8")
    )
    online_meeting = GraphOnlineMeeting.model_validate(
        {
            "id": "meeting-123",
            "subject": event.subject,
            "joinWebUrl": "https://teams.microsoft.com/l/meetup-join/example",
        }
    )
    transcript = GraphCallTranscript.model_validate_json(
        (FIXTURES_DIR / "teams-transcript-metadata.json").read_text(encoding="utf-8")
    )
    source = SourceArtifact(
        kind="msgraph-teams",
        title=event.subject,
        date=event.start.as_perth_date() if event.start else None,
        raw_text_path=raw_vtt_path,
    )
    (out_dir / "source.json").write_text(
        source.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return TeamsMeetingSourceResult(
        source=source,
        calendar_event=event,
        online_meeting=online_meeting,
        transcript=transcript,
        raw_vtt_path=raw_vtt_path,
    )


def test_teams_recipe_dry_run_does_not_claim_a_vault_write(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    vault_note = tmp_path / "Vault" / "meeting.md"

    result = run_teams_meeting_recipe(
        account="csu-teams",
        profile="default",
        out_dir=out_dir,
        vault_note=vault_note,
        dry_run=True,
        source_fetcher=_teams_source_fetcher,
    )

    assert result["updated"] is False
    assert not vault_note.exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"][-1]["stage"] == "note.write"
    assert manifest["stages"][-1]["status"] == "skipped"
