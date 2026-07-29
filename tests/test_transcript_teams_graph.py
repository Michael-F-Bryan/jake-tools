from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jake_tools.transcripts.teams_graph import (
    GraphCalendarEvent,
    GraphCallTranscript,
    TeamsGraphClient,
    source_from_teams_meeting,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcript"


class _Response:
    def __init__(self, payload: object | None = None, *, text: str = "") -> None:
        self.status_code = 200
        self.reason = "OK"
        self.text = text
        self.content = text.encode() if text else json.dumps(payload or {}).encode()
        self._payload = payload or {}

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Mapping[str, str] | None]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.requests.append((method, url, kwargs.get("headers")))
        if url.endswith("/me/events/event-123"):
            return _Response(
                json.loads((FIXTURES_DIR / "teams-calendar-event.json").read_text())
            )
        if url.endswith("/me/onlineMeetings"):
            return _Response(
                {
                    "value": [
                        {
                            "id": "meeting-123",
                            "subject": "Marine Rescue Comms Support presentation for SES",
                            "joinWebUrl": "https://teams.microsoft.com/l/meetup-join/example",
                        }
                    ]
                }
            )
        if url.endswith("/me/onlineMeetings/meeting-123/transcripts"):
            return _Response(
                {
                    "value": [
                        json.loads(
                            (
                                FIXTURES_DIR / "teams-transcript-metadata.json"
                            ).read_text()
                        )
                    ]
                }
            )
        if url.endswith("/content"):
            return _Response(text=(FIXTURES_DIR / "teams-sample.vtt").read_text())
        raise AssertionError(f"unexpected URL: {url}")


def test_graph_fixture_models_capture_join_url_and_transcript_metadata() -> None:
    event = GraphCalendarEvent.model_validate_json(
        (FIXTURES_DIR / "teams-calendar-event.json").read_text()
    )
    transcript = GraphCallTranscript.model_validate_json(
        (FIXTURES_DIR / "teams-transcript-metadata.json").read_text()
    )

    assert event.online_meeting is not None
    assert event.online_meeting.join_url.endswith("/example")
    assert event.start is not None
    event_date = event.start.as_perth_date()
    assert event_date is not None
    assert event_date.isoformat() == "2026-07-06"
    assert transcript.id == "transcript-abc"
    assert transcript.transcript_content_url.endswith("/content")


def test_source_from_teams_meeting_records_raw_vtt_as_run_artifact(
    tmp_path: Path,
) -> None:
    session = _FakeSession()

    def client_factory(_token_provider):
        return TeamsGraphClient(token_provider=lambda: "token", session=session)

    result = source_from_teams_meeting(
        account="csu-teams",
        out_dir=tmp_path,
        token_file=tmp_path / "token.json",
        event_id="event-123",
        organisation="CSU",
        project="DUM-C",
        client_factory=client_factory,
    )

    assert result.source.kind == "msgraph-teams"
    assert result.source.message_id == "msgraph-teams:meeting-123:transcript-abc"
    assert result.source.raw_text_path == (tmp_path / "transcript.vtt").resolve()
    assert result.source.organisation == "CSU"
    assert result.source.project == "DUM-C"
    assert (tmp_path / "calendar-event.json").exists()
    assert (tmp_path / "online-meeting.json").exists()
    assert (tmp_path / "transcripts.json").exists()
    assert (
        (tmp_path / "transcript.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    )
    assert (tmp_path / "source.json").exists()
    content_request = session.requests[-1]
    assert content_request[2] is not None
    assert content_request[2]["Accept"] == "text/vtt"
