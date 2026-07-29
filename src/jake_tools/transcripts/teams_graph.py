from __future__ import annotations

import datetime as dt
import json
import urllib.parse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel, Field

from .errors import TranscriptError
from .models import SourceArtifact

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
REQUIRED_SCOPES = frozenset(
    {
        "Calendars.Read",
        "OnlineMeetings.Read",
        "OnlineMeetingTranscript.Read.All",
    }
)
PERTH_TZ = ZoneInfo("Australia/Perth")

JsonObject = dict[str, object]


class HttpSession(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


class TeamsGraphError(TranscriptError):
    pass


class TokenProvider(Protocol):
    def __call__(self) -> str: ...


class TokenFileProvider:
    def __init__(self, token_file: Path) -> None:
        self._token_file = token_file.expanduser()

    def __call__(self) -> str:
        if not self._token_file.exists():
            raise TeamsGraphError(
                f"Microsoft Graph token file does not exist: {self._token_file}"
            )
        payload = json.loads(self._token_file.read_text(encoding="utf-8"))
        token = payload.get("access_token") or payload.get("accessToken")
        if not isinstance(token, str) or not token.strip():
            raise TeamsGraphError(
                f"Microsoft Graph token file is missing access_token: {self._token_file}"
            )
        return token


class GraphDateTime(BaseModel):
    date_time: str = Field(alias="dateTime")
    time_zone: str = Field(default="UTC", alias="timeZone")

    def as_perth_date(self) -> dt.date | None:
        try:
            parsed = dt.datetime.fromisoformat(self.date_time.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.astimezone(PERTH_TZ).date()


class GraphEventOnlineMeeting(BaseModel):
    join_url: str = Field(default="", alias="joinUrl")


class GraphCalendarEvent(BaseModel):
    id: str
    subject: str = ""
    start: GraphDateTime | None = None
    end: GraphDateTime | None = None
    online_meeting: GraphEventOnlineMeeting | None = Field(
        default=None, alias="onlineMeeting"
    )


class GraphCalendarResponse(BaseModel):
    value: list[GraphCalendarEvent] = Field(default_factory=list)


class GraphOnlineMeeting(BaseModel):
    id: str
    subject: str = ""
    join_web_url: str = Field(default="", alias="joinWebUrl")


class GraphOnlineMeetingsResponse(BaseModel):
    value: list[GraphOnlineMeeting] = Field(default_factory=list)


class GraphCallTranscript(BaseModel):
    id: str
    created_date_time: str | None = Field(default=None, alias="createdDateTime")
    transcript_content_url: str = Field(default="", alias="transcriptContentUrl")


class GraphCallTranscriptsResponse(BaseModel):
    value: list[GraphCallTranscript] = Field(default_factory=list)


class TeamsMeetingSourceResult(BaseModel):
    source: SourceArtifact
    calendar_event: GraphCalendarEvent
    online_meeting: GraphOnlineMeeting
    transcript: GraphCallTranscript
    raw_vtt_path: Path


class TeamsGraphClient:
    def __init__(
        self,
        *,
        token_provider: TokenProvider,
        graph_root: str = GRAPH_ROOT,
        session: HttpSession | None = None,
    ) -> None:
        self._token_provider = token_provider
        self._graph_root = graph_root.rstrip("/")
        self._session = session or requests.Session()

    def get_calendar_event(self, event_id: str) -> GraphCalendarEvent:
        data = self._graph_json(
            "GET",
            f"/me/events/{urllib.parse.quote(event_id, safe='')}",
            params={"$select": "id,subject,start,end,onlineMeeting"},
        )
        return GraphCalendarEvent.model_validate(data)

    def find_recent_calendar_event(
        self,
        *,
        days_back: int,
        query: str | None,
        now: dt.datetime | None = None,
    ) -> GraphCalendarEvent:
        if days_back < 1:
            raise TeamsGraphError("days_back must be at least 1.")
        now = now or dt.datetime.now(PERTH_TZ)
        start = now - dt.timedelta(days=days_back)
        data = self._graph_json(
            "GET",
            "/me/calendarView",
            params={
                "startDateTime": start.isoformat(),
                "endDateTime": now.isoformat(),
                "$select": "id,subject,start,end,onlineMeeting",
                "$orderby": "start/dateTime desc",
                "$top": "25",
            },
        )
        events = GraphCalendarResponse.model_validate(data).value
        if query:
            needle = query.casefold()
            events = [event for event in events if needle in event.subject.casefold()]
        events = [
            event
            for event in events
            if event.online_meeting and event.online_meeting.join_url
        ]
        if not events:
            detail = f" matching {query!r}" if query else ""
            raise TeamsGraphError(
                f"No Teams calendar events found in the last {days_back} day(s){detail}."
            )
        return events[0]

    def resolve_online_meeting(self, join_url: str) -> GraphOnlineMeeting:
        if not join_url:
            raise TeamsGraphError("Calendar event is missing onlineMeeting.joinUrl.")
        escaped = join_url.replace("'", "''")
        data = self._graph_json(
            "GET",
            "/me/onlineMeetings",
            params={"$filter": f"JoinWebUrl eq '{escaped}'"},
        )
        meetings = GraphOnlineMeetingsResponse.model_validate(data).value
        if not meetings:
            raise TeamsGraphError(
                "No onlineMeeting matched the calendar event join URL."
            )
        return meetings[0]

    def list_transcripts(self, online_meeting_id: str) -> list[GraphCallTranscript]:
        data = self._graph_json(
            "GET",
            f"/me/onlineMeetings/{urllib.parse.quote(online_meeting_id, safe='')}/transcripts",
        )
        return GraphCallTranscriptsResponse.model_validate(data).value

    def fetch_transcript_vtt(self, content_url: str) -> str:
        response = self._request(
            "GET",
            content_url,
            headers={"Accept": "text/vtt"},
        )
        return response.text

    def _graph_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> JsonObject:
        response = self._request(
            method,
            f"{self._graph_root}{path}",
            params=params,
            headers={"Accept": "application/json"},
        )
        if not response.content:
            return {}
        return cast(JsonObject, response.json())

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        request_headers = {
            "Authorization": f"Bearer {self._token_provider()}",
            **(dict(headers) if headers else {}),
        }
        try:
            response = self._session.request(
                method,
                url,
                params=params,
                headers=request_headers,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise TeamsGraphError(f"Microsoft Graph request failed: {exc}") from exc
        if response.status_code >= 400:
            raise TeamsGraphError(
                f"Microsoft Graph request failed: {response.status_code} {response.reason}\n{response.text}"
            )
        return response


def source_from_teams_meeting(
    *,
    account: str,
    out_dir: Path,
    token_file: Path,
    event_id: str | None = None,
    days_back: int = 14,
    query: str | None = None,
    organisation: str | None = None,
    project: str | None = None,
    client_factory: Callable[[TokenProvider], TeamsGraphClient] | None = None,
) -> TeamsMeetingSourceResult:
    """Fetch a Teams meeting transcript over Microsoft Graph.

    `token_file`, `organisation`, and `project` are policy the CLI layer
    decides (which account maps to which token file, org, and project); this
    function only needs a resolved token file path and, optionally, the
    provenance strings to stamp onto the resulting SourceArtifact.
    """
    provider = TokenFileProvider(token_file)
    client = (
        client_factory(provider)
        if client_factory
        else TeamsGraphClient(token_provider=provider)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    event = (
        client.get_calendar_event(event_id)
        if event_id
        else client.find_recent_calendar_event(days_back=days_back, query=query)
    )
    event_payload_path = out_dir / "calendar-event.json"
    event_payload_path.write_text(
        event.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8"
    )

    join_url = event.online_meeting.join_url if event.online_meeting else ""
    online_meeting = client.resolve_online_meeting(join_url)
    online_meeting_path = out_dir / "online-meeting.json"
    online_meeting_path.write_text(
        online_meeting.model_dump_json(by_alias=True, indent=2) + "\n", encoding="utf-8"
    )

    transcripts = client.list_transcripts(online_meeting.id)
    transcripts_path = out_dir / "transcripts.json"
    transcripts_path.write_text(
        json.dumps(
            [item.model_dump(mode="json", by_alias=True) for item in transcripts],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    transcript = next(
        (item for item in transcripts if item.transcript_content_url), None
    )
    if transcript is None:
        raise TeamsGraphError("Online meeting has no transcript content URL.")

    raw_vtt = client.fetch_transcript_vtt(transcript.transcript_content_url)
    raw_vtt_path = out_dir / "transcript.vtt"
    raw_vtt_path.write_text(raw_vtt, encoding="utf-8")

    source = SourceArtifact(
        kind="msgraph-teams",
        source_url=transcript.transcript_content_url,
        message_id=f"msgraph-teams:{online_meeting.id}:{transcript.id}",
        title=event.subject or online_meeting.subject or "Teams meeting",
        date=event.start.as_perth_date() if event.start else None,
        organisation=organisation,
        project=project,
        raw_text_path=raw_vtt_path.resolve(),
        metadata={
            "account": account,
            "calendar_event_id": event.id,
            "online_meeting_id": online_meeting.id,
            "transcript_id": transcript.id,
            "raw_vtt_path": str(raw_vtt_path.resolve()),
        },
    )
    source_path = out_dir / "source.json"
    source_path.write_text(source.model_dump_json(indent=2) + "\n", encoding="utf-8")

    return TeamsMeetingSourceResult(
        source=source,
        calendar_event=event,
        online_meeting=online_meeting,
        transcript=transcript,
        raw_vtt_path=raw_vtt_path.resolve(),
    )
