from __future__ import annotations

import json
from typing import Any

from jake_tools.clockify import (
    ClockifyClient,
    ClockifyError,
    ClockifyUser,
    JiraIssueRef,
    clockify_project_name_for_jira,
    clockify_task_name_for_jira,
    normalise_jira_key,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.content = self.text.encode()
        self._payload = payload
        self.reason = "OK"

    def json(self) -> dict[str, object]:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return self.response


def test_clockify_get_user_uses_requests_session_and_api_key() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "id": "user-123",
                "name": "Michael Bryan",
                "email": "michael@example.test",
                "activeWorkspace": "workspace-1",
                "defaultWorkspace": "workspace-2",
            },
        )
    )
    client = ClockifyClient(api_key="secret-key", session=session)

    user = client.get_user()

    assert user == ClockifyUser(
        id="user-123",
        name="Michael Bryan",
        email="michael@example.test",
        activeWorkspace="workspace-1",
        defaultWorkspace="workspace-2",
    )
    assert session.requests == [
        {
            "method": "GET",
            "url": "https://api.clockify.me/api/v1/user",
            "headers": {
                "Accept": "application/json",
                "X-Api-Key": "secret-key",
            },
            "json": None,
            "timeout": 30,
        }
    ]


def test_clockify_jira_project_names_do_not_start_with_ticket_key() -> None:
    assert (
        clockify_project_name_for_jira(
            "Production Vehicle - Investigations and Overhead"
        )
        == "Production Vehicle - Investigations and Overhead"
    )


def test_clockify_jira_task_names_keep_ticket_key() -> None:
    assert (
        clockify_task_name_for_jira(
            "sf-353",
            "Vehicle Control Logic - Preliminary Architecture",
        )
        == "SF-353 Vehicle Control Logic - Preliminary Architecture"
    )


def test_jira_issue_ref_keeps_project_note_separate_from_project_name() -> None:
    issue = JiraIssueRef(key="SF-131", summary="Production Vehicle")

    assert issue.project_name == "Production Vehicle"
    assert issue.task_name == "SF-131 Production Vehicle"
    assert issue.project_note == "Jira: SF-131"


def test_clockify_jira_names_reject_invalid_ticket_keys() -> None:
    try:
        normalise_jira_key("not a key")
    except ClockifyError as exc:
        assert "Invalid Jira issue key" in str(exc)
    else:
        raise AssertionError("invalid Jira key was accepted")
