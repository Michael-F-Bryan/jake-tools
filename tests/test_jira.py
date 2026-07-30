from __future__ import annotations

import json
from typing import Any

import pytest

from jake_tools.jira import JiraClient, JiraError, JiraIssue, normalise_jira_key


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object,
        *,
        reason: str = "OK",
    ) -> None:
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.content = self.text.encode()
        self._payload = payload
        self.reason = reason

    def json(self) -> object:
        return self._payload


class InvalidJsonResponse(FakeResponse):
    def __init__(self) -> None:
        super().__init__(200, "not-json")
        self.text = "not-json"
        self.content = b"not-json"

    def json(self) -> object:
        raise ValueError("invalid JSON")


class FakeSession:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


def issue_payload(
    key: str,
    *,
    summary: str,
    status: str = "In Progress",
    status_category: str = "In Progress",
    assignee: str | None = None,
    issue_type: str = "Task",
    parent_key: str | None = None,
    parent_summary: str | None = None,
) -> dict[str, object]:
    fields: dict[str, object] = {
        "summary": summary,
        "status": {
            "name": status,
            "statusCategory": {"name": status_category},
        },
        "assignee": {"displayName": assignee} if assignee else None,
        "issuetype": {"name": issue_type},
    }
    if parent_key is not None:
        fields["parent"] = {
            "key": parent_key,
            "fields": {"summary": parent_summary},
        }
    return {"key": key, "fields": fields}


def test_normalise_jira_key_upcases_and_strips() -> None:
    assert normalise_jira_key(" sf-353 ") == "SF-353"


def test_normalise_jira_key_rejects_malformed_keys() -> None:
    with pytest.raises(JiraError, match="Invalid Jira issue key"):
        normalise_jira_key("not a key")


def test_jira_client_fetches_active_assigned_issues_with_parent_data() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "isLast": True,
                "issues": [
                    issue_payload(
                        "SF-427",
                        summary="Evaluate PX4 external control methods",
                        assignee="Michael Bryan",
                        parent_key="SF-131",
                        parent_summary="Production Vehicle - Investigations and Overhead",
                    ),
                    issue_payload(
                        "SF-1",
                        summary="Simulator work",
                        assignee="Michael Bryan",
                        issue_type="Project / Phase",
                    ),
                ],
            },
        )
    )
    client = JiraClient(
        base_url="sunfishrobotics.atlassian.net",
        email="michael@example.test",
        api_token="secret-token",
        session=session,
    )

    issues = client.get_active_assigned_issues("SF")

    assert issues == [
        JiraIssue(
            key="SF-427",
            summary="Evaluate PX4 external control methods",
            status="In Progress",
            statusCategory="In Progress",
            assignee="Michael Bryan",
            issueType="Task",
            parentKey="SF-131",
            parentSummary="Production Vehicle - Investigations and Overhead",
        ),
        JiraIssue(
            key="SF-1",
            summary="Simulator work",
            status="In Progress",
            statusCategory="In Progress",
            assignee="Michael Bryan",
            issueType="Project / Phase",
        ),
    ]
    assert session.requests == [
        {
            "method": "POST",
            "url": "https://sunfishrobotics.atlassian.net/rest/api/3/search/jql",
            "auth": ("michael@example.test", "secret-token"),
            "headers": {
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            "json": {
                "jql": 'project = SF AND assignee = currentUser() AND status in ("In Progress", "Blocked", "In Review") ORDER BY key',
                "fields": [
                    "summary",
                    "status",
                    "assignee",
                    "issuetype",
                    "parent",
                ],
                "maxResults": 100,
            },
            "timeout": 30,
        }
    ]


def test_jira_client_gets_one_detailed_issue() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            issue_payload(
                "SF-304",
                summary="Bench test PX4",
                assignee="David Htet",
                parent_key="SF-131",
                parent_summary="Production Vehicle - Investigations and Overhead",
            ),
        )
    )
    client = JiraClient(
        base_url="https://sunfishrobotics.atlassian.net/",
        email="michael@example.test",
        api_token="secret-token",
        session=session,
    )

    issue = client.get_issue("sf-304")

    assert issue == JiraIssue(
        key="SF-304",
        summary="Bench test PX4",
        status="In Progress",
        statusCategory="In Progress",
        assignee="David Htet",
        issueType="Task",
        parentKey="SF-131",
        parentSummary="Production Vehicle - Investigations and Overhead",
    )
    assert session.requests[0]["method"] == "GET"
    assert session.requests[0]["url"].endswith("/rest/api/3/issue/SF-304")
    assert session.requests[0]["params"] == {
        "fields": "summary,status,assignee,issuetype,parent"
    }


def test_jira_client_batches_issue_lookup_and_paginates_by_token() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "nextPageToken": "second-page",
                "isLast": False,
                "issues": [
                    issue_payload(
                        "SF-304",
                        summary="Bench test PX4",
                        status="Done",
                        status_category="Done",
                        assignee="Michael Bryan",
                        parent_key="SF-131",
                        parent_summary="Production Vehicle",
                    )
                ],
            },
        ),
        FakeResponse(
            200,
            {
                "isLast": True,
                "issues": [
                    issue_payload(
                        "SF-438",
                        summary="Check newer flight controller boards",
                        status="To Do",
                        status_category="To Do",
                    )
                ],
            },
        ),
    )
    client = JiraClient(
        base_url="https://sunfishrobotics.atlassian.net",
        email="michael@example.test",
        api_token="secret-token",
        session=session,
    )

    issues = client.get_issues(["SF-438", "SF-304", "SF-438"])

    assert [issue.key for issue in issues] == ["SF-304", "SF-438"]
    assert issues[0].status_category == "Done"
    assert issues[0].parent_key == "SF-131"
    assert issues[1].assignee is None
    assert len(session.requests) == 2
    assert session.requests[0]["json"]["jql"] == ("key in (SF-304,SF-438) ORDER BY key")
    assert "nextPageToken" not in session.requests[0]["json"]
    assert session.requests[1]["json"]["nextPageToken"] == "second-page"


def test_jira_client_skips_request_for_empty_issue_lookup() -> None:
    session = FakeSession()
    client = JiraClient(
        base_url="https://sunfishrobotics.atlassian.net",
        email="michael@example.test",
        api_token="secret-token",
        session=session,
    )

    assert client.get_issues([]) == []
    assert session.requests == []


@pytest.mark.parametrize(
    ("base_url", "email", "api_token", "message"),
    [
        ("  ", "michael@example.test", "secret-token", "base URL is required"),
        (
            "https://sunfishrobotics.atlassian.net",
            "  ",
            "secret-token",
            "email is required",
        ),
        (
            "https://sunfishrobotics.atlassian.net",
            "michael@example.test",
            "  ",
            "API token is required",
        ),
    ],
)
def test_jira_client_rejects_missing_configuration(
    base_url: str,
    email: str,
    api_token: str,
    message: str,
) -> None:
    with pytest.raises(JiraError, match=message):
        JiraClient(base_url=base_url, email=email, api_token=api_token)


def test_jira_client_reports_http_failure_without_exposing_token() -> None:
    session = FakeSession(
        FakeResponse(401, {"errorMessages": ["Unauthorized"]}, reason="Unauthorized")
    )
    client = JiraClient(
        base_url="https://sunfishrobotics.atlassian.net",
        email="michael@example.test",
        api_token="secret-token",
        session=session,
    )

    with pytest.raises(JiraError) as raised:
        client.get_issues(["SF-304"])

    message = str(raised.value)
    assert "401 Unauthorized" in message
    assert "POST /rest/api/3/search/jql" in message
    assert "secret-token" not in message


def test_jira_client_reports_malformed_json_with_request_context() -> None:
    client = JiraClient(
        base_url="https://sunfishrobotics.atlassian.net",
        email="michael@example.test",
        api_token="secret-token",
        session=FakeSession(InvalidJsonResponse()),
    )

    with pytest.raises(JiraError) as raised:
        client.get_issues(["SF-304"])

    assert "invalid JSON" in str(raised.value)
    assert "POST /rest/api/3/search/jql" in str(raised.value)
    assert "not-json" in str(raised.value)
