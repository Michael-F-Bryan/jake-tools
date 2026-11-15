from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from jake_tools.clockify import (
    ClockifyClient,
    ClockifyClientRecord,
    ClockifyError,
    ClockifyPaginationError,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    JiraIssueRef,
    clockify_project_name_for_jira,
    clockify_task_name_for_jira,
)
from jake_tools.jira import JiraError


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        if headers is not None:
            self.headers = headers
        self.text = json.dumps(payload)
        self.content = self.text.encode()
        self._payload = payload
        self.reason = "OK"

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


def test_clockify_wraps_invalid_json_responses_with_request_context() -> None:
    client = ClockifyClient(
        api_key="secret-key",
        session=FakeSession(InvalidJsonResponse()),
    )

    with pytest.raises(ClockifyError) as raised:
        client.get_user()

    assert "invalid JSON" in str(raised.value)
    assert "GET /user" in str(raised.value)
    assert "not-json" in str(raised.value)


def test_clockify_rejects_blank_base_url() -> None:
    with pytest.raises(ClockifyError, match="base URL is required"):
        ClockifyClient(api_key="secret-key", base_url="  ")


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
    # normalise_jira_key lives in jira.py and raises JiraError natively; see
    # test_jira.py for that behaviour. Here we only check that a naming
    # helper built on top of it surfaces the same rejection.
    with pytest.raises(JiraError, match="Invalid Jira issue key"):
        clockify_task_name_for_jira("not a key", "Some summary")


def test_clockify_lists_active_clients_and_projects() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            [{"id": "client-1", "name": "Sunfish Robotics", "archived": False}],
        ),
        FakeResponse(
            200,
            [
                {
                    "id": "project-1",
                    "name": "Production Vehicle",
                    "note": "Jira: SF-131",
                    "archived": False,
                    "billable": False,
                    "color": "#689F38",
                    "public": True,
                    "clientId": "client-1",
                }
            ],
        ),
    )
    client = ClockifyClient(api_key="secret-key", session=session)

    clients = client.get_clients("workspace-1")
    projects = client.get_projects("workspace-1", archived=False)

    assert clients == [
        ClockifyClientRecord(
            id="client-1",
            name="Sunfish Robotics",
            archived=False,
        )
    ]
    assert projects == [
        ClockifyProject(
            id="project-1",
            name="Production Vehicle",
            note="Jira: SF-131",
            archived=False,
            billable=False,
            color="#689F38",
            public=True,
            clientId="client-1",
        )
    ]
    assert session.requests[0]["params"] == {"page-size": 5000, "page": 1}
    assert session.requests[1]["params"] == {
        "archived": "false",
        "hydrated": "false",
        "page-size": 5000,
        "page": 1,
    }


def test_clockify_get_projects_paginates_until_a_short_page() -> None:
    def project_payload(index: int) -> dict[str, object]:
        return {
            "id": f"project-{index}",
            "name": f"Project {index}",
            "note": "",
            "archived": False,
            "billable": False,
            "color": "#689F38",
            "public": True,
            "clientId": "client-1",
        }

    session = FakeSession(
        FakeResponse(200, [project_payload(1), project_payload(2)]),
        FakeResponse(200, [project_payload(3)]),
    )
    client = ClockifyClient(api_key="secret-key", session=session)
    client._PAGE_SIZE = 2  # shrink the page so a 2-page fixture is enough

    projects = client.get_projects("workspace-1", archived=False)

    assert [project.id for project in projects] == [
        "project-1",
        "project-2",
        "project-3",
    ]
    assert len(session.requests) == 2
    assert session.requests[0]["params"]["page"] == 1
    assert session.requests[1]["params"]["page"] == 2


def test_clockify_lists_active_and_done_tasks() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            [
                {
                    "id": "task-active",
                    "name": "SF-427 Evaluate PX4 external control methods",
                    "projectId": "project-1",
                    "status": "ACTIVE",
                }
            ],
        ),
        FakeResponse(
            200,
            [
                {
                    "id": "task-done",
                    "name": "SF-304 Bench test PX4",
                    "projectId": "project-1",
                    "status": "DONE",
                }
            ],
        ),
    )
    client = ClockifyClient(api_key="secret-key", session=session)

    active = client.get_tasks("workspace-1", "project-1", active=True)
    done = client.get_tasks("workspace-1", "project-1", active=False)

    assert active[0] == ClockifyTask(
        id="task-active",
        name="SF-427 Evaluate PX4 external control methods",
        projectId="project-1",
        status="ACTIVE",
    )
    assert done[0].status == "DONE"
    assert session.requests[0]["params"] == {
        "is-active": "true",
        "page-size": 5000,
        "page": 1,
    }
    assert session.requests[1]["params"] == {
        "is-active": "false",
        "page-size": 5000,
        "page": 1,
    }


def test_clockify_gets_project_and_task_by_id() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "id": "project-1",
                "name": "Production Vehicle",
                "note": "Jira: SF-131",
                "archived": False,
                "billable": False,
                "color": "#689F38",
                "public": True,
                "clientId": "client-1",
            },
        ),
        FakeResponse(
            200,
            {
                "id": "task-1",
                "name": "SF-304 Bench test PX4",
                "projectId": "project-1",
                "status": "ACTIVE",
            },
        ),
    )
    client = ClockifyClient(api_key="secret-key", session=session)

    project = client.get_project("workspace-1", "project-1")
    task = client.get_task("workspace-1", "project-1", "task-1")

    assert project.id == "project-1"
    assert task.status == "ACTIVE"
    assert session.requests[0]["url"].endswith(
        "/workspaces/workspace-1/projects/project-1"
    )
    assert session.requests[1]["url"].endswith(
        "/workspaces/workspace-1/projects/project-1/tasks/task-1"
    )


def test_clockify_creates_jira_project_and_task() -> None:
    session = FakeSession(
        FakeResponse(
            201,
            {
                "id": "project-1",
                "name": "Production Vehicle",
                "note": "Jira: SF-131",
                "archived": False,
                "billable": False,
                "color": "#689F38",
                "public": True,
                "clientId": "client-1",
            },
        ),
        FakeResponse(
            201,
            {
                "id": "task-1",
                "name": "SF-427 Evaluate PX4 external control methods",
                "projectId": "project-1",
                "status": "ACTIVE",
            },
        ),
    )
    client = ClockifyClient(api_key="secret-key", session=session)

    project = client.create_project(
        "workspace-1",
        name="Production Vehicle",
        note="Jira: SF-131",
        client_id="client-1",
    )
    task = client.create_task(
        "workspace-1",
        project.id,
        name="SF-427 Evaluate PX4 external control methods",
    )

    assert project.id == "project-1"
    assert task.id == "task-1"
    assert session.requests[0]["json"] == {
        "billable": False,
        "clientId": "client-1",
        "isPublic": True,
        "name": "Production Vehicle",
        "note": "Jira: SF-131",
    }
    assert session.requests[1]["json"] == {
        "name": "SF-427 Evaluate PX4 external control methods"
    }


def test_clockify_updates_project_name_without_losing_metadata() -> None:
    session = FakeSession(
        FakeResponse(
            200,
            {
                "id": "project-1",
                "name": "New name",
                "note": "Jira: SF-131",
                "archived": False,
                "billable": False,
                "color": "#689F38",
                "public": True,
                "clientId": "client-1",
            },
        )
    )
    client = ClockifyClient(api_key="secret-key", session=session)
    project = ClockifyProject(
        id="project-1",
        name="Old name",
        note="Jira: SF-131",
        archived=False,
        billable=False,
        color="#689F38",
        public=True,
        clientId="client-1",
    )

    updated = client.update_project_name("workspace-1", project, "New name")

    assert updated.name == "New name"
    assert session.requests[0]["json"] == {
        "archived": False,
        "billable": False,
        "clientId": "client-1",
        "color": "#689F38",
        "isPublic": True,
        "name": "New name",
        "note": "Jira: SF-131",
    }


def test_clockify_renames_and_completes_task() -> None:
    # Each update_task call re-fetches the task first to check for drift
    # (GET), then writes (PUT): four requests for two update_task calls.
    session = FakeSession(
        FakeResponse(
            200,
            {
                "id": "task-1",
                "name": "SF-305 Wet test PX4 using Tom's thesis bot",
                "projectId": "project-1",
                "status": "ACTIVE",
            },
        ),
        FakeResponse(
            200,
            {
                "id": "task-1",
                "name": "SF-305 Wet test PX4 using Zoda",
                "projectId": "project-1",
                "status": "ACTIVE",
            },
        ),
        FakeResponse(
            200,
            {
                "id": "task-1",
                "name": "SF-305 Wet test PX4 using Zoda",
                "projectId": "project-1",
                "status": "ACTIVE",
            },
        ),
        FakeResponse(
            200,
            {
                "id": "task-1",
                "name": "SF-305 Wet test PX4 using Zoda",
                "projectId": "project-1",
                "status": "DONE",
            },
        ),
    )
    client = ClockifyClient(api_key="secret-key", session=session)
    task = ClockifyTask(
        id="task-1",
        name="SF-305 Wet test PX4 using Tom's thesis bot",
        projectId="project-1",
        status="ACTIVE",
    )

    renamed = client.update_task(
        "workspace-1",
        task,
        name="SF-305 Wet test PX4 using Zoda",
    )
    completed = client.update_task(
        "workspace-1",
        renamed,
        status="DONE",
    )

    assert completed.status == "DONE"
    assert session.requests[0]["method"] == "GET"
    assert session.requests[1]["json"] == {
        "name": "SF-305 Wet test PX4 using Zoda",
        "status": "ACTIVE",
    }
    assert session.requests[2]["method"] == "GET"
    assert session.requests[3]["json"] == {
        "name": "SF-305 Wet test PX4 using Zoda",
        "status": "DONE",
    }


def test_clockify_update_task_aborts_when_remote_record_has_drifted() -> None:
    # The plan was built when the task was ACTIVE, but the pre-write re-fetch
    # finds it DONE (e.g. someone completed it in the Clockify UI meanwhile).
    # update_task must refuse to write instead of silently reverting it.
    session = FakeSession(
        FakeResponse(
            200,
            {
                "id": "task-1",
                "name": "SF-305 Wet test PX4 using Zoda",
                "projectId": "project-1",
                "status": "DONE",
            },
        ),
    )
    client = ClockifyClient(api_key="secret-key", session=session)
    stale_snapshot = ClockifyTask(
        id="task-1",
        name="SF-305 Wet test PX4 using Zoda",
        projectId="project-1",
        status="ACTIVE",
    )

    with pytest.raises(ClockifyError, match="drifted"):
        client.update_task(
            "workspace-1",
            stale_snapshot,
            name="SF-305 Wet test PX4 using Zoda v2",
        )

    # No write was attempted once drift was detected.
    assert len(session.requests) == 1


def _project(index: int) -> dict[str, object]:
    return {
        "id": f"project-{index}",
        "name": f"Project {index}",
        "note": "",
        "archived": False,
        "billable": False,
        "color": "#689F38",
        "public": True,
        "clientId": "client-1",
    }


def _scan(
    *pages: FakeResponse, page_size: int = 2, max_pages: int | None = None
) -> tuple[ClockifyClient, FakeSession, list[ClockifyProject]]:
    session = FakeSession(*pages)
    client = ClockifyClient(api_key="secret-key", session=session)
    client._PAGE_SIZE = page_size
    if max_pages is not None:
        client._MAX_PAGES = max_pages
    return client, session, client.get_projects("workspace-1", archived=False)


def _last(flag: str) -> dict[str, str]:
    return {"Last-Page": flag}


def test_short_page_with_last_page_false_continues() -> None:
    # Regression: a short page used to end the scan and silently truncate it.
    client, session, projects = _scan(
        FakeResponse(200, [_project(1)], _last("false")),
        FakeResponse(200, [_project(2)], _last("true")),
    )
    assert [p.id for p in projects] == ["project-1", "project-2"]
    assert [r["params"]["page"] for r in session.requests] == [1, 2]
    (read,) = client.pagination_reads
    assert (read["pages"], read["records"]) == (2, 2)
    assert read["completion"] == "last_page_header"


def test_full_page_with_last_page_true_ends_without_another_request() -> None:
    client, session, projects = _scan(
        FakeResponse(200, [_project(1), _project(2)], _last("true"))
    )
    assert len(projects) == 2
    assert len(session.requests) == 1
    assert client.pagination_reads[0]["completion"] == "last_page_header"


def test_last_page_header_is_case_insensitive_on_a_real_requests_response() -> None:
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps([_project(1)]).encode()
    response.headers["last-page"] = " FALSE "
    second = FakeResponse(200, [_project(2)], _last("true"))
    _, session, projects = _scan(response, second)  # type: ignore[arg-type]
    assert [p.id for p in projects] == ["project-1", "project-2"]
    assert len(session.requests) == 2


def test_absent_header_keeps_the_short_page_assumption_and_says_so() -> None:
    client, session, projects = _scan(
        FakeResponse(200, [_project(1), _project(2)]),
        FakeResponse(200, [_project(3)]),
    )
    assert len(projects) == 3
    assert len(session.requests) == 2
    assert client.pagination_reads[0]["completion"] == "short_page_assumed_last"


@pytest.mark.parametrize("value", ["", "yes-please", "1", "null", "true, false"])
def test_malformed_last_page_fails_closed(value: str) -> None:
    with pytest.raises(ClockifyPaginationError) as caught:
        _scan(FakeResponse(200, [_project(1)], _last(value)))
    failure = caught.value.failure
    assert failure.reason == "invalid_last_page"
    assert failure.last_page == "invalid"
    assert "yes-please" not in json.dumps(caught.value.detail())


def test_empty_page_with_last_page_false_fails_closed() -> None:
    with pytest.raises(ClockifyPaginationError) as caught:
        _scan(
            FakeResponse(200, [_project(1)], _last("false")),
            FakeResponse(200, [], _last("false")),
        )
    failure = caught.value.failure
    assert failure.reason == "empty_page_not_last"
    assert (failure.page, failure.records_seen, failure.last_page) == (2, 1, "false")


def test_empty_page_with_last_page_true_or_absent_ends() -> None:
    client, _, projects = _scan(FakeResponse(200, [], _last("true")))
    assert projects == []
    client, _, projects = _scan(FakeResponse(200, []))
    assert projects == []
    assert client.pagination_reads[0]["completion"] == "short_page_assumed_last"


def test_last_page_false_still_hits_the_page_bound() -> None:
    with pytest.raises(ClockifyPaginationError) as caught:
        _scan(
            FakeResponse(200, [_project(1)], _last("false")),
            FakeResponse(200, [_project(2)], _last("false")),
            max_pages=2,
        )
    assert caught.value.failure.reason == "page_bound"


def test_last_page_false_still_reports_duplicates_without_deduplicating() -> None:
    with pytest.raises(ClockifyPaginationError) as caught:
        _scan(
            FakeResponse(200, [_project(1)], _last("false")),
            FakeResponse(200, [_project(1)], _last("false")),
        )
    assert caught.value.failure.reason == "repeated_page"
    with pytest.raises(ClockifyPaginationError) as caught:
        _scan(
            FakeResponse(200, [_project(1)], _last("false")),
            FakeResponse(200, [_project(2), _project(1)], _last("true")),
        )
    assert caught.value.failure.reason == "repeated_across_pages"
    assert caught.value.failure.duplicate_count == 1


def test_non_paginated_callers_ignore_headers_and_keep_their_contract() -> None:
    session = FakeSession(FakeResponse(200, _project(1), _last("false")))
    client = ClockifyClient(api_key="secret-key", session=session)
    assert client.get_project("workspace-1", "project-1").id == "project-1"
