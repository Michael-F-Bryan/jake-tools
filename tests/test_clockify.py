from __future__ import annotations

import json
from typing import Any

import pytest

from jake_tools.clockify import (
    ClockifyClient,
    ClockifyClientRecord,
    ClockifyError,
    ClockifyProject,
    ClockifyTask,
    ClockifyUser,
    JiraIssueRef,
    clockify_project_name_for_jira,
    clockify_task_name_for_jira,
)
from jake_tools.jira import JiraError


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
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
