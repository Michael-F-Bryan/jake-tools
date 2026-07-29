from __future__ import annotations

import json
import subprocess

import pytest

from jake_tools.jira import AcliJiraClient, JiraError, JiraIssue, normalise_jira_key


def test_normalise_jira_key_upcases_and_strips() -> None:
    assert normalise_jira_key(" sf-353 ") == "SF-353"


def test_normalise_jira_key_rejects_malformed_keys() -> None:
    with pytest.raises(JiraError, match="Invalid Jira issue key"):
        normalise_jira_key("not a key")


class FakeCommandRunner:
    def __init__(self, *results: subprocess.CompletedProcess[str]) -> None:
        self.results = list(results)
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return self.results.pop(0)


def completed(payload: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(payload),
        stderr="",
    )


def test_acli_jira_client_fetches_active_assigned_issues_in_one_search() -> None:
    # A single search call carries issuetype/parent directly instead of a
    # search followed by a per-result "workitem view" subprocess.
    runner = FakeCommandRunner(
        completed(
            [
                {
                    "key": "SF-427",
                    "fields": {
                        "summary": "Evaluate PX4 external control methods",
                        "status": {
                            "name": "In Progress",
                            "statusCategory": {"name": "In Progress"},
                        },
                        "issuetype": {"name": "Task"},
                        "parent": {
                            "key": "SF-131",
                            "fields": {
                                "summary": "Production Vehicle - Investigations and Overhead"
                            },
                        },
                    },
                },
                {
                    "key": "SF-1",
                    "fields": {
                        "summary": "Simulator work",
                        "status": {
                            "name": "In Progress",
                            "statusCategory": {"name": "In Progress"},
                        },
                        "issuetype": {"name": "Project / Phase"},
                    },
                },
            ]
        ),
    )
    client = AcliJiraClient(runner=runner)

    issues = client.get_active_assigned_issues("SF")

    assert issues == [
        JiraIssue(
            key="SF-427",
            summary="Evaluate PX4 external control methods",
            status="In Progress",
            statusCategory="In Progress",
            issueType="Task",
            parentKey="SF-131",
            parentSummary="Production Vehicle - Investigations and Overhead",
        ),
        JiraIssue(
            key="SF-1",
            summary="Simulator work",
            status="In Progress",
            statusCategory="In Progress",
            issueType="Project / Phase",
        ),
    ]
    assert len(runner.commands) == 1
    assert runner.commands[0] == [
        "acli",
        "jira",
        "workitem",
        "search",
        "--jql",
        'project = SF AND assignee = currentUser() AND status in ("In Progress", "Blocked", "In Review") ORDER BY key',
        "--fields",
        "key,summary,status,issuetype,parent",
        "--json",
        "--paginate",
    ]


def test_acli_jira_client_gets_one_detailed_issue() -> None:
    runner = FakeCommandRunner(
        completed(
            {
                "key": "SF-304",
                "fields": {
                    "summary": "Bench test PX4",
                    "status": {
                        "name": "In Progress",
                        "statusCategory": {"name": "In Progress"},
                    },
                    "issuetype": {"name": "Task"},
                    "assignee": {"displayName": "David Htet"},
                    "parent": {
                        "key": "SF-131",
                        "fields": {
                            "summary": "Production Vehicle - Investigations and Overhead"
                        },
                    },
                },
            }
        )
    )
    client = AcliJiraClient(runner=runner)

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
    assert runner.commands == [
        [
            "acli",
            "jira",
            "workitem",
            "view",
            "SF-304",
            "--fields",
            "*all",
            "--json",
        ]
    ]


def test_acli_jira_client_batches_issue_lookup_by_key() -> None:
    runner = FakeCommandRunner(
        completed(
            [
                {
                    "key": "SF-304",
                    "fields": {
                        "summary": "Bench test PX4",
                        "status": {
                            "name": "Done",
                            "statusCategory": {"name": "Done"},
                        },
                        "assignee": {"displayName": "Michael Bryan"},
                        "issuetype": {"name": "Task"},
                        "parent": {
                            "key": "SF-131",
                            "fields": {"summary": "Production Vehicle"},
                        },
                    },
                },
                {
                    "key": "SF-438",
                    "fields": {
                        "summary": "Check newer flight controller boards",
                        "status": {
                            "name": "To Do",
                            "statusCategory": {"name": "To Do"},
                        },
                        "assignee": None,
                    },
                },
            ]
        )
    )
    client = AcliJiraClient(runner=runner)

    issues = client.get_issues(["SF-438", "SF-304", "SF-438"])

    assert [issue.key for issue in issues] == ["SF-304", "SF-438"]
    assert issues[0].status_category == "Done"
    assert issues[0].issue_type == "Task"
    assert issues[0].parent_key == "SF-131"
    assert issues[1].assignee is None
    assert runner.commands[0][5] == "key in (SF-304,SF-438) ORDER BY key"
    assert runner.commands[0][7] == "key,summary,status,assignee,issuetype,parent"
    assert runner.commands[0][-1] == "--paginate"


def test_acli_jira_client_skips_command_for_empty_issue_lookup() -> None:
    runner = FakeCommandRunner()
    client = AcliJiraClient(runner=runner)

    assert client.get_issues([]) == []
    assert runner.commands == []


def test_acli_jira_client_reports_command_failure() -> None:
    runner = FakeCommandRunner(
        subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="not authenticated",
        )
    )
    client = AcliJiraClient(runner=runner)

    with pytest.raises(JiraError) as raised:
        client.get_issues(["SF-304"])

    assert "not authenticated" in str(raised.value)
    assert "acli jira workitem search" in str(raised.value)


def test_acli_jira_client_reports_malformed_json() -> None:
    runner = FakeCommandRunner(
        subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="not-json",
            stderr="",
        )
    )
    client = AcliJiraClient(runner=runner)

    with pytest.raises(JiraError) as raised:
        client.get_issues(["SF-304"])

    assert "invalid JSON" in str(raised.value)
    assert "acli jira workitem search" in str(raised.value)
    assert "not-json" in str(raised.value)
