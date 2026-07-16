from __future__ import annotations

import json
import subprocess

import pytest

from jake_tools.clockify_jira_sync import AcliJiraClient, JiraError, JiraIssue


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


def test_acli_jira_client_hydrates_active_assigned_issues() -> None:
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
                    },
                }
            ]
        ),
        completed(
            {
                "key": "SF-427",
                "fields": {
                    "summary": "Evaluate PX4 external control methods",
                    "status": {
                        "name": "In Progress",
                        "statusCategory": {"name": "In Progress"},
                    },
                    "issuetype": {"name": "Task"},
                    "assignee": {"displayName": "Michael Bryan"},
                    "parent": {
                        "key": "SF-131",
                        "fields": {
                            "summary": "Production Vehicle - Investigations and Overhead"
                        },
                    },
                },
            }
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
            assignee="Michael Bryan",
            issueType="Task",
            parentKey="SF-131",
            parentSummary="Production Vehicle - Investigations and Overhead",
        )
    ]
    assert runner.commands[0] == [
        "acli",
        "jira",
        "workitem",
        "search",
        "--jql",
        'project = SF AND assignee = currentUser() AND status in ("In Progress", "Blocked", "In Review") ORDER BY key',
        "--fields",
        "key,summary,status,priority",
        "--json",
    ]
    assert runner.commands[1] == [
        "acli",
        "jira",
        "workitem",
        "view",
        "SF-427",
        "--fields",
        "*all",
        "--json",
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
    assert issues[1].assignee is None
    assert runner.commands[0][5] == "key in (SF-304,SF-438) ORDER BY key"


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

    with pytest.raises(JiraError, match="not authenticated"):
        client.get_issues(["SF-304"])


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

    with pytest.raises(JiraError, match="invalid JSON"):
        client.get_issues(["SF-304"])
