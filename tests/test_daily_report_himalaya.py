from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from jake_tools.daily_report.himalaya import Command, CommandResult, run_himalaya_preflight


@dataclass
class FakeRunner:
    responses: dict[Command, CommandResult]
    calls: list[Command] = field(default_factory=list)

    def __call__(self, command: Command) -> CommandResult:
        self.calls.append(command)
        return self.responses.get(
            command,
            CommandResult(command, 99, stderr=f"unexpected command: {command!r}"),
        )


def result(command: Command, returncode: int = 0, stdout: object = "", stderr: str = "") -> CommandResult:
    if not isinstance(stdout, str):
        stdout = json.dumps(stdout)
    return CommandResult(command, returncode, stdout=stdout, stderr=stderr)


COMMAND_V = ("command", "-v", "himalaya")
ACCOUNT_LIST = ("himalaya", "account", "list", "--output", "json")
FOLDER_LIST = ("himalaya", "folder", "list", "-a", "work", "--output", "json")
INBOX_ENVELOPES = (
    "himalaya",
    "envelope",
    "list",
    "-a",
    "work",
    "--page-size",
    "5",
    "--output",
    "json",
)
SENT_ENVELOPES = (
    "himalaya",
    "envelope",
    "list",
    "-a",
    "work",
    "--folder",
    "Sent Items",
    "--page-size",
    "5",
    "--output",
    "json",
)


def test_missing_command_records_evidence_and_stops() -> None:
    runner = FakeRunner({COMMAND_V: result(COMMAND_V, 1, stderr="not found")})

    preflight = run_himalaya_preflight(page_size=5, runner=runner)

    assert not preflight.ok
    assert not preflight.command_found
    assert preflight.errors == ["himalaya command not found"]
    assert runner.calls == [COMMAND_V]
    assert [command.command for command in preflight.commands] == [list(COMMAND_V)]


def test_account_list_failure_records_evidence_and_stops() -> None:
    runner = FakeRunner(
        {
            COMMAND_V: result(COMMAND_V, stdout="/opt/homebrew/bin/himalaya\n"),
            ACCOUNT_LIST: result(ACCOUNT_LIST, 2, stderr="not configured"),
        }
    )

    preflight = run_himalaya_preflight(page_size=5, runner=runner)

    assert not preflight.ok
    assert preflight.command_found
    assert preflight.accounts is None
    assert preflight.errors == ["himalaya account list failed"]
    assert runner.calls == [COMMAND_V, ACCOUNT_LIST]
    assert preflight.commands[-1].stderr == "not configured"


def test_malformed_account_json_records_error() -> None:
    runner = FakeRunner(
        {
            COMMAND_V: result(COMMAND_V, stdout="/usr/bin/himalaya\n"),
            ACCOUNT_LIST: result(ACCOUNT_LIST, stdout="{not json"),
        }
    )

    preflight = run_himalaya_preflight(page_size=5, runner=runner)

    assert not preflight.ok
    assert preflight.accounts is None
    assert preflight.account_names == []
    assert len(preflight.errors) == 1
    assert preflight.errors[0].startswith("himalaya account list returned malformed JSON:")
    assert runner.calls == [COMMAND_V, ACCOUNT_LIST]


def test_happy_path_lists_accounts_folders_inbox_and_sent_envelopes() -> None:
    accounts = [{"name": "work", "email": "michael@example.com"}]
    folders = [{"name": "Inbox"}, {"name": "Sent Items"}]
    inbox = [{"id": "1", "subject": "Hello", "from": "a@example.com"}]
    sent = [{"id": "2", "subject": "Sent", "to": ["b@example.com"]}]
    runner = FakeRunner(
        {
            COMMAND_V: result(COMMAND_V, stdout="/opt/homebrew/bin/himalaya\n"),
            ACCOUNT_LIST: result(ACCOUNT_LIST, stdout=accounts),
            FOLDER_LIST: result(FOLDER_LIST, stdout=folders),
            INBOX_ENVELOPES: result(INBOX_ENVELOPES, stdout=inbox),
            SENT_ENVELOPES: result(SENT_ENVELOPES, stdout=sent),
        }
    )

    preflight = run_himalaya_preflight(page_size=5, runner=runner)

    assert preflight.ok
    assert preflight.account_names == ["work"]
    assert preflight.accounts == accounts
    assert len(preflight.account_preflights) == 1
    account = preflight.account_preflights[0]
    assert account.account == "work"
    assert account.folders == folders
    assert account.inbox_envelopes == inbox
    assert account.sent_envelopes == sent
    assert runner.calls == [COMMAND_V, ACCOUNT_LIST, FOLDER_LIST, INBOX_ENVELOPES, SENT_ENVELOPES]


def test_partial_folder_and_envelope_failures_are_kept_as_evidence() -> None:
    runner = FakeRunner(
        {
            COMMAND_V: result(COMMAND_V, stdout="/opt/homebrew/bin/himalaya\n"),
            ACCOUNT_LIST: result(ACCOUNT_LIST, stdout=[{"name": "work"}]),
            FOLDER_LIST: result(FOLDER_LIST, 1, stderr="folder error"),
            INBOX_ENVELOPES: result(INBOX_ENVELOPES, stdout=[{"id": "1"}]),
            SENT_ENVELOPES: result(SENT_ENVELOPES, 1, stderr="sent error"),
        }
    )

    preflight = run_himalaya_preflight(page_size=5, runner=runner)

    assert not preflight.ok
    assert preflight.errors == []
    account = preflight.account_preflights[0]
    assert account.folders is None
    assert account.inbox_envelopes == [{"id": "1"}]
    assert account.sent_envelopes is None
    assert account.errors == ["folder list failed", "sent envelope list failed"]
    assert [command.stderr for command in preflight.commands[-3:]] == [
        "folder error",
        "",
        "sent error",
    ]


def test_forbidden_commands_are_not_emitted() -> None:
    runner = FakeRunner(
        {
            COMMAND_V: result(COMMAND_V, stdout="/opt/homebrew/bin/himalaya\n"),
            ACCOUNT_LIST: result(ACCOUNT_LIST, stdout=[{"name": "work"}]),
            FOLDER_LIST: result(FOLDER_LIST, stdout=[]),
            INBOX_ENVELOPES: result(INBOX_ENVELOPES, stdout=[]),
            SENT_ENVELOPES: result(SENT_ENVELOPES, stdout=[]),
        }
    )

    preflight = run_himalaya_preflight(page_size=5, runner=runner)

    assert preflight.ok
    emitted = runner.calls
    assert ("himalaya", "message", "read", "-a", "work") not in emitted
    assert all("message" not in command for command in emitted)
    assert all("send" not in command for command in emitted)
    assert all("delete" not in command for command in emitted)
    assert all("move" not in command for command in emitted)
    assert all(command[:7] != ("himalaya", "envelope", "list", "-a", "work", "--folder", "Inbox") for command in emitted)
    assert emitted == [COMMAND_V, ACCOUNT_LIST, FOLDER_LIST, INBOX_ENVELOPES, SENT_ENVELOPES]


def test_rejects_invalid_page_size_before_emitting_commands() -> None:
    runner = FakeRunner({})

    with pytest.raises(ValueError, match="page_size"):
        run_himalaya_preflight(page_size=0, runner=runner)

    assert runner.calls == []
