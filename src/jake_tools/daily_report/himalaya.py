from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

Command = tuple[str, ...]
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


@dataclass(frozen=True)
class CommandResult:
    args: Command
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Command], CommandResult]


@dataclass(frozen=True)
class CommandEvidence:
    command: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None

    def to_json(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class AccountPreflight:
    account: str
    folders: JsonValue | None = None
    inbox_envelopes: JsonValue | None = None
    sent_envelopes: JsonValue | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "folders": self.folders,
            "inbox_envelopes": self.inbox_envelopes,
            "sent_envelopes": self.sent_envelopes,
            "errors": self.errors,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class HimalayaPreflight:
    command_found: bool
    accounts: JsonValue | None
    account_names: list[str]
    account_preflights: list[AccountPreflight]
    commands: list[CommandEvidence]
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.command_found and not self.errors and all(
            account.ok for account in self.account_preflights
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "command_found": self.command_found,
            "accounts": self.accounts,
            "account_names": self.account_names,
            "account_preflights": [account.to_json() for account in self.account_preflights],
            "commands": [command.to_json() for command in self.commands],
            "errors": self.errors,
            "ok": self.ok,
        }


_ALLOWED_STATIC_COMMANDS: frozenset[Command] = frozenset(
    {
        ("command", "-v", "himalaya"),
        ("himalaya", "account", "list", "--output", "json"),
    }
)


def default_runner(command: Command) -> CommandResult:
    _ensure_allowed(command)
    if command == ("command", "-v", "himalaya"):
        completed = subprocess.run(
            "command -v himalaya",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    return CommandResult(
        args=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_himalaya_preflight(
    *,
    page_size: int = 25,
    runner: Runner = default_runner,
) -> HimalayaPreflight:
    if page_size < 1:
        raise ValueError("page_size must be at least 1")

    commands: list[CommandEvidence] = []
    errors: list[str] = []

    command_found = _run(("command", "-v", "himalaya"), runner, commands).ok
    if not command_found:
        errors.append("himalaya command not found")
        return HimalayaPreflight(
            command_found=False,
            accounts=None,
            account_names=[],
            account_preflights=[],
            commands=commands,
            errors=errors,
        )

    accounts_evidence = _run(
        ("himalaya", "account", "list", "--output", "json"), runner, commands
    )
    if not accounts_evidence.ok:
        errors.append("himalaya account list failed")
        return HimalayaPreflight(
            command_found=True,
            accounts=None,
            account_names=[],
            account_preflights=[],
            commands=commands,
            errors=errors,
        )

    accounts, parse_error = _parse_json(accounts_evidence.stdout)
    if parse_error is not None:
        errors.append(f"himalaya account list returned malformed JSON: {parse_error}")
        return HimalayaPreflight(
            command_found=True,
            accounts=None,
            account_names=[],
            account_preflights=[],
            commands=commands,
            errors=errors,
        )

    account_names = _account_names(accounts)
    if not account_names:
        errors.append("himalaya account list returned no account names")

    account_preflights = [
        _run_account_preflight(account, page_size, runner, commands)
        for account in account_names
    ]

    return HimalayaPreflight(
        command_found=True,
        accounts=accounts,
        account_names=account_names,
        account_preflights=account_preflights,
        commands=commands,
        errors=errors,
    )


def _run_account_preflight(
    account: str,
    page_size: int,
    runner: Runner,
    commands: list[CommandEvidence],
) -> AccountPreflight:
    errors: list[str] = []
    folders: JsonValue | None = None
    inbox_envelopes: JsonValue | None = None
    sent_envelopes: JsonValue | None = None

    folder_evidence = _run(
        ("himalaya", "folder", "list", "-a", account, "--output", "json"),
        runner,
        commands,
    )
    if folder_evidence.ok:
        folders, parse_error = _parse_json(folder_evidence.stdout)
        if parse_error is not None:
            errors.append(f"folder list returned malformed JSON: {parse_error}")
    else:
        errors.append("folder list failed")

    inbox_evidence = _run(
        (
            "himalaya",
            "envelope",
            "list",
            "-a",
            account,
            "--page-size",
            str(page_size),
            "--output",
            "json",
        ),
        runner,
        commands,
    )
    if inbox_evidence.ok:
        inbox_envelopes, parse_error = _parse_json(inbox_evidence.stdout)
        if parse_error is not None:
            errors.append(f"inbox envelope list returned malformed JSON: {parse_error}")
    else:
        errors.append("inbox envelope list failed")

    sent_evidence = _run(
        (
            "himalaya",
            "envelope",
            "list",
            "-a",
            account,
            "--folder",
            "Sent Items",
            "--page-size",
            str(page_size),
            "--output",
            "json",
        ),
        runner,
        commands,
    )
    if sent_evidence.ok:
        sent_envelopes, parse_error = _parse_json(sent_evidence.stdout)
        if parse_error is not None:
            errors.append(f"sent envelope list returned malformed JSON: {parse_error}")
    else:
        errors.append("sent envelope list failed")

    return AccountPreflight(
        account=account,
        folders=folders,
        inbox_envelopes=inbox_envelopes,
        sent_envelopes=sent_envelopes,
        errors=errors,
    )


def _run(command: Command, runner: Runner, commands: list[CommandEvidence]) -> CommandEvidence:
    _ensure_allowed(command)
    try:
        result = runner(command)
    except Exception as error:  # pragma: no cover - defensive evidence path
        evidence = CommandEvidence(
            command=list(command),
            returncode=None,
            error=f"{type(error).__name__}: {error}",
        )
    else:
        evidence = CommandEvidence(
            command=list(command),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    commands.append(evidence)
    return evidence


def _ensure_allowed(command: Command) -> None:
    if command in _ALLOWED_STATIC_COMMANDS:
        return
    if _matches_folder_list(command):
        return
    if _matches_inbox_envelope_list(command):
        return
    if _matches_sent_envelope_list(command):
        return
    raise ValueError(f"Forbidden Himalaya command: {list(command)!r}")


def _matches_folder_list(command: Command) -> bool:
    return (
        len(command) == 7
        and command[:5] == ("himalaya", "folder", "list", "-a", command[4])
        and command[5:] == ("--output", "json")
        and command[4] != ""
    )


def _matches_inbox_envelope_list(command: Command) -> bool:
    return (
        len(command) == 9
        and command[:5] == ("himalaya", "envelope", "list", "-a", command[4])
        and command[5] == "--page-size"
        and _is_positive_int(command[6])
        and command[7:] == ("--output", "json")
        and command[4] != ""
    )


def _matches_sent_envelope_list(command: Command) -> bool:
    return (
        len(command) == 11
        and command[:5] == ("himalaya", "envelope", "list", "-a", command[4])
        and command[5:7] == ("--folder", "Sent Items")
        and command[7] == "--page-size"
        and _is_positive_int(command[8])
        and command[9:] == ("--output", "json")
        and command[4] != ""
    )


def _is_positive_int(value: str) -> bool:
    try:
        return int(value) > 0
    except ValueError:
        return False


def _parse_json(payload: str) -> tuple[JsonValue | None, str | None]:
    try:
        parsed: JsonValue = json.loads(payload)
    except json.JSONDecodeError as error:
        return None, str(error)
    return parsed, None


def _account_names(accounts: JsonValue) -> list[str]:
    if isinstance(accounts, list):
        names = [_account_name(account) for account in accounts]
        return [name for name in names if name is not None]
    if isinstance(accounts, dict):
        if "accounts" in accounts:
            return _account_names(accounts["accounts"])
        name = _account_name(accounts)
        return [] if name is None else [name]
    return []


def _account_name(account: object) -> str | None:
    if isinstance(account, str) and account:
        return account
    if not isinstance(account, dict):
        return None
    for key in ("name", "account", "id", "email"):
        value = account.get(key)
        if isinstance(value, str) and value:
            return value
    return None
