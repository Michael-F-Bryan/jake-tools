from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

Command = tuple[str, ...]
JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class HimalayaCommandKind(StrEnum):
    CHECK_INSTALLED = "check_installed"
    ACCOUNT_LIST = "account_list"
    FOLDER_LIST = "folder_list"
    INBOX_ENVELOPE_LIST = "inbox_envelope_list"
    SENT_ENVELOPE_LIST = "sent_envelope_list"


@dataclass(frozen=True)
class HimalayaCommand:
    kind: HimalayaCommandKind
    account: str = ""
    page_size: int = 0

    def __post_init__(self) -> None:
        if (
            self.kind
            in {
                HimalayaCommandKind.FOLDER_LIST,
                HimalayaCommandKind.INBOX_ENVELOPE_LIST,
                HimalayaCommandKind.SENT_ENVELOPE_LIST,
            }
            and not self.account
        ):
            raise ValueError("account must be non-empty")
        if (
            self.kind
            in {
                HimalayaCommandKind.INBOX_ENVELOPE_LIST,
                HimalayaCommandKind.SENT_ENVELOPE_LIST,
            }
            and self.page_size < 1
        ):
            raise ValueError("page_size must be at least 1")

    @classmethod
    def check_installed(cls) -> HimalayaCommand:
        return cls(HimalayaCommandKind.CHECK_INSTALLED)

    @classmethod
    def account_list(cls) -> HimalayaCommand:
        return cls(HimalayaCommandKind.ACCOUNT_LIST)

    @classmethod
    def folder_list(cls, account: str) -> HimalayaCommand:
        return cls(HimalayaCommandKind.FOLDER_LIST, account=account)

    @classmethod
    def inbox_envelopes(cls, account: str, page_size: int) -> HimalayaCommand:
        return cls(
            HimalayaCommandKind.INBOX_ENVELOPE_LIST,
            account=account,
            page_size=page_size,
        )

    @classmethod
    def sent_envelopes(cls, account: str, page_size: int) -> HimalayaCommand:
        return cls(
            HimalayaCommandKind.SENT_ENVELOPE_LIST, account=account, page_size=page_size
        )

    def build(self) -> Command:
        match self.kind:
            case HimalayaCommandKind.CHECK_INSTALLED:
                return ("command", "-v", "himalaya")
            case HimalayaCommandKind.ACCOUNT_LIST:
                return ("himalaya", "account", "list", "--output", "json")
            case HimalayaCommandKind.FOLDER_LIST:
                return (
                    "himalaya",
                    "folder",
                    "list",
                    "-a",
                    self.account,
                    "--output",
                    "json",
                )
            case HimalayaCommandKind.INBOX_ENVELOPE_LIST:
                return (
                    "himalaya",
                    "envelope",
                    "list",
                    "-a",
                    self.account,
                    "--page-size",
                    str(self.page_size),
                    "--output",
                    "json",
                )
            case HimalayaCommandKind.SENT_ENVELOPE_LIST:
                return (
                    "himalaya",
                    "envelope",
                    "list",
                    "-a",
                    self.account,
                    "--folder",
                    "Sent Items",
                    "--page-size",
                    str(self.page_size),
                    "--output",
                    "json",
                )

    @classmethod
    def parse(cls, command: Command) -> HimalayaCommand:
        for candidate in (
            cls.check_installed(),
            cls.account_list(),
        ):
            if command == candidate.build():
                return candidate

        if len(command) == 7 and command[:4] == ("himalaya", "folder", "list", "-a"):
            account = command[4]
            if command[5:] == ("--output", "json"):
                try:
                    return cls.folder_list(account)
                except ValueError:
                    pass

        if len(command) == 9 and command[:5] == (
            "himalaya",
            "envelope",
            "list",
            "-a",
            command[4],
        ):
            account = command[4]
            if command[5] == "--page-size" and command[7:] == ("--output", "json"):
                try:
                    page_size = int(command[6])
                except ValueError:
                    pass
                else:
                    try:
                        parsed = cls.inbox_envelopes(account, page_size)
                    except ValueError:
                        pass
                    else:
                        if parsed.build() == command:
                            return parsed

        if len(command) == 11 and command[:5] == (
            "himalaya",
            "envelope",
            "list",
            "-a",
            command[4],
        ):
            account = command[4]
            if (
                command[5:7] == ("--folder", "Sent Items")
                and command[7] == "--page-size"
            ):
                try:
                    page_size = int(command[8])
                except ValueError:
                    pass
                else:
                    try:
                        parsed = cls.sent_envelopes(account, page_size)
                    except ValueError:
                        pass
                    else:
                        if parsed.build() == command:
                            return parsed

        raise ValueError(f"Forbidden Himalaya command: {list(command)!r}")


@dataclass(frozen=True)
class CommandResult:
    args: Command
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Command], CommandResult]


class CommandEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


class AccountPreflight(BaseModel):
    model_config = ConfigDict(frozen=True)

    account: str
    folders: JsonValue | None = None
    inbox_envelopes: JsonValue | None = None
    sent_envelopes: JsonValue | None = None
    errors: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        return not self.errors


class HimalayaPreflight(BaseModel):
    model_config = ConfigDict(frozen=True)

    command_found: bool
    accounts: JsonValue | None
    account_names: list[str]
    account_preflights: list[AccountPreflight]
    commands: list[CommandEvidence]
    errors: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        return (
            self.command_found
            and not self.errors
            and all(account.ok for account in self.account_preflights)
        )


def default_runner(command: Command) -> CommandResult:
    _ensure_allowed(command)
    if command == HimalayaCommand.check_installed().build():
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

    command_found = _run(HimalayaCommand.check_installed().build(), runner, commands).ok
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

    accounts_evidence = _run(HimalayaCommand.account_list().build(), runner, commands)
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
        HimalayaCommand.folder_list(account).build(),
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
        HimalayaCommand.inbox_envelopes(account, page_size).build(),
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
        HimalayaCommand.sent_envelopes(account, page_size).build(),
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


def _run(
    command: Command, runner: Runner, commands: list[CommandEvidence]
) -> CommandEvidence:
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
    HimalayaCommand.parse(command)


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
