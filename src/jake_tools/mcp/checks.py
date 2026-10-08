"""Deployment checks behind ``python -m jake_tools.mcp doctor``.

Each check returns a typed :class:`Check`; nothing here prints. Details name
the integration, the variable and an HTTP status, and are scrubbed of every
secret value before they leave this module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..clockify import ClockifyClient, ClockifyError
from ..config import Config, XdgPaths
from ..jira import JiraClient, JiraError

CheckStatus = Literal["ok", "warn", "fail", "skip"]

CLAUDE_TIMEOUT_SECONDS = 30.0
REDACTED = "<redacted>"


class Check(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: CheckStatus
    detail: str
    required: bool = False

    @property
    def failed_required(self) -> bool:
        return self.required and self.status == "fail"


class ConfigFileStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    status: Literal["ok", "missing", "error"]
    detail: str = ""


def inspect_config_files(environ: Mapping[str, str]) -> list[ConfigFileStatus]:
    """Parse every candidate config file independently of ``load_config``."""
    results: list[ConfigFileStatus] = []
    for path in XdgPaths.from_environ(environ).config_files:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            results.append(ConfigFileStatus(path=path, status="missing"))
            continue
        except OSError as exc:
            results.append(
                ConfigFileStatus(path=path, status="error", detail=exc.strerror or "")
            )
            continue
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            results.append(ConfigFileStatus(path=path, status="error", detail=str(exc)))
        else:
            results.append(ConfigFileStatus(path=path, status="ok"))
    return results


def check_config_files(files: list[ConfigFileStatus]) -> Check:
    bad = [f for f in files if f.status == "error"]
    if bad:
        detail = "; ".join(f"{f.path}: {f.detail}" for f in bad)
        return Check(name="config", status="fail", detail=detail, required=True)
    parsed = sum(f.status == "ok" for f in files)
    return Check(
        name="config",
        status="ok",
        detail=f"{parsed} config file(s) parsed, {len(files) - parsed} absent",
        required=True,
    )


def check_runs_dir(config: Config) -> Check:
    path = config.runs_dir.value.expanduser().absolute()
    try:
        existed = path.is_dir()
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not existed:
            path.chmod(0o700)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".doctor-") as handle:
            handle.write(b"doctor")
            handle.flush()
    except OSError as exc:
        return Check(
            name="runs_dir",
            status="fail",
            detail=f"{path} is not writable: {exc.strerror or exc}",
            required=True,
        )
    return Check(name="runs_dir", status="ok", detail=f"{path} writable", required=True)


def check_claude_cli(environ: Mapping[str, str]) -> Check:
    """``claude`` on PATH and logged in, via ``claude auth status`` (spends nothing)."""

    def result(status: CheckStatus, detail: str) -> Check:
        return Check(name="claude_cli", status=status, detail=detail, required=True)

    executable = shutil.which("claude", path=environ.get("PATH", ""))
    if executable is None:
        return result("fail", "`claude` not found on PATH")
    env = dict(environ)
    try:
        version = subprocess.run(
            [executable, "--version"],
            env=env,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        ).stdout.strip()
        proc = subprocess.run(
            [executable, "auth", "status", "--json"],
            env=env,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return result("warn", f"`claude` found but could not be run: {exc}")
    label = f"claude {version}" if version else "claude"
    try:
        payload = json.loads(proc.stdout)
        logged_in = payload["loggedIn"]
    except ValueError, KeyError, TypeError:
        return result(
            "warn",
            f"{label} present; `claude auth status` gave no usable answer "
            f"(exit {proc.returncode}); authentication was not verified",
        )
    if logged_in is True:
        method = payload.get("authMethod", "unknown")
        return result("ok", f"{label} authenticated (method: {method})")
    return result(
        "fail", f"{label} is not authenticated under HOME={environ.get('HOME', '')}"
    )


def _secret_values(config: Config) -> list[str]:
    return [s.value for s in config.settings() if s.secret and isinstance(s.value, str)]


def scrub(text: str, secrets: list[str]) -> str:
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


def _first_line(exc: Exception, secrets: list[str]) -> str:
    # The client error's first line is "<service> request failed for GET /x:
    # <status> <reason>"; later lines are the response body, which is dropped.
    message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return scrub(message, secrets)


def check_clockify(config: Config) -> Check:
    key = config.clockify_api_key.value
    if key is None:
        return Check(name="clockify", status="skip", detail="CLOCKIFY_API_KEY is unset")
    secrets = _secret_values(config)
    try:
        client = ClockifyClient(
            api_key=key, base_url=config.clockify_api_base_url.value
        )
        client.get_user()
        workspaces = client.get_workspace_ids()
    except (ClockifyError, ValueError) as exc:
        return Check(
            name="clockify",
            status="fail",
            detail="CLOCKIFY_API_KEY rejected or API unreachable: "
            f"{_first_line(exc, secrets)}",
            required=True,
        )
    allowlist = config.clockify_workspaces.value
    if not allowlist:
        return Check(
            name="clockify",
            status="warn",
            detail=(
                f"authenticated ({len(workspaces)} workspace(s)); "
                "allowlist empty (needed from phase 2)"
            ),
            required=True,
        )
    missing = [w for w in allowlist if w not in workspaces]
    if missing:
        return Check(
            name="clockify",
            status="fail",
            detail="clockify.workspaces entries not visible to this key: "
            + ", ".join(missing),
            required=True,
        )
    return Check(
        name="clockify",
        status="ok",
        detail=f"authenticated; {len(allowlist)} allowlisted workspace(s) found",
        required=True,
    )


def check_jira(config: Config) -> Check:
    missing = config.missing_jira_variables
    if len(missing) == 3:
        return Check(
            name="jira",
            status="skip",
            detail="JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN are all unset",
        )
    if missing:
        return Check(
            name="jira",
            status="fail",
            detail="partially configured; unset: " + ", ".join(missing),
            required=True,
        )
    secrets = _secret_values(config)
    try:
        JiraClient(
            base_url=config.jira_base_url.value or "",
            email=config.jira_email.value or "",
            api_token=config.jira_api_token.value or "",
        ).check_authenticated()
    except (JiraError, ValueError) as exc:
        return Check(
            name="jira",
            status="fail",
            detail="JIRA_* credentials rejected or API unreachable: "
            f"{_first_line(exc, secrets)}",
            required=True,
        )
    return Check(name="jira", status="ok", detail="authenticated", required=True)


def check_session_store(config: Config) -> Check:
    path = config.session_store_path.value
    if path is None:
        return Check(
            name="session_store",
            status="skip",
            detail="path unresolved (set HERMES_HOME or JAKE_TOOLS_SESSION_STORE); "
            "not used until phase 3",
        )
    path = path.expanduser().absolute()
    if path.exists():
        return Check(
            name="session_store",
            status="skip",
            detail=f"{path} exists; not opened until phase 3",
        )
    return Check(
        name="session_store",
        status="warn",
        detail=f"{path} does not exist; not used until phase 3",
    )


def run_checks(
    config: Config, files: list[ConfigFileStatus], environ: Mapping[str, str]
) -> list[Check]:
    return [
        check_config_files(files),
        check_runs_dir(config),
        check_claude_cli(environ),
        check_clockify(config),
        check_jira(config),
        check_session_store(config),
    ]
