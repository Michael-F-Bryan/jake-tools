"""Deployment checks behind ``python -m jake_tools.mcp doctor``.

Each check returns a typed :class:`Check`; nothing here prints. Details name
the integration, the variable and an HTTP status, and are scrubbed of every
secret value before they leave this module.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping
from http import HTTPStatus
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ConfigDict

from ..clockify import ClockifyClient, ClockifyError
from ..config import Config, XdgPaths
from ..jira import JiraClient, JiraError
from ..transcription import TranscriptionClient, TranscriptionError

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
    mode = path.stat().st_mode & 0o777
    if mode != 0o700:
        return Check(
            name="runs_dir",
            status="warn",
            detail=f"{path} writable but mode is {mode:04o}, expected 0700",
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


_STATUS_PATTERN = re.compile(r"failed for [A-Z]+ \S+: (\d{3})\b")
_MIN_REDACT_LENGTH = 4


def _sensitive_values(config: Config) -> list[str]:
    """Secrets plus the pieces of the Jira base URL that identify a tenant."""
    values: list[str] = [
        s.value for s in config.settings() if s.secret and isinstance(s.value, str)
    ]
    base = config.jira_base_url.value
    if base:
        text = base.strip().rstrip("/")
        parsed = urlsplit(text if "://" in text else f"https://{text}")
        values += [text, parsed.netloc, parsed.hostname or "", parsed.path.strip("/")]
    return values


def scrub(text: str, secrets: list[str]) -> str:
    """Defence in depth: redact sensitive values of meaningful length."""
    candidates = {s for s in secrets if len(s) >= _MIN_REDACT_LENGTH}
    for secret in sorted(candidates, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


def describe_failure(exc: Exception) -> str:
    """A fixed-vocabulary category for a client error; never any exception text."""
    cause = exc.__cause__
    if isinstance(cause, requests.Timeout):
        return "request timed out"
    if isinstance(cause, requests.ConnectionError):
        return "connection failed"
    if isinstance(cause, requests.RequestException):
        return "request failed"
    match = _STATUS_PATTERN.search(str(exc))
    if match:
        code = int(match.group(1))
        try:
            reason = HTTPStatus(code).phrase
        except ValueError:
            reason = "unexpected status"
        return f"HTTP {code} {reason}"
    if isinstance(exc, ValueError):
        return "invalid configuration value"
    return "invalid or unexpected response"


def check_clockify(config: Config) -> Check:
    key = config.clockify_api_key.value
    if key is None:
        return Check(name="clockify", status="skip", detail="CLOCKIFY_API_KEY is unset")
    secrets = _sensitive_values(config)
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
            f"{scrub(describe_failure(exc), secrets)}",
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
    secrets = _sensitive_values(config)
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
            f"{scrub(describe_failure(exc), secrets)}",
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


def check_transcription(config: Config) -> Check:
    """The native transcription service: reachable, ready, and the token works.

    Skipped when no endpoint is configured (the tool is then not exposed).
    With an endpoint, a missing token or an unreachable, unready or
    token-rejecting service is a failure: the tool would be advertised but
    every call would fail.
    """
    base_url = config.transcription_api_base_url.value
    if base_url is None:
        return Check(
            name="transcription",
            status="skip",
            detail="transcription.api_base_url is unset; the transcribe tool is not exposed",
        )
    token = config.transcription_api_token.value
    if not token:
        return Check(
            name="transcription",
            status="fail",
            detail="transcription.api_base_url is set but TRANSCRIPTION_API_TOKEN is unset; "
            "transcribe is exposed and would fail with missing_credentials",
            required=True,
        )
    secrets = _sensitive_values(config)
    try:
        client = TranscriptionClient(
            base_url, token, timeout_seconds=config.transcription_timeout_seconds.value
        )
    except ValueError as exc:
        return Check(
            name="transcription",
            status="fail",
            detail=f"invalid transcription.api_base_url: {scrub(str(exc), secrets)}",
            required=True,
        )
    try:
        health = client.health()
    except TranscriptionError as exc:
        return Check(
            name="transcription",
            status="fail",
            detail=f"service unreachable or unhealthy: {scrub(exc.safe_summary or exc.kind, secrets)}",
            required=True,
        )
    try:
        client.check_token()
    except TranscriptionError as exc:
        if exc.kind == "unauthorized":
            return Check(
                name="transcription",
                status="fail",
                detail="TRANSCRIPTION_API_TOKEN was rejected by the service",
                required=True,
            )
        if exc.kind != "not_ready":
            return Check(
                name="transcription",
                status="fail",
                detail=f"token check failed: {scrub(exc.safe_summary or exc.kind, secrets)}",
                required=True,
            )
    engine = (
        f"{health.engine.name} {health.engine.version}"
        if health.engine
        else "engine unknown"
    )
    if not health.ready:
        return Check(
            name="transcription",
            status="warn",
            detail=f"service reachable but {health.status}"
            + (f" ({scrub(health.detail, secrets)})" if health.detail else "")
            + "; transcribe calls will fail with not_ready until it is",
            required=True,
        )
    return Check(
        name="transcription",
        status="ok",
        detail=f"ready; {engine}; queue {health.queue.queued}/{health.queue.max_queued} waiting",
        required=True,
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
        check_transcription(config),
    ]
