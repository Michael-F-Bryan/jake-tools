"""Quiet Codex subscription-window alerts for Hermes cron."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .http import HttpSession

USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
THRESHOLDS = (5, 10, 20)
REARM_REMAINING = 30
PERTH = ZoneInfo("Australia/Perth")


class CodexUsageError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodexUsageWindow:
    label: str
    used_percent: float
    reset_at: datetime | None


@dataclass(frozen=True)
class CodexUsageSnapshot:
    plan: str | None
    windows: tuple[CodexUsageWindow, ...]


@dataclass(frozen=True)
class WindowAlertState:
    reset_at: str | None
    bucket: int | None


class CodexUsageClient:
    def __init__(
        self,
        *,
        auth_path: Path,
        session: HttpSession | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._auth_path = auth_path
        self._session = session or requests.Session()
        self._now = now or (lambda: datetime.now(UTC))

    def fetch(self) -> CodexUsageSnapshot:
        token, account_id = _credentials(self._auth_path, self._now())
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "jake-tools",
            "ChatGPT-Account-Id": account_id,
        }
        try:
            response = self._session.request(
                "GET", USAGE_URL, headers=headers, timeout=15
            )
        except requests.RequestException as exc:
            raise CodexUsageError(f"Codex usage request failed: {exc}") from exc

        status = int(getattr(response, "status_code", 0))
        if not 200 <= status < 300:
            hint = (
                "; reauthenticate with `hermes login --provider openai-codex`"
                if status in {401, 403}
                else ""
            )
            raise CodexUsageError(f"Codex usage request returned HTTP {status}{hint}")
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise CodexUsageError("Codex usage response was not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise CodexUsageError("Codex usage response was not a JSON object")

        rate_limit = payload.get("rate_limit")
        if not isinstance(rate_limit, Mapping):
            raise CodexUsageError("Codex usage response did not contain rate limits")
        windows = tuple(
            window
            for key, label in (
                ("primary_window", "Session"),
                ("secondary_window", "Weekly"),
            )
            if (window := _parse_window(rate_limit.get(key), label)) is not None
        )
        if not windows:
            raise CodexUsageError("Codex usage response contained no usable windows")

        raw_plan = payload.get("plan_type")
        plan = (
            str(raw_plan).strip().replace("-", " ").replace("_", " ").title()
            if raw_plan
            else None
        )
        return CodexUsageSnapshot(plan=plan or None, windows=windows)


def run_codex_usage_alert(
    *,
    client: CodexUsageClient,
    state_path: Path,
    timezone: ZoneInfo = PERTH,
) -> list[str]:
    snapshot = client.fetch()
    previous_state = _load_state(state_path)
    next_state: dict[str, WindowAlertState] = {}
    alerts: list[str] = []

    for window in snapshot.windows:
        used = max(0, min(100, round(window.used_percent)))
        remaining = 100 - used
        bucket = _threshold_bucket(remaining)
        reset_at = window.reset_at.isoformat() if window.reset_at else None
        state_key = window.label.lower()
        previous = previous_state.get(state_key)
        if previous is None and state_key == "weekly":
            previous = previous_state.get("session")
        previous_bucket = previous.bucket if previous else None
        # Rolling windows can drift by seconds or briefly recover. Rearm only after
        # enough quota returns to make a later low-water crossing meaningful.
        if previous_bucket is not None and remaining <= REARM_REMAINING:
            recorded_bucket = (
                previous_bucket
                if bucket is None or bucket > previous_bucket
                else bucket
            )
        else:
            previous_bucket = None
            recorded_bucket = bucket
        next_state[state_key] = WindowAlertState(reset_at, recorded_bucket)

        crossed = bucket is not None and (
            previous_bucket is None or bucket < previous_bucket
        )
        if not crossed:
            continue
        reset = (
            f", resets {window.reset_at.astimezone(timezone):%Y-%m-%d %H:%M %Z}"
            if window.reset_at
            else ""
        )
        alerts.append(f"- {window.label}: {remaining}% remaining ({used}% used){reset}")

    _save_state(state_path, next_state)
    if not alerts:
        return []
    plan = f" ({snapshot.plan})" if snapshot.plan else ""
    return [f"Codex usage is getting low{plan}:", *alerts]


def _credentials(auth_path: Path, now: datetime) -> tuple[str, str]:
    try:
        raw = json.loads(auth_path.read_text())
        token = str(raw["providers"]["openai-codex"]["tokens"]["access_token"]).strip()
    except FileNotFoundError as exc:
        raise CodexUsageError(f"Hermes auth file not found: {auth_path}") from exc
    except (KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        raise CodexUsageError(
            f"Hermes auth file has no usable OpenAI Codex token: {auth_path}"
        ) from exc
    if not token:
        raise CodexUsageError(
            "No OpenAI Codex token found; run `hermes login --provider openai-codex`"
        )

    claims = _jwt_claims(token)
    expires_at = claims.get("exp")
    if isinstance(expires_at, int | float) and now.timestamp() >= float(expires_at):
        raise CodexUsageError(
            "The OpenAI Codex token has expired; use Hermes once or run "
            "`hermes login --provider openai-codex`"
        )
    auth_claim = claims.get("https://api.openai.com/auth")
    account_id = (
        str(auth_claim.get("chatgpt_account_id") or "").strip()
        if isinstance(auth_claim, Mapping)
        else ""
    )
    if not account_id:
        raise CodexUsageError("The OpenAI Codex token has no ChatGPT account ID")
    return token, account_id


def _jwt_claims(token: str) -> Mapping[str, Any]:
    try:
        encoded = token.split(".")[1]
        encoded += "=" * (-len(encoded) % 4)
        claims = json.loads(base64.urlsafe_b64decode(encoded))
    except IndexError, ValueError, json.JSONDecodeError:
        return {}
    return claims if isinstance(claims, Mapping) else {}


def _parse_window(value: Any, label: str) -> CodexUsageWindow | None:
    if not isinstance(value, Mapping):
        return None
    used = value.get("used_percent")
    if isinstance(used, bool) or not isinstance(used, int | float):
        return None
    return CodexUsageWindow(
        _window_label(value.get("limit_window_seconds"), label),
        float(used),
        _parse_datetime(value.get("reset_at")),
    )


def _window_label(value: Any, fallback: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    if 4 * 60 * 60 <= value <= 6 * 60 * 60:
        return "Session"
    if 6 * 24 * 60 * 60 <= value <= 8 * 24 * 60 * 60:
        return "Weekly"
    return fallback


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            pass
    return None


def _threshold_bucket(remaining: int) -> int | None:
    return next((limit for limit in THRESHOLDS if remaining <= limit), None)


def _load_state(path: Path) -> dict[str, WindowAlertState]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError, OSError, json.JSONDecodeError:
        return {}
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): WindowAlertState(
            str(value.get("reset_at")) if value.get("reset_at") else None,
            int(value["bucket"])
            if isinstance(value.get("bucket"), int)
            and not isinstance(value.get("bucket"), bool)
            else None,
        )
        for key, value in raw.items()
        if isinstance(value, Mapping)
    }


def _save_state(path: Path, state: Mapping[str, WindowAlertState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: {"reset_at": value.reset_at, "bucket": value.bucket}
        for key, value in sorted(state.items())
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
