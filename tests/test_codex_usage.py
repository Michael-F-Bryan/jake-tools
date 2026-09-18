from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from jake_tools.codex_usage import (
    CodexUsageClient,
    CodexUsageError,
    run_codex_usage_alert,
)


class FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append((method, url, kwargs))
        return self.responses.pop(0)


def _jwt(*, expires_at: datetime, account_id: str = "account-123") -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    claims = {
        "exp": int(expires_at.timestamp()),
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.signature"


def _write_auth(path: Path, access_token: str) -> None:
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "openai-codex": {
                        "tokens": {"access_token": access_token},
                    }
                }
            }
        )
    )


def _usage_payload(
    *,
    session_used: float,
    weekly_used: float,
    session_reset_at: int = 1_800_000_000,
) -> dict[str, Any]:
    return {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {
                "used_percent": session_used,
                "limit_window_seconds": 18_000,
                "reset_at": session_reset_at,
            },
            "secondary_window": {
                "used_percent": weekly_used,
                "limit_window_seconds": 604_800,
                "reset_at": 1_800_086_400,
            },
        },
    }


def test_client_reads_hermes_auth_and_fetches_codex_usage(tmp_path: Path) -> None:
    now = datetime(2026, 9, 10, tzinfo=UTC)
    auth_path = tmp_path / "auth.json"
    token = _jwt(expires_at=datetime(2026, 9, 20, tzinfo=UTC))
    _write_auth(auth_path, token)
    session = FakeSession(FakeResponse(_usage_payload(session_used=81, weekly_used=4)))

    snapshot = CodexUsageClient(
        auth_path=auth_path,
        session=session,
        now=lambda: now,
    ).fetch()

    assert snapshot.plan == "Plus"
    assert [(window.label, window.used_percent) for window in snapshot.windows] == [
        ("Session", 81.0),
        ("Weekly", 4.0),
    ]
    assert session.requests == [
        (
            "GET",
            "https://chatgpt.com/backend-api/wham/usage",
            {
                "headers": {
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "User-Agent": "jake-tools",
                    "ChatGPT-Account-Id": "account-123",
                },
                "timeout": 15,
            },
        )
    ]


def test_alert_is_quiet_until_each_new_threshold_crossing(tmp_path: Path) -> None:
    now = datetime(2026, 9, 10, tzinfo=UTC)
    auth_path = tmp_path / "auth.json"
    state_path = tmp_path / "state.json"
    _write_auth(auth_path, _jwt(expires_at=datetime(2026, 9, 20, tzinfo=UTC)))
    session = FakeSession(
        FakeResponse(_usage_payload(session_used=81, weekly_used=4)),
        FakeResponse(_usage_payload(session_used=81, weekly_used=4)),
        FakeResponse(_usage_payload(session_used=91, weekly_used=4)),
        FakeResponse(_usage_payload(session_used=96, weekly_used=4)),
    )
    client = CodexUsageClient(auth_path=auth_path, session=session, now=lambda: now)

    first = run_codex_usage_alert(client=client, state_path=state_path)
    duplicate = run_codex_usage_alert(client=client, state_path=state_path)
    ten_percent = run_codex_usage_alert(client=client, state_path=state_path)
    five_percent = run_codex_usage_alert(client=client, state_path=state_path)

    assert first[0] == "Codex usage is getting low (Plus):"
    assert first[1].startswith("- Session: 19% remaining (81% used)")
    assert duplicate == []
    assert ten_percent[1].startswith("- Session: 9% remaining (91% used)")
    assert five_percent[1].startswith("- Session: 4% remaining (96% used)")


def test_alert_does_not_repeat_when_reset_time_drifts_near_low_water(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 10, tzinfo=UTC)
    auth_path = tmp_path / "auth.json"
    state_path = tmp_path / "state.json"
    _write_auth(auth_path, _jwt(expires_at=datetime(2026, 9, 20, tzinfo=UTC)))
    session = FakeSession(
        FakeResponse(
            _usage_payload(
                session_used=87,
                weekly_used=4,
                session_reset_at=1_800_000_000,
            )
        ),
        FakeResponse(
            _usage_payload(
                session_used=88,
                weekly_used=4,
                session_reset_at=1_800_000_017,
            )
        ),
        FakeResponse(
            _usage_payload(
                session_used=79,
                weekly_used=4,
                session_reset_at=1_800_000_030,
            )
        ),
        FakeResponse(
            _usage_payload(
                session_used=88,
                weekly_used=4,
                session_reset_at=1_800_000_047,
            )
        ),
    )
    client = CodexUsageClient(auth_path=auth_path, session=session, now=lambda: now)

    first = run_codex_usage_alert(client=client, state_path=state_path)
    reset_drift = run_codex_usage_alert(client=client, state_path=state_path)
    minor_recovery = run_codex_usage_alert(client=client, state_path=state_path)
    recrossing = run_codex_usage_alert(client=client, state_path=state_path)

    assert first[1].startswith("- Session: 13% remaining (87% used)")
    assert reset_drift == []
    assert minor_recovery == []
    assert recrossing == []


def test_client_labels_single_seven_day_primary_window_as_weekly(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 10, tzinfo=UTC)
    auth_path = tmp_path / "auth.json"
    token = _jwt(expires_at=datetime(2026, 9, 20, tzinfo=UTC))
    _write_auth(auth_path, token)
    payload = _usage_payload(session_used=87, weekly_used=4)
    payload["rate_limit"]["primary_window"]["limit_window_seconds"] = 604_800
    payload["rate_limit"]["secondary_window"] = None
    session = FakeSession(FakeResponse(payload))

    snapshot = CodexUsageClient(
        auth_path=auth_path,
        session=session,
        now=lambda: now,
    ).fetch()

    assert [(window.label, window.used_percent) for window in snapshot.windows] == [
        ("Weekly", 87.0)
    ]


def test_weekly_window_reuses_state_from_the_old_session_label(tmp_path: Path) -> None:
    now = datetime(2026, 9, 10, tzinfo=UTC)
    auth_path = tmp_path / "auth.json"
    state_path = tmp_path / "state.json"
    _write_auth(auth_path, _jwt(expires_at=datetime(2026, 9, 20, tzinfo=UTC)))
    state_path.write_text(
        json.dumps(
            {
                "session": {
                    "bucket": 10,
                    "reset_at": "2026-09-19T12:12:30+00:00",
                }
            }
        )
    )
    payload = _usage_payload(session_used=90, weekly_used=4)
    payload["rate_limit"]["primary_window"]["limit_window_seconds"] = 604_800
    payload["rate_limit"]["secondary_window"] = None
    session = FakeSession(FakeResponse(payload))
    client = CodexUsageClient(auth_path=auth_path, session=session, now=lambda: now)

    alert = run_codex_usage_alert(client=client, state_path=state_path)

    assert alert == []
    assert json.loads(state_path.read_text()) == {
        "weekly": {
            "bucket": 10,
            "reset_at": "2027-01-15T08:00:00+00:00",
        }
    }


def test_client_rejects_an_expired_token_without_attempting_refresh(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 10, tzinfo=UTC)
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path, _jwt(expires_at=datetime(2026, 9, 9, tzinfo=UTC)))
    session = FakeSession()

    with pytest.raises(CodexUsageError, match="expired"):
        CodexUsageClient(
            auth_path=auth_path,
            session=session,
            now=lambda: now,
        ).fetch()

    assert session.requests == []
