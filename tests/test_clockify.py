from __future__ import annotations

import json
from typing import Any

from jake_tools.clockify import ClockifyClient, ClockifyUser


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self.text = json.dumps(payload)
        self.content = self.text.encode()
        self._payload = payload
        self.reason = "OK"

    def json(self) -> dict[str, object]:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return self.response


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
