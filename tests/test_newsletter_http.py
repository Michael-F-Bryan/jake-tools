from __future__ import annotations

import json
from typing import Any

from jake_tools.newsletters import NewsletterClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object] | None) -> None:
        self.status_code = status_code
        self.reason = "OK"
        self._payload = payload
        self.text = "" if payload is None else json.dumps(payload)
        self.content = self.text.encode()

    def json(self) -> dict[str, object]:
        if self._payload is None:
            raise AssertionError("json should not be called for empty responses")
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.requests.append({"method": method, "url": url, **kwargs})
        return FakeResponse(200, {"value": []})


def test_newsletter_client_uses_requests_session_for_json_requests() -> None:
    session = FakeSession()
    client = NewsletterClient(
        token_provider=lambda resource: f"token-for-{resource}",
        graph_root="https://graph.example.test/v1.0",
        site_id="site id",
        list_id="list-id",
        session=session,
    )

    assert client.list_items(limit=3) == []

    assert len(session.requests) == 1
    request = session.requests[0]
    assert request["method"] == "GET"
    assert request["url"].startswith(
        "https://graph.example.test/v1.0/sites/site%20id/lists/list-id/items?"
    )
    assert request["headers"] == {
        "Authorization": "Bearer token-for-https://graph.microsoft.com",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    assert request["timeout"] == 30
