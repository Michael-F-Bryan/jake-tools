from __future__ import annotations

import json
from typing import Any

import pytest

from jake_tools.newsletters import NewsletterClient, NewsletterError


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


class NonJsonResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.reason = "OK"
        self.text = text
        self.content = text.encode()

    def json(self) -> dict[str, object]:
        raise ValueError("invalid JSON")


class NonMappingResponse:
    def __init__(self, payload: list[object]) -> None:
        self.status_code = 200
        self.reason = "OK"
        self.text = json.dumps(payload)
        self.content = self.text.encode()
        self._payload = payload

    def json(self) -> list[object]:
        return self._payload


class FakeSession:
    def __init__(self, *responses: object) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = list(responses) or None

    def request(self, method: str, url: str, **kwargs: Any) -> object:
        self.requests.append({"method": method, "url": url, **kwargs})
        if self._responses is not None:
            return self._responses.pop(0)
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


def test_newsletter_client_wraps_invalid_json_200_response_with_request_context() -> (
    None
):
    session = FakeSession(NonJsonResponse(200, "not-json"))
    client = NewsletterClient(
        token_provider=lambda resource: "token",
        session=session,
    )

    with pytest.raises(NewsletterError) as raised:
        client.get_item("295")

    assert "invalid JSON" in str(raised.value)
    assert "GET" in str(raised.value)
    assert "not-json" in str(raised.value)


def test_newsletter_client_rejects_non_mapping_json_payload() -> None:
    session = FakeSession(NonMappingResponse(["not", "a", "mapping"]))
    client = NewsletterClient(
        token_provider=lambda resource: "token",
        session=session,
    )

    with pytest.raises(NewsletterError, match="unexpected payload"):
        client.get_item("295")
