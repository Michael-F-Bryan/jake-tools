from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jake_tools.newsletters import (
    NewsletterAttachment,
    NewsletterClient,
    NewsletterError,
)


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


def test_newsletter_client_truncates_error_response_body_to_500_chars() -> None:
    long_message = "x" * 1000
    session = FakeSession(FakeResponse(404, {"error": long_message}))
    client = NewsletterClient(
        token_provider=lambda resource: "token",
        session=session,
    )

    with pytest.raises(NewsletterError) as raised:
        client.get_item("295")

    message = str(raised.value)
    assert "404" in message
    assert "GET" in message
    assert long_message not in message
    assert "x" * 100 in message


def test_newsletter_client_create_item_creates_then_fetches() -> None:
    # create_item POSTs the new item, then re-fetches it by id (so the
    # returned NewsletterItem reflects what Graph actually persisted,
    # including any server-assigned fields) rather than trusting the create
    # response alone.
    session = FakeSession(
        FakeResponse(201, {"id": "296"}),
        FakeResponse(
            200,
            {
                "id": "296",
                "webUrl": "https://example.test/296",
                "fields": {
                    "Title": "New item",
                    "Body": "<p>Body text.</p>",
                    "Created": "2026-06-21T01:00:00Z",
                    "Modified": "2026-06-21T01:00:00Z",
                },
            },
        ),
    )
    client = NewsletterClient(token_provider=lambda resource: "token", session=session)

    item = client.create_item(title="New item", body="Body text.")

    assert item.id == "296"
    assert item.title == "New item"
    assert item.body == "Body text."
    assert len(session.requests) == 2
    assert session.requests[0]["method"] == "POST"
    assert session.requests[0]["url"].endswith("/items")
    assert session.requests[0]["json"] == {
        "fields": {"Title": "New item", "Body": "<p>Body text.</p>"}
    }
    assert session.requests[1]["method"] == "GET"
    assert session.requests[1]["url"].endswith("/items/296?$expand=fields")


def test_newsletter_client_update_item_skips_patch_when_no_fields_change() -> None:
    # No title/body change and no attachments means there is nothing to
    # PATCH; update_item should only re-fetch the item, not send an empty
    # (or accidentally destructive) PATCH request.
    session = FakeSession(
        FakeResponse(
            200,
            {
                "id": "295",
                "webUrl": "https://example.test/295",
                "fields": {
                    "Title": "Existing title",
                    "Body": "",
                    "Created": "2026-06-20T10:52:41Z",
                    "Modified": "2026-06-20T10:52:41Z",
                },
            },
        ),
    )
    client = NewsletterClient(token_provider=lambda resource: "token", session=session)

    item = client.update_item("295", title=None, body=None)

    assert item.title == "Existing title"
    assert len(session.requests) == 1
    assert session.requests[0]["method"] == "GET"


def test_newsletter_client_update_item_patches_only_the_given_fields() -> None:
    session = FakeSession(
        FakeResponse(200, {}),
        FakeResponse(
            200,
            {
                "id": "295",
                "webUrl": "https://example.test/295",
                "fields": {"Title": "Updated title"},
            },
        ),
    )
    client = NewsletterClient(token_provider=lambda resource: "token", session=session)

    item = client.update_item("295", title="Updated title", body=None)

    assert item.title == "Updated title"
    assert len(session.requests) == 2
    assert session.requests[0]["method"] == "PATCH"
    assert session.requests[0]["json"] == {"Title": "Updated title"}
    assert session.requests[1]["method"] == "GET"


def test_newsletter_client_add_attachment_escapes_apostrophe_in_filename(
    tmp_path: Path,
) -> None:
    # Document the current escaping: OData doubles an embedded single quote
    # ("'" -> "''") before percent-encoding the whole filename, so a literal
    # apostrophe ends up as %27%27 in the request URL.
    attachment_path = tmp_path / "Mum's Notes.pdf"
    attachment_path.write_bytes(b"fake pdf bytes")
    session = FakeSession(FakeResponse(200, {}))
    client = NewsletterClient(
        token_provider=lambda resource: "token",
        sharepoint_root="https://sharepoint.example.test/_api",
        list_id="list-id",
        session=session,
    )

    client.add_attachment("295", NewsletterAttachment(attachment_path))

    assert len(session.requests) == 1
    request = session.requests[0]
    assert request["method"] == "POST"
    assert request["url"] == (
        "https://sharepoint.example.test/_api/web/lists(guid'list-id')/items(295)"
        "/AttachmentFiles/add(FileName='Mum%27%27s%20Notes.pdf')"
    )
    assert request["data"] == b"fake pdf bytes"
