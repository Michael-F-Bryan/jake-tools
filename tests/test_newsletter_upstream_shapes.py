"""Wrong-shaped replies and redirects, through the real client over HTTP.

A syntactically valid but malformed upstream reply must stay an *upstream*
failure that keeps whatever evidence the caller already holds (an entry ID,
step effects), and a redirect must never be followed: a 307/308 would replay a
write, and a token POST carries the refresh token.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest
from fake_microsoft import (
    ACCOUNT,
    CLIENT,
    GRAPH_RESOURCE,
    REFRESH_TOKEN,
    TENANT,
    FakeMicrosoft,
)

from jake_tools.microsoft_auth import (
    MicrosoftAuthError,
    MicrosoftSettings,
    MicrosoftTokenProvider,
)
from jake_tools.newsletter_entries import (
    NewsletterWriteError,
    PreparedAttachment,
    create_entry,
    get_entry,
    list_entries,
    update_entry,
)
from jake_tools.newsletters import NewsletterClient, NewsletterError


@pytest.fixture
def fake() -> Iterator[FakeMicrosoft]:
    service = FakeMicrosoft().start()
    try:
        yield service
    finally:
        service.stop()


@pytest.fixture
def client(fake: FakeMicrosoft) -> NewsletterClient:
    provider = MicrosoftTokenProvider(
        MicrosoftSettings(
            tenant_id=TENANT,
            client_id=CLIENT,
            account=ACCOUNT,
            refresh_token=REFRESH_TOKEN,
            authority_url=fake.base_url,
        )
    )
    return NewsletterClient(
        token_provider=provider,
        graph_root=fake.graph_root,
        sharepoint_root=fake.sharepoint_root,
    )


def attachment(name: str, content: bytes) -> PreparedAttachment:
    return PreparedAttachment(name, content, hashlib.sha256(content).hexdigest())


def statuses(error: NewsletterWriteError) -> dict[str, str]:
    return {f"{e.kind}:{e.subject}": e.status for e in error.effects}


# -- malformed replies --------------------------------------------------------


@pytest.mark.parametrize("reply", ["{}", '{"id": 5}', '{"id": null}', '{"id": ["1"]}'])
def test_a_create_accepted_without_a_usable_id_is_uncertain_not_a_crash(
    fake: FakeMicrosoft, client: NewsletterClient, reply: str
) -> None:
    fake.script["POST items"] = ["apply-then-json:" + reply]

    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[attachment("a", b"A")])

    error = caught.value
    assert error.outcome == "create_uncertain" and error.retry_safe is False
    assert error.entry_id is None
    assert statuses(error) == {
        "create_entry:entry": "uncertain",
        "add_attachment:a": "not_attempted",
    }
    assert "unexpected response shape" in (error.effects[0].detail or "")
    assert (
        len(fake.creates()) == 1 and len(fake.entries) == 1
    )  # applied, never repeated


def test_a_malformed_readback_after_the_writes_keeps_the_entry_and_effects(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["GET item"] = ['apply-then-json:{"id": "1", "fields": "nope"}']

    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[attachment("a", b"A")])

    error = caught.value
    assert error.outcome == "verification_unavailable"
    assert error.entry_id == "1" and error.entry_url
    assert statuses(error) == {
        "create_entry:entry": "applied",
        "add_attachment:a": "applied",
    }
    assert all(e.verified is None for e in error.effects)  # nothing was confirmed
    assert fake.entries[1].attachments == {"a": b"A"}


@pytest.mark.parametrize(
    "reply", ['{"value": "x"}', '{"value": [{"FileName": 7}]}', '{"value": [5]}']
)
def test_a_malformed_attachment_inventory_after_the_writes_is_unavailable_not_invalid(
    fake: FakeMicrosoft, client: NewsletterClient, reply: str
) -> None:
    fake.script["GET attachment-list"] = ["apply-then-json:" + reply]

    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[attachment("a", b"A")])

    assert caught.value.outcome == "verification_unavailable"
    assert caught.value.entry_id == "1"
    assert isinstance(caught.value.__cause__, NewsletterError)


def test_a_malformed_inventory_does_not_hide_an_uncertain_attachment(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["POST attachment-add:a"] = ["drop"]
    fake.script["GET attachment-list"] = ['apply-then-json:{"value": "x"}']

    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[attachment("a", b"A")])

    assert caught.value.outcome == "partial" and caught.value.entry_id == "1"
    assert statuses(caught.value)["add_attachment:a"] == "uncertain"  # not settled


def test_update_preread_with_a_malformed_reply_fails_before_any_write(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("T", "<p>B</p>")
    fake.script["GET item"] = ['apply-then-json:{"id": 5}']

    with pytest.raises(NewsletterError, match="unexpected"):
        update_entry(client, str(entry.id), title="N", body=None, attachments=[])

    assert not fake.calls("PATCH", "") and fake.entries[1].title == "T"


def test_malformed_reads_are_upstream_errors_without_payload_text(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("T", "<p>B</p>")
    fake.script["GET items"] = ['apply-then-json:{"value": [{"id": 3}]}']
    fake.script["GET attachment-list"] = ['apply-then-json:{"value": "secret-ish"}']
    with pytest.raises(NewsletterError) as listing:
        list_entries(client, limit=5, cursor=None, include_body=False)
    with pytest.raises(NewsletterError) as inventory:
        get_entry(client, str(entry.id))
    for caught in (listing, inventory):
        assert "unexpected response shape" in (caught.value.safe_summary or "")
        assert "secret-ish" not in str(caught.value)


# -- redirects ----------------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirected_create_is_never_followed_or_replayed(
    fake: FakeMicrosoft, client: NewsletterClient, status: int
) -> None:
    fake.script["POST items"] = [f"redirect:{status}"]

    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[])

    assert caught.value.outcome == "create_uncertain"  # a redirect proves nothing
    assert len(fake.creates()) == 1
    assert not fake.calls("POST", "/redirected") and not fake.calls(
        "GET", "/redirected"
    )
    assert "HTTP" in (caught.value.effects[0].detail or "")


def test_a_redirected_attachment_upload_is_not_followed(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["POST attachment-add:a"] = ["redirect:308"]

    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[attachment("a", b"A")])

    assert caught.value.outcome == "partial"
    assert statuses(caught.value)["add_attachment:a"] == "uncertain"
    assert not fake.calls("POST", "/redirected")
    assert len(fake.calls("POST", "/add(")) == 1
    assert fake.entries[1].attachments == {}


def test_a_redirected_read_is_an_error_not_a_followed_request(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.add_entry("T", "<p>B</p>")
    fake.script["GET item"] = ["redirect:302"]
    with pytest.raises(NewsletterError) as caught:
        client.get_item("1")
    assert caught.value.status == 302
    assert not fake.calls("GET", "/redirected")


@pytest.mark.parametrize("status", [302, 307, 308])
def test_a_redirected_token_request_never_re_sends_the_refresh_token(
    fake: FakeMicrosoft, status: int
) -> None:
    fake.token_redirect = status
    provider = MicrosoftTokenProvider(
        MicrosoftSettings(
            tenant_id=TENANT,
            client_id=CLIENT,
            account=ACCOUNT,
            refresh_token=REFRESH_TOKEN,
            client_secret="s3cret-value",
            authority_url=fake.base_url,
        )
    )
    with pytest.raises(MicrosoftAuthError) as caught:
        provider(GRAPH_RESOURCE)
    assert caught.value.kind == "rejected" and caught.value.status == status
    assert len(fake.token_requests) == 1
    assert not fake.calls("POST", "/redirected") and not fake.calls(
        "GET", "/redirected"
    )
    assert REFRESH_TOKEN not in str(caught.value)


# -- absent fields are not empty results --------------------------------------


def test_a_200_with_no_collection_is_not_an_empty_list_or_inventory(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("T", "<p>B</p>")
    fake.script["GET items"] = ["apply-then-json:{}"]
    fake.script["GET attachment-list"] = ["apply-then-json:{}"]

    with pytest.raises(NewsletterError, match="unexpected list items"):
        list_entries(client, limit=5, cursor=None, include_body=False)
    with pytest.raises(NewsletterError, match="unexpected list attachments"):
        get_entry(client, str(entry.id))


def test_a_200_get_without_an_entry_is_not_a_blank_entry(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("T", "<p>B</p>")
    for reply in ("{}", '{"id": "1"}', '{"fields": {"Title": "x"}}'):
        fake.script["GET item"] = ["apply-then-json:" + reply]
        with pytest.raises(NewsletterError, match="unexpected get item"):
            get_entry(client, str(entry.id))


def test_an_empty_write_readback_is_a_failed_verification_path_not_success(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["GET item"] = ["apply-then-json:{}"]
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[])
    assert caught.value.outcome == "verification_unavailable"
    assert caught.value.entry_id == "1"


def test_legitimately_empty_collections_and_blank_bodies_stay_valid(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    page = list_entries(client, limit=5, cursor=None, include_body=True)
    assert page.entries == [] and page.complete is True

    entry = fake.add_entry("Blank body", "")
    got = get_entry(client, str(entry.id))
    assert got.body == "" and got.attachments == []
