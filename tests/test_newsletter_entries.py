"""Verified entry workflows over the real client, provider and a fake upstream.

Everything but Microsoft itself is production code: ``NewsletterClient`` over
``requests``, the refresh-aware provider, and the workflows under test.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_microsoft import (
    ACCOUNT,
    CLIENT,
    GRAPH_RESOURCE,
    REFRESH_TOKEN,
    SHAREPOINT_RESOURCE,
    TENANT,
    FakeMicrosoft,
)

from jake_tools.microsoft_auth import MicrosoftSettings, MicrosoftTokenProvider
from jake_tools.newsletter_entries import (
    NewsletterInputError,
    NewsletterWriteError,
    PreparedAttachment,
    create_entry,
    get_entry,
    list_entries,
    prepare_attachments,
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


# -- reads ------------------------------------------------------------------


def test_listing_pages_to_the_end_with_explicit_coverage(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    for number in range(5):
        fake.add_entry(f"Entry {number}", f"<p>body {number}</p>")

    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = list_entries(client, limit=2, cursor=cursor, include_body=False)
        pages += 1
        seen += [e.title for e in page.entries]
        assert page.count == len(page.entries) <= 2
        assert page.complete is (page.next_cursor is None)
        assert all(e.body == "" and e.attachments is None for e in page.entries)
        cursor = page.next_cursor
        if page.complete:
            break
    assert pages == 3
    assert seen == [f"Entry {n}" for n in (4, 3, 2, 1, 0)]  # newest first, no gaps


def test_a_server_side_page_cap_is_reported_not_hidden(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    for number in range(4):
        fake.add_entry(f"E{number}", "<p>x</p>")
    fake.page_size_cap = 1  # asked for 100, the service returns 1 and a next link

    page = list_entries(client, limit=100, cursor=None, include_body=True)

    assert page.count == 1 and page.complete is False and page.next_cursor
    assert page.entries[0].body == "x"


def test_a_next_link_without_a_usable_token_fails_loudly(
    fake: FakeMicrosoft, client: NewsletterClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    for number in range(3):
        fake.add_entry(f"E{number}", "<p>x</p>")
    fake.page_size_cap = 1
    original = fake._list  # noqa: SLF001

    def without_token(query: dict[str, list[str]]) -> dict[str, object]:
        payload = original(query)
        payload["@odata.nextLink"] = "https://example.test/items?$top=1"
        return payload

    monkeypatch.setattr(fake, "_list", without_token)
    with pytest.raises(NewsletterError, match="continuation"):
        list_entries(client, limit=10, cursor=None, include_body=False)


def test_get_returns_content_and_attachment_metadata(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry(
        "Doctrine", "<p>one</p><p>two</p>", attachments={"a b.pdf": b"%PDF"}
    )

    got = get_entry(client, str(entry.id))

    assert (got.id, got.title, got.body) == (str(entry.id), "Doctrine", "one\n\ntwo")
    assert got.url.endswith(f"ID={entry.id}")
    [att] = got.attachments or []
    assert att.name == "a b.pdf"
    assert (
        att.url
        == f"{fake.base_url}/sites/leadershipteam/Lists/Newsletter/Attachments/{entry.id}/a%20b.pdf"
    )
    assert att.size_bytes is None and att.sha256 is None  # not listed upstream


# -- create -----------------------------------------------------------------


def test_create_with_attachments_is_read_back_and_bytes_compared(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    files = [attachment("one.bin", b"\x00\x01" * 50), attachment("it's.txt", b"hi")]

    result = create_entry(
        client, title="Vehicle phones", body="Para 1\n\nPara 2 <b>", attachments=files
    )

    assert result.verified and result.operation == "create"
    assert result.entry.title == "Vehicle phones"
    assert result.entry.body == "Para 1\n\nPara 2 <b>"
    assert [
        (a.name, a.size_bytes, a.sha256) for a in result.entry.attachments or []
    ] == [
        ("one.bin", 100, files[0].sha256),
        ("it's.txt", 2, files[1].sha256),
    ]
    assert {e.status for e in result.effects} == {"applied"}
    assert all(e.verified for e in result.effects)
    assert "attachment:one.bin (bytes)" in result.compared
    [stored] = fake.entries.values()
    assert stored.attachments == {"one.bin": files[0].content, "it's.txt": b"hi"}
    # Bytes really were downloaded for the comparison, not inferred from the POST.
    assert fake.calls("GET", "one.bin")[0].path.endswith("$value")


def test_a_corrupted_attachment_download_fails_verification_with_the_entry_id(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.corrupt_downloads.add("a.bin")

    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(
            client, title="T", body="B", attachments=[attachment("a.bin", b"data")]
        )

    error = caught.value
    assert error.outcome == "verification_failed"
    assert error.entry_id == "1" and error.mismatches == ["attachment:a.bin (bytes)"]
    assert statuses(error)["add_attachment:a.bin"] == "applied"
    assert error.effects[1].verified is False
    assert error.effects[0].verified is True  # the fields themselves matched
    assert error.retry_safe is False and "do not blindly repeat" in error.next_step
    assert len(fake.entries) == 1


def test_a_body_the_readback_does_not_show_fails_verification(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.mangle_body = True
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[])
    assert caught.value.outcome == "verification_failed"
    assert caught.value.mismatches == ["body"]


def test_an_unavailable_readback_is_reported_with_the_entry_id(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["GET item"] = ["503"]
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[])
    assert caught.value.outcome == "verification_unavailable"
    assert caught.value.entry_id == "1" and len(fake.entries) == 1


@pytest.mark.parametrize(
    ("script", "outcome", "status"),
    [("403", "not_applied", "failed"), ("400", "not_applied", "failed")],
)
def test_a_rejected_create_changed_nothing_and_is_retry_safe(
    fake: FakeMicrosoft,
    client: NewsletterClient,
    script: str,
    outcome: str,
    status: str,
) -> None:
    fake.script["POST items"] = [script]
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[attachment("a", b"x")])
    error = caught.value
    assert (error.outcome, error.retry_safe, error.entry_id) == (outcome, True, None)
    assert statuses(error) == {
        "create_entry:entry": status,
        "add_attachment:a": "not_attempted",
    }
    assert not fake.entries


@pytest.mark.parametrize("script", ["drop", "502", "apply-then-500"])
def test_an_unanswered_create_is_uncertain_and_never_retried(
    fake: FakeMicrosoft, client: NewsletterClient, script: str
) -> None:
    fake.script["POST items"] = [script]
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[attachment("a", b"x")])
    error = caught.value
    assert error.outcome == "create_uncertain" and error.retry_safe is False
    assert error.entry_id is None
    assert statuses(error)["create_entry:entry"] == "uncertain"
    assert "Do NOT create again" in error.next_step
    assert len(fake.creates()) == 1  # exactly one attempt, ever


def test_a_failed_attachment_preserves_the_entry_and_later_ones_are_not_attempted(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["POST attachment-add:b.bin"] = ["403"]
    files = [
        attachment("a.bin", b"A"),
        attachment("b.bin", b"B"),
        attachment("c.bin", b"C"),
    ]

    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=files)

    error = caught.value
    assert error.outcome == "partial" and error.entry_id == "1"
    assert error.entry_url
    assert statuses(error) == {
        "create_entry:entry": "applied",
        "add_attachment:a.bin": "applied",
        "add_attachment:b.bin": "failed",
        "add_attachment:c.bin": "not_attempted",
    }
    assert len(fake.entries) == 1 and list(fake.entries[1].attachments) == ["a.bin"]
    assert len(fake.creates()) == 1


def test_recovery_by_update_finishes_the_attachments_without_duplicating_anything(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["POST attachment-add:b.bin"] = ["500"]
    files = [attachment("a.bin", b"A"), attachment("b.bin", b"B")]
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=files)
    assert statuses(caught.value)["add_attachment:b.bin"] == "uncertain"

    # The caller recovers with the SAME attachments against the known entry.
    result = update_entry(
        client, caught.value.entry_id or "", title=None, body=None, attachments=files
    )

    assert result.verified
    assert {e.subject: e.status for e in result.effects} == {
        "a.bin": "unchanged",
        "b.bin": "applied",
    }
    assert len(fake.entries) == 1  # still exactly one entry
    assert fake.entries[1].attachments == {"a.bin": b"A", "b.bin": b"B"}
    assert len(fake.creates()) == 1
    assert len(fake.calls("POST", "/add(")) == 3  # a, failed b, b again


def test_an_attachment_applied_before_a_lost_reply_is_settled_by_readback(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["POST attachment-add:a.bin"] = ["apply-then-500"]
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(
            client, title="T", body="B", attachments=[attachment("a.bin", b"A")]
        )
    [_, att] = caught.value.effects
    assert (att.status, att.verified) == ("applied", True)
    assert caught.value.outcome == "partial"  # still reported: the reply was lost


def test_a_dropped_attachment_upload_stays_uncertain_when_the_readback_cannot_settle_it(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["POST attachment-add:a.bin"] = ["drop"]
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(
            client, title="T", body="B", attachments=[attachment("a.bin", b"A")]
        )
    assert statuses(caught.value)["add_attachment:a.bin"] == "uncertain"
    assert (
        "retry only attachments marked failed or not_attempted"
        in caught.value.next_step
    )


# -- update -----------------------------------------------------------------


def test_field_update_changes_only_what_was_supplied_and_verifies_the_rest(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("Old", "<p>keep me</p>", attachments={"x.bin": b"X"})

    result = update_entry(client, str(entry.id), title="New", body=None, attachments=[])

    assert (result.entry.title, result.entry.body) == ("New", "keep me")
    assert fake.entries[1].body_html == "<p>keep me</p>"
    patch = fake.calls("PATCH", "/fields")
    assert len(patch) == 1 and patch[0].body == b'{"Title": "New"}'
    assert {"title", "body unchanged", "attachment:x.bin still present"} <= set(
        result.compared
    )


def test_an_attachment_only_update_sends_no_field_write_and_preserves_content(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("Title", "<p>Body</p>", attachments={"old.pdf": b"OLD"})
    new = attachment("new.pdf", b"NEW" * 100)

    result = update_entry(
        client, str(entry.id), title=None, body=None, attachments=[new]
    )

    assert not fake.calls("PATCH", "")
    assert (fake.entries[1].title, fake.entries[1].body_html) == (
        "Title",
        "<p>Body</p>",
    )
    assert [a.name for a in result.entry.attachments or []] == ["old.pdf", "new.pdf"]
    assert {e.kind for e in result.effects} == {"add_attachment"}
    assert {"title unchanged", "body unchanged", "attachment:new.pdf (bytes)"} <= set(
        result.compared
    )


def test_an_untouched_field_that_changed_underneath_is_caught(
    fake: FakeMicrosoft, client: NewsletterClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = fake.add_entry("Title", "<p>Body</p>")
    original = client.add_attachment_bytes

    def edit_concurrently(item_id: str, name: str, content: bytes) -> None:
        original(item_id, name, content)
        fake.entries[1].title = "Someone else's edit"

    monkeypatch.setattr(client, "add_attachment_bytes", edit_concurrently)
    with pytest.raises(NewsletterWriteError) as caught:
        update_entry(
            client,
            str(entry.id),
            title=None,
            body=None,
            attachments=[attachment("a", b"A")],
        )
    assert caught.value.outcome == "verification_failed"
    assert caught.value.mismatches == ["title unchanged"]


def test_an_existing_attachment_with_different_bytes_is_refused_before_any_write(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("T", "<p>B</p>", attachments={"a.pdf": b"v1"})
    with pytest.raises(NewsletterInputError, match="does not replace"):
        update_entry(
            client,
            str(entry.id),
            title="New",
            body=None,
            attachments=[attachment("A.PDF", b"v2"), attachment("n.pdf", b"n")],
        )
    assert not fake.calls("PATCH", "") and not fake.calls("POST", "/add(")
    assert fake.entries[1].title == "T"


def test_a_rejected_field_update_is_not_applied_and_skips_attachments(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("T", "<p>B</p>")
    fake.script["PATCH fields"] = ["403"]
    with pytest.raises(NewsletterWriteError) as caught:
        update_entry(
            client,
            str(entry.id),
            title="N",
            body=None,
            attachments=[attachment("a", b"A")],
        )
    assert caught.value.outcome == "not_applied" and caught.value.retry_safe
    assert statuses(caught.value) == {
        "update_fields:entry": "failed",
        "add_attachment:a": "not_attempted",
    }
    assert not fake.calls("POST", "/add(")


def test_an_unanswered_field_update_is_partial_and_uncertain(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    entry = fake.add_entry("T", "<p>B</p>")
    fake.script["PATCH fields"] = ["drop"]
    with pytest.raises(NewsletterWriteError) as caught:
        update_entry(client, str(entry.id), title="N", body=None, attachments=[])
    assert caught.value.outcome == "partial" and not caught.value.retry_safe
    assert statuses(caught.value) == {"update_fields:entry": "uncertain"}
    assert len(fake.calls("PATCH", "")) == 1


def test_an_unknown_entry_is_a_not_found_error_before_any_write(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    with pytest.raises(NewsletterError) as caught:
        update_entry(client, "404", title="N", body=None, attachments=[])
    assert caught.value.status == 404
    assert not fake.calls("PATCH", "")


# -- resource separation, permissions ---------------------------------------


def test_graph_and_sharepoint_calls_use_their_own_audience_tokens(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    create_entry(client, title="T", body="B", attachments=[attachment("a", b"A")])

    graph = {r.token for r in fake.requests if r.path.startswith("/graph")}
    sharepoint = {r.token for r in fake.requests if r.path.startswith("/sp")}
    assert len(graph) == 1 and len(sharepoint) == 1 and graph != sharepoint
    assert len(fake.refreshes(GRAPH_RESOURCE)) == 1
    assert len(fake.refreshes(SHAREPOINT_RESOURCE)) == 1


def test_a_graph_token_does_not_work_for_sharepoint(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    graph_only = NewsletterClient(
        token_provider=lambda resource: client._token_provider(GRAPH_RESOURCE),  # noqa: SLF001
        graph_root=fake.graph_root,
        sharepoint_root=fake.sharepoint_root,
    )
    entry = fake.add_entry("T", "<p>B</p>")
    graph_only.get_item(str(entry.id))
    with pytest.raises(NewsletterError) as caught:
        graph_only.list_attachments(str(entry.id))
    assert caught.value.status == 401


def test_a_denied_attachment_after_create_keeps_the_entry_and_says_why(
    fake: FakeMicrosoft, client: NewsletterClient
) -> None:
    fake.script["POST attachment-add:a"] = ["403"]
    with pytest.raises(NewsletterWriteError) as caught:
        create_entry(client, title="T", body="B", attachments=[attachment("a", b"A")])
    cause = caught.value.__cause__
    assert isinstance(cause, NewsletterError) and cause.status == 403
    assert caught.value.entry_id == "1"


# -- inputs -----------------------------------------------------------------


def test_prepare_attachments_reads_and_hashes_every_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"alpha")
    [prepared] = prepare_attachments([tmp_path / "a.txt"])
    assert prepared == attachment("a.txt", b"alpha")


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["missing.txt"], "not a file"),
        (["bad#name.txt"], "not a usable SharePoint file name"),
        ([".hidden"], "not a usable SharePoint file name"),
        (["dup.txt", "DUP.txt"], "two attachments"),
    ],
)
def test_bad_attachment_inputs_are_refused_before_any_request(
    tmp_path: Path, names: list[str], message: str
) -> None:
    for name in names:
        if name != "missing.txt":
            (tmp_path / name).write_bytes(b"x")
    with pytest.raises(NewsletterInputError, match=message):
        prepare_attachments([tmp_path / n for n in names])


def test_too_many_attachments_and_directories_are_refused(tmp_path: Path) -> None:
    with pytest.raises(NewsletterInputError, match="at most"):
        prepare_attachments([tmp_path / f"{n}.txt" for n in range(11)])
    with pytest.raises(NewsletterInputError, match="not a file"):
        prepare_attachments([tmp_path])
