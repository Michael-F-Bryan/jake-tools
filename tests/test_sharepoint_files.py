"""The SharePoint file client and workflows over real HTTP.

The real :class:`DriveClient`, the real refresh-aware token provider and the
real workflows talk to ``fake_drive.FakeDrive`` on a loopback port (token
endpoint, Graph drive items, pre-authenticated download/upload URLs). This is
protocol-level evidence about OUR requests and OUR handling of the documented
responses. It says nothing about live Microsoft behaviour (issue #41).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fake_drive import DRIVE_ID, ROOT_ID, FakeDrive
from fake_microsoft import ACCOUNT, CLIENT, REFRESH_TOKEN, TENANT

from jake_tools import file_transfers
from jake_tools.file_transfers import (
    FileInputError,
    FileReadError,
    FileTransferError,
    VersionConflictError,
    check_link,
    check_name,
    download_file,
    download_pdf,
    get_item,
    list_folder,
    read_source,
    replace_file,
    resolve_channel,
    resolve_link,
    upload_new,
)
from jake_tools.microsoft_auth import MicrosoftSettings, MicrosoftTokenProvider
from jake_tools.sharepoint_files import DriveClient, FilesError, encode_sharing_url

HOSTS = ("csuses.sharepoint.com",)
LINK = "https://csuses.sharepoint.com/:f:/s/team/EabcDEF?e=SECRETLINKTOKEN"


@pytest.fixture
def fake() -> Iterator[FakeDrive]:
    service = FakeDrive().start()
    try:
        yield service
    finally:
        service.stop()


@pytest.fixture
def client(fake: FakeDrive) -> DriveClient:
    settings = MicrosoftSettings(
        tenant_id=TENANT,
        client_id=CLIENT,
        account=ACCOUNT,
        refresh_token=REFRESH_TOKEN,
        authority_url=fake.base_url,
    )
    return DriveClient(
        token_provider=MicrosoftTokenProvider(settings), graph_root=fake.graph_root
    )


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def requests_to(fake: FakeDrive, method: str, fragment: str) -> list[dict[str, str]]:
    return [h for m, p, h in fake.header_log if m == method and fragment in p]


def headers_of(fake: FakeDrive, method: str, fragment: str) -> dict[str, str]:
    found = requests_to(fake, method, fragment)
    assert found, f"no {method} {fragment}"
    return found[-1]


def one_shot(fake: FakeDrive, hook: str, action: Callable[[], None]) -> None:
    def run() -> None:
        fake.hooks.pop(hook, None)
        action()

    fake.hooks[hook] = run


# -- locating ------------------------------------------------------------------


def test_a_link_resolves_through_the_shares_api_with_the_documented_encoding(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.links[LINK] = folder.id

    result = resolve_link(client, LINK, hosts=HOSTS)

    assert result.resolved_by == "link"
    assert (result.item.drive_id, result.item.item_id) == (DRIVE_ID, folder.id)
    assert result.item.kind == "folder" and result.item.etag == folder.etag
    path = fake.calls("GET", "/shares/")[0].path
    assert path == f"/graph/v1.0/shares/{encode_sharing_url(LINK)}/driveItem"
    assert encode_sharing_url("https://x/y?a=b+c/d") == ("u!aHR0cHM6Ly94L3k_YT1iK2MvZA")
    assert headers_of(fake, "GET", "/shares/")["Authorization"].startswith("Bearer ")


def test_a_link_on_another_host_is_refused_before_any_request(
    fake: FakeDrive, client: DriveClient
) -> None:
    for url in (
        "https://other.sharepoint.com/sites/x",
        "http://csuses.sharepoint.com/sites/x",
        "https://user:pw@csuses.sharepoint.com/sites/x",
        "not a url",
    ):
        with pytest.raises(FileInputError) as caught:
            resolve_link(client, url, hosts=HOSTS)
        assert caught.value.argument == "url"
    assert fake.requests == []
    assert check_link("HTTPS://CSUSES.SharePoint.com/x", HOSTS)


def test_an_unknown_link_is_a_404_not_an_invented_item(
    client: DriveClient,
) -> None:
    with pytest.raises(FilesError) as caught:
        resolve_link(client, "https://csuses.sharepoint.com/nope", hosts=HOSTS)
    assert caught.value.status == 404 and caught.value.code == "itemNotFound"
    assert "nope" not in str(caught.value)  # the link is never echoed in an error


def test_a_channel_resolves_to_its_files_folder(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("General")
    fake.channels[("team-1", "19:abc@thread.tacv2")] = folder.id

    result = resolve_channel(client, team_id="team-1", channel_id="19:abc@thread.tacv2")

    assert result.resolved_by == "channel" and result.item.item_id == folder.id
    assert fake.calls("GET", "/teams/team-1/channels/19:abc@thread.tacv2/filesFolder")


def test_get_item_reports_the_version_identity(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("plan.docx", b"hello", by="Pat")

    item = get_item(client, drive_id=DRIVE_ID, item_id=node.id)

    assert item.kind == "file" and item.size_bytes == 5
    assert item.etag == node.etag and item.ctag == node.ctag
    assert item.modified_by == "Pat" and item.name == "plan.docx"
    assert item.web_url.endswith("/plan.docx")
    with pytest.raises(FilesError) as caught:
        get_item(client, drive_id=DRIVE_ID, item_id="nope")
    assert caught.value.status == 404


# -- listing -------------------------------------------------------------------


def test_listing_pages_with_explicit_coverage(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Docs")
    for number in range(5):
        fake.add_file(f"f{number}.txt", b"x", parent=folder.id)

    pages = [
        list_folder(client, drive_id=DRIVE_ID, item_id=folder.id, limit=2, cursor=None)
    ]
    while not pages[-1].complete:
        pages.append(
            list_folder(
                client,
                drive_id=DRIVE_ID,
                item_id=folder.id,
                limit=2,
                cursor=pages[-1].next_cursor,
            )
        )

    assert [p.count for p in pages] == [2, 2, 1]
    assert [p.complete for p in pages] == [False, False, True]
    assert pages[-1].next_cursor is None
    assert [e.name for p in pages for e in p.entries] == [f"f{n}.txt" for n in range(5)]
    assert all(e.etag and e.item_id for p in pages for e in p.entries)


def test_a_server_cap_below_the_limit_is_not_reported_complete(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Docs")
    for number in range(3):
        fake.add_file(f"f{number}.txt", b"x", parent=folder.id)
    fake.children_cap = 1

    page = list_folder(
        client, drive_id=DRIVE_ID, item_id=folder.id, limit=200, cursor=None
    )

    assert page.count == 1 and page.complete is False and page.next_cursor


def test_a_next_link_without_a_token_is_an_error_not_a_shorter_folder(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Docs")
    fake.script["GET children"] = [
        'apply-then-json:{"value": [], "@odata.nextLink": "https://x/y?$top=1"}'
    ]
    with pytest.raises(FilesError, match="no continuation token"):
        list_folder(client, drive_id=DRIVE_ID, item_id=folder.id, limit=5, cursor=None)


def test_a_reply_without_the_collection_is_not_an_empty_folder(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Docs")
    fake.script["GET children"] = ['apply-then-json:{"unexpected": true}']
    with pytest.raises(FilesError, match="unexpected response shape"):
        list_folder(client, drive_id=DRIVE_ID, item_id=folder.id, limit=5, cursor=None)


# -- download ------------------------------------------------------------------


def test_download_follows_the_redirect_without_the_token_and_saves_the_version(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("plan.docx", b"original bytes")
    target = tmp_path / "plan.docx"

    result = download_file(
        client, drive_id=DRIVE_ID, item_id=node.id, destination=target
    )

    assert target.read_bytes() == b"original bytes"
    assert result.size_bytes == 14 and result.sha256 == sha(b"original bytes")
    assert result.item.etag == node.etag and result.item.item_id == node.id
    assert result.format == "original"
    assert any("etag unchanged" in c for c in result.compared)
    # The /content call carried the bearer token; the pre-authenticated one did not
    # (the service answers 401 to any Authorization header there).
    assert headers_of(fake, "GET", "/content")["Authorization"].startswith("Bearer ")
    assert "Authorization" not in headers_of(fake, "GET", "/download/")


def test_a_file_that_changed_while_it_was_read_is_not_saved(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("plan.docx", b"v1")
    target = tmp_path / "plan.docx"
    calls = {"n": 0}

    def maybe_edit() -> None:
        calls["n"] += 1
        if calls["n"] == 1:  # after the first metadata read, before the content
            fake.edit(node, b"v2 edited")

    fake.hooks["after_get_item"] = maybe_edit
    # The edit lands after the first read of the item, so the second shows it.
    with pytest.raises(FileReadError) as caught:
        download_file(client, drive_id=DRIVE_ID, item_id=node.id, destination=target)

    assert caught.value.reason == "changed_during_download"
    assert caught.value.item is not None and caught.value.item.etag == node.etag
    assert not target.exists()


def test_a_wrong_sized_download_is_not_saved(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("plan.docx", b"12345")
    fake.corrupt_download_names.add("plan.docx")
    with pytest.raises(FileReadError) as caught:
        download_file(
            client,
            drive_id=DRIVE_ID,
            item_id=node.id,
            destination=tmp_path / "out.docx",
        )
    assert caught.value.reason == "size_mismatch"
    assert not (tmp_path / "out.docx").exists()


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda fake: fake.add_folder("Dir"), "not_a_file"),
    ],
)
def test_download_refuses_what_is_not_a_file(
    fake: FakeDrive,
    client: DriveClient,
    tmp_path: Path,
    make: Callable[[FakeDrive], object],
    reason: str,
) -> None:
    node = make(fake)
    with pytest.raises(FileInputError) as caught:
        download_file(
            client,
            drive_id=DRIVE_ID,
            item_id=node.id,  # type: ignore[attr-defined]
            destination=tmp_path / "x",
        )
    assert caught.value.reason == reason
    assert fake.calls("GET", "/content") == []


def test_download_never_overwrites_and_needs_a_real_absolute_destination(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("a.txt", b"x")
    existing = tmp_path / "keep.txt"
    existing.write_bytes(b"mine")
    for destination, reason in (
        (existing, "destination_exists"),
        (Path("relative.txt"), None),
        (tmp_path / "missing-dir" / "a.txt", None),
    ):
        with pytest.raises(FileInputError) as caught:
            download_file(
                client, drive_id=DRIVE_ID, item_id=node.id, destination=destination
            )
        assert caught.value.argument == "destination"
        assert caught.value.reason == reason
    assert existing.read_bytes() == b"mine"
    assert fake.requests == []  # refused before any request, token included


def test_a_file_over_the_bound_is_refused_before_its_content_is_requested(
    fake: FakeDrive,
    client: DriveClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = fake.add_file("big.bin", b"0123456789")
    monkeypatch.setattr(file_transfers, "MAX_TRANSFER_BYTES", 5)
    with pytest.raises(FileInputError) as caught:
        download_file(
            client, drive_id=DRIVE_ID, item_id=node.id, destination=tmp_path / "b"
        )
    assert caught.value.reason == "too_large"
    assert fake.calls("GET", "/content") == []


def test_a_content_body_past_the_bound_is_cut_off(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("big.bin", b"0123456789")
    with pytest.raises(FilesError, match="exceeds 4 bytes"):
        client.download(DRIVE_ID, node.id, max_bytes=4)


def test_a_failed_content_request_is_reported_without_the_secret_url(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("a.txt", b"x")
    fake.script["GET download"] = ["503"]
    with pytest.raises(FilesError) as caught:
        download_file(
            client, drive_id=DRIVE_ID, item_id=node.id, destination=tmp_path / "a"
        )
    assert caught.value.status == 503
    token = fake.calls("GET", "/download/")[0].path.rsplit("/", 1)[-1]
    assert token not in str(caught.value) and fake.base_url not in str(caught.value)
    assert not (tmp_path / "a").exists()


# -- rendered PDF --------------------------------------------------------------


def test_a_supported_file_is_rendered_as_a_pdf_with_the_documented_request(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("minutes.docx", b"PK word")

    result = download_pdf(
        client, drive_id=DRIVE_ID, item_id=node.id, destination=tmp_path / "m.pdf"
    )

    assert (tmp_path / "m.pdf").read_bytes().startswith(b"%PDF-")
    assert result.format == "pdf" and result.item.etag == node.etag
    assert "starts with %PDF-" in result.compared
    assert fake.calls("GET", "/content")[0].path.endswith("/content")
    assert result.sha256 == sha((tmp_path / "m.pdf").read_bytes())


def test_an_unsupported_source_is_refused_before_any_conversion_request(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("photo.png", b"\x89PNG")
    with pytest.raises(FileInputError) as caught:
        download_pdf(
            client, drive_id=DRIVE_ID, item_id=node.id, destination=tmp_path / "p.pdf"
        )
    assert caught.value.reason == "unsupported_conversion"
    assert "docx" in caught.value.extra["supported_extensions"]
    assert fake.calls("GET", "/content") == []
    assert not (tmp_path / "p.pdf").exists()


def test_a_conversion_microsoft_refuses_is_reported_as_unsupported(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("huge.docx", b"PK")
    fake.refuse_pdf_names.add("huge.docx")
    with pytest.raises(FileInputError) as caught:
        download_pdf(
            client, drive_id=DRIVE_ID, item_id=node.id, destination=tmp_path / "h.pdf"
        )
    assert caught.value.reason == "unsupported_conversion"
    assert caught.value.extra["http_status"] == 400
    assert "2 MB" in str(caught.value)


def test_a_rendering_that_is_not_a_pdf_is_not_saved(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("odd.docx", b"PK")
    fake.not_pdf_names.add("odd.docx")
    with pytest.raises(FileReadError) as caught:
        download_pdf(
            client, drive_id=DRIVE_ID, item_id=node.id, destination=tmp_path / "o.pdf"
        )
    assert caught.value.reason == "not_a_pdf"
    assert not (tmp_path / "o.pdf").exists()


# -- new files -----------------------------------------------------------------


def upload(client: DriveClient, folder_id: str, name: str, data: bytes, **kw):
    return upload_new(
        client,
        drive_id=DRIVE_ID,
        parent_id=folder_id,
        name=name,
        content=data,
        conflict=kw.pop("conflict", "fail"),
    )


def test_a_new_file_is_created_through_an_upload_session_and_read_back(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")

    result = upload(client, folder.id, "list 2026.xlsx", b"PK-contact-list")

    saved = fake.named(folder.id, "list 2026.xlsx")
    assert saved is not None and saved.content == b"PK-contact-list"
    assert result.verified is True and result.operation == "create"
    assert result.item.item_id == saved.id and result.item.etag == saved.etag
    assert result.item.parent_id == folder.id and result.renamed is False
    assert result.sha256 == sha(b"PK-contact-list") and result.size_bytes == 15
    assert result.changed_since_write is False and result.etag_at_write == saved.etag
    assert "parent folder" in result.compared
    # The documented request shapes, and the secret upload URL without a token.
    session = fake.calls("POST", "createUploadSession")[0]
    assert session.path.endswith(
        f"/items/{folder.id}:/list 2026.xlsx:/createUploadSession"
    )
    assert b'"@microsoft.graph.conflictBehavior": "fail"' in session.body
    assert "Authorization" in headers_of(fake, "POST", "createUploadSession")
    put = headers_of(fake, "PUT", "/upload/")
    assert "Authorization" not in put and put["Content-Range"] == "bytes 0-14/15"


def test_a_name_clash_fails_without_touching_the_existing_file(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    existing = fake.add_file("list.xlsx", b"theirs", parent=folder.id)

    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "list.xlsx", b"mine")

    error = caught.value
    assert (error.outcome, error.reason, error.retry_safe) == (
        "not_applied",
        "name_exists",
        True,
    )
    assert "already exists" in error.next_step
    assert existing.content == b"theirs" and existing.version == 1
    assert len(fake.children_of(folder.id)) == 1


def test_rename_keeps_both_and_says_so(fake: FakeDrive, client: DriveClient) -> None:
    folder = fake.add_folder("Contacts")
    fake.add_file("list.xlsx", b"theirs", parent=folder.id)

    result = upload(client, folder.id, "list.xlsx", b"mine", conflict="rename")

    assert result.renamed is True and result.item.name == "list 1.xlsx"
    assert result.requested_name == "list.xlsx"
    assert fake.named(folder.id, "list.xlsx").content == b"theirs"  # type: ignore[union-attr]
    assert fake.named(folder.id, "list 1.xlsx").content == b"mine"  # type: ignore[union-attr]
    assert any("renamed" in c for c in result.compared)


def test_a_destination_that_is_not_a_folder_is_refused_first(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("a.txt", b"x")
    with pytest.raises(FileInputError) as caught:
        upload(client, node.id, "b.txt", b"y")
    assert caught.value.reason == "not_a_folder"
    assert fake.calls("POST", "createUploadSession") == []


def test_a_failed_session_start_means_nothing_was_written_even_on_a_5xx(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.script["POST session"] = ["503"]

    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"x")

    assert caught.value.outcome == "not_applied" and caught.value.retry_safe
    assert fake.calls("PUT", "/upload/") == []  # no content was ever sent
    assert fake.children_of(folder.id) == []


def test_a_rejected_content_upload_is_not_applied(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.script["PUT upload"] = ["403"]
    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"x")
    assert (caught.value.outcome, caught.value.reason) == ("not_applied", "rejected")
    assert isinstance(caught.value.__cause__, FilesError)
    assert fake.children_of(folder.id) == []


def test_a_5xx_after_the_write_is_uncertain_and_the_look_finds_the_bytes(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.script["PUT upload"] = ["apply-then-500"]

    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"payload")

    error = caught.value
    assert error.outcome == "write_uncertain" and error.retry_safe is False
    assert error.observed is not None and error.observed.state == "found_identical"
    assert error.observed.item is not None and error.observed.item.name == "a.txt"
    assert "Do NOT repeat" in error.next_step
    assert len(fake.calls("PUT", "/upload/")) == 1  # never retried
    detail = error.detail()
    assert detail["outcome"] == "write_uncertain" and "observed" in detail


def test_a_dropped_connection_that_wrote_nothing_is_uncertain_and_not_found(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.script["PUT upload"] = ["drop"]
    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"payload")
    assert caught.value.outcome == "write_uncertain"
    assert caught.value.observed is not None
    assert caught.value.observed.state == "not_found"


def test_a_lost_reply_after_a_commit_is_uncertain_not_a_failure(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.script["PUT upload"] = ["apply-then-drop"]
    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"payload")
    assert caught.value.outcome == "write_uncertain"
    assert caught.value.observed is not None
    assert caught.value.observed.state == "found_identical"


def test_an_unreadable_2xx_after_the_upload_is_uncertain(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.script["PUT upload"] = ['apply-then-json:{"unexpected": 1}']
    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"payload")
    assert caught.value.outcome == "write_uncertain"
    assert caught.value.observed is not None
    assert caught.value.observed.state == "found_identical"


def test_remote_content_that_differs_from_the_upload_fails_verification(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.corrupt_commit = True

    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"payload")

    error = caught.value
    assert error.outcome == "verification_failed" and not error.retry_safe
    assert any(m.startswith("content") for m in error.mismatches)
    assert error.item is not None and error.item.name == "a.txt"
    assert error.changed_since_write is False
    assert "do not repeat" in error.next_step.lower()


def test_a_failed_readback_is_unavailable_not_success(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    fake.script["GET content"] = ["500"]  # the readback's download

    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"payload")

    assert caught.value.outcome == "verification_unavailable"
    assert caught.value.item is not None  # the item that was acknowledged
    assert fake.named(folder.id, "a.txt") is not None
    assert len(fake.calls("PUT", "/upload/")) == 1


def test_a_change_after_the_write_is_reported_not_hidden(
    fake: FakeDrive, client: DriveClient
) -> None:
    folder = fake.add_folder("Contacts")
    seen = {"n": 0}

    def edit_before_readback() -> None:
        seen["n"] += 1
        if seen["n"] == 2:  # parent check is 1; the readback metadata is 2
            node = fake.named(folder.id, "a.txt")
            assert node is not None
            fake.edit(node, b"someone else's edit")

    fake.hooks["before_get_item"] = edit_before_readback
    with pytest.raises(FileTransferError) as caught:
        upload(client, folder.id, "a.txt", b"payload")
    assert caught.value.outcome == "verification_failed"
    assert caught.value.changed_since_write is True


@pytest.mark.parametrize(
    "name",
    [
        "",
        " lead.txt",
        "trail.txt ",
        "dot.",
        "a/b.txt",
        'q"uote.txt',
        "~$temp.docx",
        "x" * 256,
    ],
)
def test_unusable_names_are_refused(name: str) -> None:
    with pytest.raises(FileInputError):
        check_name(name)


def test_usable_names_pass() -> None:
    for name in ("Contact list (2026) & more.xlsx", "100% #1.docx", ".gitkeep.txt"):
        assert check_name(name) == name


def test_the_source_file_is_checked_before_anything_is_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    ok = tmp_path / "ok.txt"
    ok.write_bytes(b"abc")
    for bad in (Path("relative.txt"), tmp_path / "missing.txt", tmp_path, empty):
        with pytest.raises(FileInputError):
            read_source(bad)
    assert read_source(ok) == b"abc"
    monkeypatch.setattr(file_transfers, "MAX_TRANSFER_BYTES", 2)
    with pytest.raises(FileInputError) as caught:
        read_source(ok)
    assert caught.value.reason == "too_large"


# -- conditional replacement ---------------------------------------------------


def replace(client: DriveClient, node_id: str, data: bytes, etag: str):
    return replace_file(
        client,
        drive_id=DRIVE_ID,
        item_id=node_id,
        content=data,
        expected_etag=etag,
    )


def test_a_replacement_at_the_expected_version_sends_if_match_and_verifies(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("plan.docx", b"v1")
    before = node.etag

    result = replace(client, node.id, b"version two", before)

    assert result.operation == "replace" and result.verified is True
    assert node.content == b"version two" and node.version == 2
    assert result.expected_etag == before and result.item.etag == node.etag
    assert result.item.etag != before and result.changed_since_write is False
    assert result.sha256 == sha(b"version two")
    header = headers_of(fake, "POST", "createUploadSession")
    assert header["If-Match"] == before  # verbatim, quotes and braces included
    assert fake.calls("POST", "createUploadSession")[0].path.endswith(
        f"/items/{node.id}/createUploadSession"
    )


def test_a_file_changed_before_the_call_is_refused_without_a_session(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("plan.docx", b"v1")
    seen = node.etag
    fake.edit(node, b"their edit", by="Sam Editor")

    with pytest.raises(VersionConflictError) as caught:
        replace(client, node.id, b"mine", seen)

    assert caught.value.expected == seen
    assert caught.value.current.etag == node.etag
    assert caught.value.current.modified_by == "Sam Editor"
    assert node.content == b"their edit"
    assert fake.calls("POST", "createUploadSession") == []
    assert fake.calls("PUT", "/upload/") == []


def test_an_edit_that_lands_after_the_check_is_stopped_by_if_match(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("plan.docx", b"v1")
    seen = node.etag
    one_shot(fake, "before_session", lambda: fake.edit(node, b"their late edit", "Sam"))

    with pytest.raises(FileTransferError) as caught:
        replace(client, node.id, b"mine", seen)

    error = caught.value
    assert (error.outcome, error.reason) == ("not_applied", "version_conflict")
    assert error.retry_safe and "Download the current version" in error.next_step
    assert error.item is not None and error.item.etag == node.etag  # who has it now
    assert error.item.modified_by == "Sam"
    assert node.content == b"their late edit"  # not overwritten
    assert fake.calls("PUT", "/upload/") == []  # no content was sent


def test_replacing_a_folder_is_refused(fake: FakeDrive, client: DriveClient) -> None:
    folder = fake.add_folder("Dir")
    with pytest.raises(FileInputError) as caught:
        replace(client, folder.id, b"x", folder.etag)
    assert caught.value.reason == "not_a_file"


def test_a_locked_file_is_reported_as_not_applied(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("plan.docx", b"v1")
    fake.script["PUT upload"] = ["423"]
    with pytest.raises(FileTransferError) as caught:
        replace(client, node.id, b"mine", node.etag)
    assert (caught.value.outcome, caught.value.reason) == ("not_applied", "locked")
    assert node.content == b"v1"


def test_an_uncertain_replacement_is_not_retried_and_looks_at_the_item(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("plan.docx", b"v1")
    fake.script["PUT upload"] = ["apply-then-500"]

    with pytest.raises(FileTransferError) as caught:
        replace(client, node.id, b"mine now", node.etag)

    assert caught.value.outcome == "write_uncertain"
    assert caught.value.observed is not None
    assert caught.value.observed.state == "found_identical"
    assert len(fake.calls("PUT", "/upload/")) == 1


def test_a_replacement_whose_readback_differs_is_not_verified(
    fake: FakeDrive, client: DriveClient
) -> None:
    node = fake.add_file("plan.docx", b"v1")
    fake.corrupt_commit = True
    with pytest.raises(FileTransferError) as caught:
        replace(client, node.id, b"mine now", node.etag)
    assert caught.value.outcome == "verification_failed"


def test_the_token_is_refreshed_not_reused_past_expiry_and_never_in_errors(
    fake: FakeDrive, client: DriveClient, tmp_path: Path
) -> None:
    node = fake.add_file("a.txt", b"x")
    fake.script["GET item"] = ["403"]
    with pytest.raises(FilesError) as caught:
        get_item(client, drive_id=DRIVE_ID, item_id=node.id)
    assert caught.value.status == 403
    for secret in (REFRESH_TOKEN, *fake.issued):
        assert secret not in str(caught.value)
    assert len(fake.refreshes("https://graph.microsoft.com")) == 1
