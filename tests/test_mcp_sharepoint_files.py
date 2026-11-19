"""The SharePoint file tools through the real MCP server over stdio.

The server runs under a hermetic HOME with only the configuration these tests
pass in. The one fake is Microsoft at its HTTP boundary (``fake_drive.py``).
Argument validation, handlers, the shared token provider, the client and the
workflows are ours and unmodified. This proves protocol and mechanics, NOT live
Microsoft compatibility (issue #41).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest
from fake_drive import DRIVE_ID, FakeDrive
from fake_microsoft import CLIENT_SECRET, REFRESH_TOKEN
from mcp import ClientSession
from mcp.types import TextResourceContents

from jake_tools.mcp.errors import ToolErrorPayload

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]
EnvFactory = Callable[..., dict[str, str]]
TOOLS = {
    "sharepoint_resolve",
    "sharepoint_get",
    "sharepoint_list",
    "sharepoint_download",
    "sharepoint_pdf",
    "sharepoint_upload",
}
LINK = "https://csuses.sharepoint.com/:f:/s/team/EabcDEF?e=SECRETLINKTOKEN"


@pytest.fixture
def fake() -> Iterator[FakeDrive]:
    service = FakeDrive().start()
    try:
        yield service
    finally:
        service.stop()


@pytest.fixture
def env(fake: FakeDrive, mcp_env_factory: EnvFactory, mcp_home: Path) -> dict[str, str]:
    return mcp_env_factory(mcp_home, **fake.env())


@pytest.fixture
def errlog(tmp_path: Path) -> Iterator[Any]:
    with (tmp_path / "server.stderr").open("w+") as handle:
        yield handle


def payload(result: Any) -> dict[str, Any]:
    assert result.is_error is False, result.structured_content
    assert result.structured_content is not None, result
    return result.structured_content


def error(result: Any) -> ToolErrorPayload:
    assert result.is_error is True, result.structured_content
    return ToolErrorPayload.model_validate(result.structured_content)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_no_secret(fake: FakeDrive, *texts: str) -> None:
    secrets = {REFRESH_TOKEN, CLIENT_SECRET, "SECRETLINKTOKEN", *fake.issued}
    secrets |= set(fake.downloads) | set(fake.sessions)
    for text in texts:
        for secret in secrets:
            assert secret not in text, secret


# -- discovery and configuration -------------------------------------------------


async def test_discovery_lists_the_tools_bounds_and_the_bundled_guidance(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        resource = await session.read_resource("skill://sharepoint-files")

    assert set(tools) >= TOOLS
    # Replacing an existing file is deliberately not offered (no commit-time
    # version condition is documented): no tool may be named for it.
    assert not [name for name in tools if "replace" in name]
    listing = tools["sharepoint_list"].input_schema["properties"]
    assert (listing["limit"]["minimum"], listing["limit"]["maximum"]) == (1, 200)
    assert listing["drive_id"]["pattern"] and listing["item_id"]["pattern"]
    upload = tools["sharepoint_upload"].input_schema["properties"]
    assert upload["on_conflict"]["enum"] == ["fail", "rename"]  # never "replace"
    for name in TOOLS:
        description = " ".join((tools[name].description or "").split())
        assert (
            "Before using this tool, read the MCP resource `skill://sharepoint-files`"
            in description
        )
    assert "WRITE ACTION" in (tools["sharepoint_upload"].description or "")
    content = resource.contents[0]
    assert isinstance(content, TextResourceContents)
    for needed in (
        "write_uncertain",
        "unsupported_conversion",
        "Replacing an existing file is not offered",
        "verified: true",
    ):
        assert needed in content.text


async def test_without_configuration_each_tool_names_what_is_missing(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        read = await session.call_tool(
            "sharepoint_get", {"drive_id": DRIVE_ID, "item_id": "x"}
        )
        write = await session.call_tool(
            "sharepoint_upload",
            {"drive_id": DRIVE_ID, "parent_id": "x", "source_path": "/tmp/x"},
        )
    for outcome in (read, write):
        failure = error(outcome)
        assert failure.code == "missing_credentials"
        assert failure.detail and failure.detail["variables"] == [
            "MICROSOFT_REFRESH_TOKEN"
        ]
        assert "SharePoint files" in failure.message


async def test_argument_bounds_keep_the_error_contract(
    mcp_server: ServerFactory, env: dict[str, str]
) -> None:
    async with mcp_server(env) as session:
        bad = [
            await session.call_tool(
                "sharepoint_list", {"drive_id": DRIVE_ID, "item_id": "a", "limit": 201}
            ),
            await session.call_tool(
                "sharepoint_get", {"drive_id": "../x", "item_id": "a"}
            ),
            await session.call_tool(
                "sharepoint_get", {"drive_id": DRIVE_ID, "item_id": "a/b"}
            ),
            await session.call_tool("sharepoint_resolve", {}),
            await session.call_tool(
                "sharepoint_resolve", {"url": LINK, "team_id": "t", "channel_id": "c"}
            ),
            await session.call_tool("sharepoint_resolve", {"team_id": "x" * 3}),
            await session.call_tool(
                "sharepoint_upload",
                {
                    "drive_id": DRIVE_ID,
                    "parent_id": "a",
                    "source_path": "/x",
                    "on_conflict": "replace",
                },
            ),
            await session.call_tool(
                "sharepoint_download",
                {"drive_id": DRIVE_ID, "item_id": "a", "destination": "rel.txt"},
            ),
        ]
    for outcome in bad:
        assert error(outcome).code == "invalid_argument"


# -- an end-to-end collaboration ---------------------------------------------------


async def test_resolve_list_download_pdf_and_new_uploads(
    mcp_server: ServerFactory,
    fake: FakeDrive,
    env: dict[str, str],
    tmp_path: Path,
    errlog: Any,
) -> None:
    channel = fake.add_folder("General")
    fake.channels[("0a3a5574-cfda-4314-952e-c0b3e1dcac6d", "19:abc@thread.tacv2")] = (
        channel.id
    )
    fake.links[LINK] = channel.id
    fake.add_file("Minutes.docx", b"PK minutes v1", parent=channel.id, by="Pat")
    for number in range(3):
        fake.add_file(f"extra{number}.txt", b"e", parent=channel.id)
    contacts = tmp_path / "contacts.xlsx"
    contacts.write_bytes(b"PK contacts")
    out = tmp_path / "got.docx"
    pdf = tmp_path / "got.pdf"

    async with mcp_server(env, errlog=errlog) as session:
        by_link = payload(await session.call_tool("sharepoint_resolve", {"url": LINK}))
        by_channel = payload(
            await session.call_tool(
                "sharepoint_resolve",
                {
                    "team_id": "0a3a5574-cfda-4314-952e-c0b3e1dcac6d",
                    "channel_id": "19:abc@thread.tacv2",
                },
            )
        )
        folder = by_link["item"]
        pages = [
            payload(
                await session.call_tool(
                    "sharepoint_list",
                    {
                        "drive_id": folder["drive_id"],
                        "item_id": folder["item_id"],
                        "limit": 3,
                    },
                )
            )
        ]
        while not pages[-1]["complete"]:
            pages.append(
                payload(
                    await session.call_tool(
                        "sharepoint_list",
                        {
                            "drive_id": folder["drive_id"],
                            "item_id": folder["item_id"],
                            "limit": 3,
                            "cursor": pages[-1]["next_cursor"],
                        },
                    )
                )
            )
        entry = next(
            e for p in pages for e in p["entries"] if e["name"] == "Minutes.docx"
        )
        downloaded = payload(
            await session.call_tool(
                "sharepoint_download",
                {
                    "drive_id": DRIVE_ID,
                    "item_id": entry["item_id"],
                    "destination": str(out),
                },
            )
        )
        rendered = payload(
            await session.call_tool(
                "sharepoint_pdf",
                {
                    "drive_id": DRIVE_ID,
                    "item_id": entry["item_id"],
                    "destination": str(pdf),
                },
            )
        )
        created = payload(
            await session.call_tool(
                "sharepoint_upload",
                {
                    "drive_id": DRIVE_ID,
                    "parent_id": folder["item_id"],
                    "source_path": str(contacts),
                },
            )
        )
        clash = error(
            await session.call_tool(
                "sharepoint_upload",
                {
                    "drive_id": DRIVE_ID,
                    "parent_id": folder["item_id"],
                    "source_path": str(contacts),
                },
            )
        )
        renamed = payload(
            await session.call_tool(
                "sharepoint_upload",
                {
                    "drive_id": DRIVE_ID,
                    "parent_id": folder["item_id"],
                    "source_path": str(contacts),
                    "on_conflict": "rename",
                },
            )
        )

    assert by_link["resolved_by"] == "link" and by_channel["resolved_by"] == "channel"
    assert by_link["item"]["item_id"] == by_channel["item"]["item_id"] == channel.id
    assert [p["count"] for p in pages] == [3, 1] and pages[-1]["complete"] is True
    assert pages[0]["complete"] is False
    assert (
        downloaded["sha256"] == sha(b"PK minutes v1")
        and out.read_bytes() == b"PK minutes v1"
    )
    assert downloaded["item"]["etag"] == entry["etag"]
    assert rendered["format"] == "pdf" and pdf.read_bytes().startswith(b"%PDF-")

    assert created["verified"] and created["renamed"] is False
    assert created["item"]["name"] == "contacts.xlsx"
    assert (clash.code, clash.detail["outcome"], clash.detail["reason"]) == (  # type: ignore[index]
        "invalid_argument",
        "not_applied",
        "name_exists",
    )
    assert renamed["renamed"] is True and renamed["item"]["name"] == "contacts 1.xlsx"
    names = sorted(n.name for n in fake.children_of(channel.id))
    assert "contacts.xlsx" in names and "contacts 1.xlsx" in names

    # One refresh served every call, and no secret URL or token leaked anywhere.
    assert len(fake.refreshes("https://graph.microsoft.com")) == 1
    everything = json.dumps(
        [
            by_link,
            pages,
            downloaded,
            rendered,
            created,
            clash.model_dump(),
            renamed,
        ]
    )
    assert_no_secret(fake, everything, errlog.read())


async def test_unsupported_pdf_and_not_found_are_actionable(
    mcp_server: ServerFactory, fake: FakeDrive, env: dict[str, str], tmp_path: Path
) -> None:
    image = fake.add_file("photo.png", b"\x89PNG")
    async with mcp_server(env) as session:
        unsupported = error(
            await session.call_tool(
                "sharepoint_pdf",
                {
                    "drive_id": DRIVE_ID,
                    "item_id": image.id,
                    "destination": str(tmp_path / "p.pdf"),
                },
            )
        )
        missing = error(
            await session.call_tool(
                "sharepoint_get", {"drive_id": DRIVE_ID, "item_id": "nope"}
            )
        )
        other_host = error(
            await session.call_tool(
                "sharepoint_resolve", {"url": "https://evil.example.com/x"}
            )
        )
        forbidden_host = error(
            await session.call_tool(
                "sharepoint_resolve", {"url": "https://other.sharepoint.com/sites/x"}
            )
        )

    assert unsupported.code == "invalid_argument" and unsupported.detail
    assert unsupported.detail["reason"] == "unsupported_conversion"
    assert "docx" in unsupported.detail["supported_extensions"]
    assert missing.code == "invalid_argument" and missing.detail
    assert missing.detail["reason"] == "not_found"
    for refused in (other_host, forbidden_host):
        assert refused.code == "invalid_argument" and refused.detail
        assert refused.detail["reason"] == "host_not_allowed"
        assert refused.detail["allowed_hosts"] == ["csuses.sharepoint.com"]
    assert fake.calls("GET", "/shares/") == []


async def test_a_403_is_access_denied_and_a_write_says_nothing_was_applied(
    mcp_server: ServerFactory, fake: FakeDrive, env: dict[str, str], tmp_path: Path
) -> None:
    folder = fake.add_folder("Locked down")
    source = tmp_path / "a.txt"
    source.write_bytes(b"x")
    fake.script["GET children"] = ["403"]
    fake.script["GET item"] = ["403"]
    async with mcp_server(env) as session:
        read = error(
            await session.call_tool(
                "sharepoint_list", {"drive_id": DRIVE_ID, "item_id": folder.id}
            )
        )
        write = error(
            await session.call_tool(
                "sharepoint_upload",
                {
                    "drive_id": DRIVE_ID,
                    "parent_id": folder.id,
                    "source_path": str(source),
                },
            )
        )
    for failure in (read, write):
        assert failure.code == "access_denied" and failure.detail
        assert failure.detail["reason"] == "insufficient_permission"
    assert write.detail and write.detail["applied"] is False
    assert fake.calls("POST", "createUploadSession") == []


# -- honest write outcomes through the protocol --------------------------------------


async def test_an_uncertain_upload_is_an_error_with_outcome_observation_and_no_retry(
    mcp_server: ServerFactory,
    fake: FakeDrive,
    env: dict[str, str],
    tmp_path: Path,
    errlog: Any,
) -> None:
    folder = fake.add_folder("Contacts")
    source = tmp_path / "list.xlsx"
    source.write_bytes(b"PK list")
    fake.script["PUT upload"] = ["apply-then-500"]

    async with mcp_server(env, errlog=errlog) as session:
        result = await session.call_tool(
            "sharepoint_upload",
            {"drive_id": DRIVE_ID, "parent_id": folder.id, "source_path": str(source)},
        )

    failure = error(result)
    assert failure.code == "upstream_error" and failure.detail
    detail = failure.detail
    assert detail["outcome"] == "write_uncertain" and detail["retry_safe"] is False
    assert detail["observed"]["state"] == "found_identical"
    assert detail["observed"]["item"]["name"] == "list.xlsx"
    assert "Do NOT repeat" in detail["next_step"]
    assert len(fake.calls("PUT", "/upload/")) == 1
    assert len(fake.children_of(folder.id)) == 1
    assert_no_secret(fake, json.dumps(failure.model_dump()), errlog.read())


async def test_a_readback_mismatch_is_an_error_that_keeps_the_item_identity(
    mcp_server: ServerFactory, fake: FakeDrive, env: dict[str, str], tmp_path: Path
) -> None:
    folder = fake.add_folder("Contacts")
    source = tmp_path / "list.xlsx"
    source.write_bytes(b"PK list")
    fake.corrupt_commit = True

    async with mcp_server(env) as session:
        result = await session.call_tool(
            "sharepoint_upload",
            {"drive_id": DRIVE_ID, "parent_id": folder.id, "source_path": str(source)},
        )

    failure = error(result)
    assert failure.detail
    assert failure.detail["outcome"] == "verification_failed"
    assert failure.detail["item"]["name"] == "list.xlsx"
    assert failure.detail["mismatches"][0].startswith("content")
    assert failure.detail["retry_safe"] is False
    assert "sharepoint_get" in failure.detail["next_step"]
