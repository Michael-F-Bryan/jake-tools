"""The newsletter tools through the real MCP server over stdio.

The server is spawned under a hermetic HOME with only the configuration these
tests pass in. The one fake is Microsoft at its HTTP boundary
(``fake_microsoft.py``: token endpoint, Graph, SharePoint REST). Everything of
ours runs unmodified: argument validation, handlers, the shared token provider,
the client and the workflows. This proves protocol and mechanics, NOT live
Microsoft compatibility.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import anyio
import pytest
from fake_microsoft import (
    CLIENT_SECRET,
    GRAPH_RESOURCE,
    REFRESH_TOKEN,
    SHAREPOINT_RESOURCE,
    FakeMicrosoft,
)
from mcp import ClientSession
from mcp.types import TextResourceContents

from jake_tools.mcp.errors import ToolErrorPayload

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]
EnvFactory = Callable[..., dict[str, str]]
TOOLS = {"newsletter_list", "newsletter_get", "newsletter_create", "newsletter_update"}


@pytest.fixture
def fake() -> Iterator[FakeMicrosoft]:
    service = FakeMicrosoft().start()
    try:
        yield service
    finally:
        service.stop()


@pytest.fixture
def env(
    fake: FakeMicrosoft, mcp_env_factory: EnvFactory, mcp_home: Path
) -> dict[str, str]:
    return mcp_env_factory(mcp_home, **fake.env())


@pytest.fixture
def errlog(tmp_path: Path) -> Iterator[Any]:
    with (tmp_path / "server.stderr").open("w+") as handle:
        yield handle


def payload(result: Any) -> dict[str, Any]:
    assert result.structured_content is not None, result
    return result.structured_content


def error(result: Any) -> ToolErrorPayload:
    assert result.is_error is True, result.structured_content
    return ToolErrorPayload.model_validate(result.structured_content)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assert_no_secret(fake: FakeMicrosoft, *texts: str) -> None:
    secrets = {REFRESH_TOKEN, CLIENT_SECRET, *fake.issued, "rotated-refresh"}
    for text in texts:
        for secret in secrets:
            assert secret not in text, secret


# -- discovery ----------------------------------------------------------------


async def test_discovery_lists_the_tools_with_bounds_and_the_bundled_guidance(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        resource = await session.read_resource("skill://newsletters")

    assert set(tools) >= TOOLS
    schema = tools["newsletter_list"].input_schema["properties"]
    assert (schema["limit"]["minimum"], schema["limit"]["maximum"]) == (1, 100)
    assert tools["newsletter_get"].input_schema["required"] == ["entry_id"]
    assert tools["newsletter_get"].input_schema["properties"]["entry_id"]["pattern"]
    assert tools["newsletter_create"].input_schema["required"] == ["title", "body"]
    for name in TOOLS:
        description = " ".join((tools[name].description or "").split())
        assert (
            "Before using this tool, read the MCP resource `skill://newsletters`"
            in description
        )
        assert "recovery" in description or name in {
            "newsletter_list",
            "newsletter_get",
        }
    for name in ("newsletter_create", "newsletter_update"):
        assert "PUBLICATION ACTION" in (tools[name].description or "")
    content = resource.contents[0]
    assert isinstance(content, TextResourceContents)
    for needed in ("Publication boundary", "create_uncertain", "Never create again"):
        assert needed in content.text


async def test_without_configuration_each_tool_names_what_is_missing(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        result = await session.call_tool("newsletter_list", {})
        write = await session.call_tool(
            "newsletter_create", {"title": "t", "body": "b"}
        )

    for outcome in (result, write):
        failure = error(outcome)
        assert failure.code == "missing_credentials"
        assert failure.detail == {
            "variables": ["MICROSOFT_REFRESH_TOKEN"],
            "settings": [
                "microsoft.tenant_id",
                "microsoft.client_id",
                "microsoft.account",
            ],
        }
        assert "No other account is used" in failure.message


async def test_argument_bounds_keep_the_error_contract(
    mcp_server: ServerFactory, env: dict[str, str]
) -> None:
    async with mcp_server(env) as session:
        too_many = await session.call_tool("newsletter_list", {"limit": 101})
        bad_id = await session.call_tool("newsletter_get", {"entry_id": "1/../2"})
        empty = await session.call_tool("newsletter_update", {"entry_id": "1"})
        relative = await session.call_tool(
            "newsletter_create",
            {"title": "t", "body": "b", "attachments": ["relative.txt"]},
        )
    for outcome in (too_many, bad_id, empty, relative):
        assert error(outcome).code == "invalid_argument"


# -- reads --------------------------------------------------------------------


async def test_list_pages_with_explicit_coverage_and_get_reads_attachments(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str], errlog: Any
) -> None:
    for number in range(5):
        fake.add_entry(
            f"Entry {number}",
            f"<p>body {number}</p>",
            attachments={"a b.pdf": b"%PDF"} if number == 4 else None,
        )

    async with mcp_server(env, errlog=errlog) as session:
        pages = [payload(await session.call_tool("newsletter_list", {"limit": 2}))]
        while not pages[-1]["complete"]:
            pages.append(
                payload(
                    await session.call_tool(
                        "newsletter_list",
                        {"limit": 2, "cursor": pages[-1]["next_cursor"]},
                    )
                )
            )
        with_body = payload(
            await session.call_tool(
                "newsletter_list", {"limit": 1, "include_body": True}
            )
        )
        got = payload(await session.call_tool("newsletter_get", {"entry_id": "5"}))
        missing = error(await session.call_tool("newsletter_get", {"entry_id": "99"}))

    assert [p["count"] for p in pages] == [2, 2, 1]
    assert [p["complete"] for p in pages] == [False, False, True]
    assert pages[-1].get("next_cursor") is None
    assert [e["title"] for p in pages for e in p["entries"]] == [
        f"Entry {n}" for n in (4, 3, 2, 1, 0)
    ]
    assert pages[0]["bodies_included"] is False and pages[0]["entries"][0]["body"] == ""
    assert with_body["entries"][0]["body"] == "body 4"
    assert with_body["complete"] is False  # one of five: coverage is not implied
    assert got["id"] == "5" and got["attachments"][0]["name"] == "a b.pdf"
    assert missing.code == "invalid_argument" and missing.detail
    assert missing.detail["argument"] == "entry_id"
    assert_no_secret(fake, json.dumps(pages), json.dumps(got), errlog.read())


# -- create and update --------------------------------------------------------


async def test_create_publishes_reads_back_and_compares_attachment_bytes(
    mcp_server: ServerFactory,
    fake: FakeMicrosoft,
    env: dict[str, str],
    tmp_path: Path,
    errlog: Any,
) -> None:
    pdf = tmp_path / "doctrine.pdf"
    pdf.write_bytes(b"%PDF-1.7 " + bytes(range(200)))
    async with mcp_server(env, errlog=errlog) as session:
        result = await session.call_tool(
            "newsletter_create",
            {
                "title": "Vehicle phones",
                "body": "First paragraph.\n\nSecond <b>paragraph</b>.",
                "attachments": [str(pdf)],
            },
        )
        again = payload(await session.call_tool("newsletter_get", {"entry_id": "1"}))

    saved = payload(result)
    assert result.is_error is False
    assert saved["verified"] is True and saved["operation"] == "create"
    entry = saved["entry"]
    assert (entry["id"], entry["title"]) == ("1", "Vehicle phones")
    assert entry["body"] == "First paragraph.\n\nSecond <b>paragraph</b>."
    assert entry["url"].endswith("ID=1")
    [attachment] = entry["attachments"]
    assert attachment["name"] == "doctrine.pdf"
    assert attachment["size_bytes"] == pdf.stat().st_size
    assert attachment["sha256"] == sha(pdf.read_bytes())
    assert [e["status"] for e in saved["effects"]] == ["applied", "applied"]
    assert all(e["verified"] for e in saved["effects"])
    assert "attachment:doctrine.pdf (bytes)" in saved["compared"]
    assert fake.entries[1].attachments == {"doctrine.pdf": pdf.read_bytes()}
    assert again["title"] == "Vehicle phones"

    # Separate audiences: Graph and SharePoint each got their own token, and
    # each service saw only its own.
    graph_tokens = {r.token for r in fake.requests if r.path.startswith("/graph")}
    sp_tokens = {r.token for r in fake.requests if r.path.startswith("/sp")}
    assert len(graph_tokens) == len(sp_tokens) == 1 and graph_tokens != sp_tokens
    assert len(fake.refreshes(GRAPH_RESOURCE)) == 1
    assert len(fake.refreshes(SHAREPOINT_RESOURCE)) == 1
    assert_no_secret(fake, json.dumps(saved), errlog.read())


async def test_field_update_and_attachment_only_update_preserve_untouched_fields(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str], tmp_path: Path
) -> None:
    fake.add_entry(
        "Old title", "<p>Keep this body</p>", attachments={"old.txt": b"old"}
    )
    extra = tmp_path / "new.txt"
    extra.write_bytes(b"new bytes")
    async with mcp_server(env) as session:
        fields = payload(
            await session.call_tool(
                "newsletter_update", {"entry_id": "1", "title": "New title"}
            )
        )
        only_attachment = payload(
            await session.call_tool(
                "newsletter_update", {"entry_id": "1", "attachments": [str(extra)]}
            )
        )

    assert fields["verified"] and fields["entry"]["title"] == "New title"
    assert fields["entry"]["body"] == "Keep this body"
    assert "body unchanged" in fields["compared"]
    assert only_attachment["verified"]
    assert only_attachment["entry"]["title"] == "New title"
    assert only_attachment["entry"]["body"] == "Keep this body"
    assert [a["name"] for a in only_attachment["entry"]["attachments"]] == [
        "old.txt",
        "new.txt",
    ]
    assert {"title unchanged", "body unchanged"} <= set(only_attachment["compared"])
    # The attachment-only call wrote no field at all.
    assert len(fake.calls("PATCH", "/fields")) == 1
    assert fake.entries[1].attachments["new.txt"] == b"new bytes"


async def test_a_readback_mismatch_is_an_error_that_keeps_the_entry_identity(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str], tmp_path: Path
) -> None:
    fake.corrupt_downloads.add("a.bin")
    file = tmp_path / "a.bin"
    file.write_bytes(b"payload")
    async with mcp_server(env) as session:
        result = await session.call_tool(
            "newsletter_create",
            {"title": "T", "body": "B", "attachments": [str(file)]},
        )

    failure = error(result)
    assert failure.code == "upstream_error"
    assert failure.detail
    assert failure.detail["outcome"] == "verification_failed"
    assert failure.detail["entry_id"] == "1"
    assert failure.detail["mismatches"] == ["attachment:a.bin (bytes)"]
    assert failure.detail["retry_safe"] is False
    assert "newsletter_get" in failure.detail["next_step"]
    assert len(fake.creates()) == 1


async def test_a_failed_attachment_reports_effects_and_recovery_adds_no_duplicate(
    mcp_server: ServerFactory,
    fake: FakeMicrosoft,
    env: dict[str, str],
    tmp_path: Path,
    errlog: Any,
) -> None:
    first, second = tmp_path / "one.txt", tmp_path / "two.txt"
    first.write_bytes(b"1")
    second.write_bytes(b"2")
    fake.script["POST attachment-add:two.txt"] = ["500"]
    files = [str(first), str(second)]

    async with mcp_server(env, errlog=errlog) as session:
        partial = error(
            await session.call_tool(
                "newsletter_create",
                {"title": "T", "body": "B", "attachments": files},
            )
        )
        assert partial.detail
        entry_id = partial.detail["entry_id"]
        recovered = payload(
            await session.call_tool(
                "newsletter_update", {"entry_id": entry_id, "attachments": files}
            )
        )

    assert (
        partial.detail["outcome"] == "partial" and partial.detail["retry_safe"] is False
    )
    statuses = {e["subject"]: e["status"] for e in partial.detail["effects"]}
    assert statuses == {
        "entry": "applied",
        "one.txt": "applied",
        "two.txt": "uncertain",
    }
    assert "Do NOT create again" in partial.detail["next_step"]
    assert recovered["verified"] is True
    assert {e["subject"]: e["status"] for e in recovered["effects"]} == {
        "one.txt": "unchanged",
        "two.txt": "applied",
    }
    assert len(fake.entries) == 1 and len(fake.creates()) == 1
    assert fake.entries[1].attachments == {"one.txt": b"1", "two.txt": b"2"}
    assert_no_secret(fake, json.dumps(partial.detail), errlog.read())


async def test_an_uncertain_create_is_reported_and_never_retried(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str]
) -> None:
    fake.script["POST items"] = ["drop"]
    async with mcp_server(env) as session:
        failure = error(
            await session.call_tool("newsletter_create", {"title": "T", "body": "B"})
        )
    assert failure.detail and failure.detail["outcome"] == "create_uncertain"
    assert failure.detail["entry_id"] is None and failure.detail["retry_safe"] is False
    assert "Do NOT create again" in failure.detail["next_step"]
    assert len(fake.creates()) == 1


async def test_unusable_attachments_are_refused_before_anything_is_written(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str], tmp_path: Path
) -> None:
    bad = tmp_path / "bad#name.txt"
    bad.write_bytes(b"x")
    async with mcp_server(env) as session:
        missing = error(
            await session.call_tool(
                "newsletter_create",
                {"title": "T", "body": "B", "attachments": [str(tmp_path / "no.txt")]},
            )
        )
        illegal = error(
            await session.call_tool(
                "newsletter_create",
                {"title": "T", "body": "B", "attachments": [str(bad)]},
            )
        )
    assert missing.code == illegal.code == "invalid_argument"
    assert missing.detail and missing.detail["applied"] is False
    assert not fake.creates() and not fake.entries


# -- credentials: expiry, concurrency, failures -------------------------------


async def test_an_expiring_token_is_refreshed_inside_one_long_running_server(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str]
) -> None:
    fake.expires_in = 2  # refreshed from 1.6 s on; unusable after 2 s
    fake.add_entry("T", "<p>B</p>")
    async with mcp_server(env) as session:
        assert not (
            await session.call_tool("newsletter_get", {"entry_id": "1"})
        ).is_error
        assert not (
            await session.call_tool("newsletter_get", {"entry_id": "1"})
        ).is_error
        assert len(fake.refreshes(GRAPH_RESOURCE)) == 1  # reused while fresh
        await anyio.sleep(2.2)
        after = await session.call_tool("newsletter_get", {"entry_id": "1"})
        assert not after.is_error

    graph_tokens = [r.token for r in fake.requests if r.path.startswith("/graph")]
    assert len(set(graph_tokens)) == 2 and graph_tokens[0] != graph_tokens[-1]
    assert len(fake.refreshes(GRAPH_RESOURCE)) == 2


async def test_concurrent_calls_share_one_refresh_in_the_server(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str]
) -> None:
    fake.add_entry("T", "<p>B</p>")
    fake.token_delay = 0.4
    results: list[Any] = []

    async def call(session: ClientSession) -> None:
        results.append(await session.call_tool("newsletter_get", {"entry_id": "1"}))

    async with mcp_server(env) as session, anyio.create_task_group() as group:
        for _ in range(8):
            group.start_soon(call, session)

    assert len(results) == 8 and not any(r.is_error for r in results)
    assert len(fake.refreshes(GRAPH_RESOURCE)) == 1
    assert len(fake.refreshes(SHAREPOINT_RESOURCE)) == 1  # one per audience


@pytest.mark.parametrize(
    ("token_error", "status", "reason", "code"),
    [
        (
            {"error": "invalid_grant", "error_description": "AADSTS70043: revoked"},
            400,
            "grant_invalid",
            "access_denied",
        ),
        (
            {"error": "interaction_required"},
            400,
            "interaction_required",
            "access_denied",
        ),
        ({"error": "invalid_client"}, 401, "client_rejected", "access_denied"),
        ({"error": "invalid_resource"}, 400, "resource_not_granted", "access_denied"),
        ({"error": "temporarily_unavailable"}, 503, "unavailable", "upstream_error"),
    ],
)
async def test_token_failures_are_actionable_and_leak_nothing(
    mcp_server: ServerFactory,
    fake: FakeMicrosoft,
    env: dict[str, str],
    errlog: Any,
    token_error: dict[str, str],
    status: int,
    reason: str,
    code: str,
) -> None:
    fake.token_error, fake.token_status = token_error, status
    env["MICROSOFT_CLIENT_SECRET"] = CLIENT_SECRET
    async with mcp_server(env, errlog=errlog) as session:
        failure = error(await session.call_tool("newsletter_list", {}))

    assert failure.code == code
    assert failure.detail and failure.detail["reason"] == reason
    assert failure.detail["action"] and failure.detail["resource"] == GRAPH_RESOURCE
    assert_no_secret(fake, json.dumps(failure.model_dump()), errlog.read())
    assert "127.0.0.1" not in failure.message


async def test_a_grant_for_another_account_is_refused_not_used(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str]
) -> None:
    fake.id_claims = {**fake.id_claims, "preferred_username": "someone@example.test"}
    async with mcp_server(env) as session:
        failure = error(await session.call_tool("newsletter_list", {}))
    assert failure.code == "access_denied"
    assert failure.detail and failure.detail["reason"] == "identity_mismatch"
    assert not [r for r in fake.requests if r.path.startswith(("/graph", "/sp"))]


async def test_insufficient_permission_is_distinct_from_a_revoked_grant(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str], tmp_path: Path
) -> None:
    fake.forbid_graph_list = True
    fake.script["POST attachment-add:a.txt"] = ["403"]
    file = tmp_path / "a.txt"
    file.write_bytes(b"a")
    async with mcp_server(env) as session:
        listing = error(await session.call_tool("newsletter_list", {}))
        write = error(
            await session.call_tool(
                "newsletter_create",
                {"title": "T", "body": "B", "attachments": [str(file)]},
            )
        )

    assert listing.code == "access_denied"
    assert listing.detail and listing.detail["reason"] == "insufficient_permission"
    assert "does not switch accounts" in listing.message
    # After the entry was created, the SharePoint denial still carries its ID.
    assert write.code == "access_denied" and write.detail
    assert write.detail["reason"] == "insufficient_permission"
    assert write.detail["entry_id"] == "1" and write.detail["outcome"] == "partial"


async def test_the_server_never_prints_credentials_to_its_log_or_results(
    mcp_server: ServerFactory,
    fake: FakeMicrosoft,
    env: dict[str, str],
    tmp_path: Path,
    errlog: Any,
) -> None:
    fake.rotate = True
    fake.expires_in = 1
    env["MICROSOFT_CLIENT_SECRET"] = CLIENT_SECRET
    file = tmp_path / "f.txt"
    file.write_bytes(b"f")
    seen: list[str] = []
    async with mcp_server(env, errlog=errlog) as session:
        for _ in range(2):
            for name, args in (
                (
                    "newsletter_create",
                    {"title": "T", "body": "B", "attachments": [str(file)]},
                ),
                ("newsletter_list", {}),
                ("newsletter_get", {"entry_id": "99"}),
            ):
                result = await session.call_tool(name, args)
                seen.append(json.dumps(result.structured_content))
                seen.append(" ".join(str(c) for c in result.content))
            await anyio.sleep(1.2)
    assert fake.issued and len(fake.token_requests) > 2
    assert_no_secret(fake, *seen, errlog.read())


async def test_a_wrong_shaped_upstream_reply_is_an_upstream_error_with_its_evidence(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str]
) -> None:
    """Malformed upstream data is never reported as the caller's bad argument."""
    fake.script["POST items"] = ["apply-then-json:{}"]
    fake.script["GET items"] = ['apply-then-json:{"value": [{"id": 3}]}']
    async with mcp_server(env) as session:
        create = error(
            await session.call_tool("newsletter_create", {"title": "T", "body": "B"})
        )
        listing = error(await session.call_tool("newsletter_list", {}))
        fake.script["GET item"] = ['apply-then-json:{"id": 5}']
        get = error(await session.call_tool("newsletter_get", {"entry_id": "1"}))

    assert create.code == "upstream_error" and create.detail
    assert create.detail["outcome"] == "create_uncertain"
    assert create.detail["retry_safe"] is False and create.detail["entry_id"] is None
    assert len(fake.creates()) == 1
    for failure in (listing, get):
        assert failure.code == "upstream_error"
        assert "unexpected response shape" in failure.message


async def test_a_redirect_is_reported_and_never_followed_by_the_server(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str]
) -> None:
    fake.script["POST items"] = ["redirect:307"]
    async with mcp_server(env) as session:
        failure = error(
            await session.call_tool("newsletter_create", {"title": "T", "body": "B"})
        )
    assert failure.detail and failure.detail["outcome"] == "create_uncertain"
    assert len(fake.creates()) == 1
    assert not fake.calls("POST", "/redirected")


async def test_a_configured_object_id_pin_refuses_another_user_with_the_same_name(
    mcp_server: ServerFactory, fake: FakeMicrosoft, env: dict[str, str]
) -> None:
    fake.id_claims = {**fake.id_claims, "oid": "11111111-1111-1111-1111-111111111111"}
    env["JAKE_TOOLS_MICROSOFT_ACCOUNT_OBJECT_ID"] = (
        "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )
    async with mcp_server(env) as session:
        failure = error(await session.call_tool("newsletter_list", {}))
    assert failure.code == "access_denied"
    assert failure.detail and failure.detail["reason"] == "identity_mismatch"
    assert not [r for r in fake.requests if r.path.startswith(("/graph", "/sp"))]
