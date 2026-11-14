"""Deterministic mechanics of ``jake_tools.artifacts``.

The Claude query is the one external boundary and is faked here with the
Artifact tool's *recorded* result texts (``tests/artifact_fixtures``). That
proves parsing, auditing and byte comparison; it does not prove Claude Code
still behaves this way: ``test_artifacts_live.py`` (``--slow``, ``live``) does.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterable, AsyncIterator, Callable
from pathlib import Path
from typing import Any

import click.testing
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from click.testing import CliRunner

from jake_tools.artifacts import SourceError, publish, read
from jake_tools.artifacts.parse import (
    hosted_html,
    parse_header,
    parse_published,
    remove_shell,
)
from jake_tools.claude import ClaudeAgent
from jake_tools.operations.errors import ToolError

FIXTURES = Path(__file__).parent / "artifact_fixtures"
SOURCE = (FIXTURES / "probe.html").read_bytes()
PUBLISH_TEXT = (FIXTURES / "publish_result.txt").read_text()
READ_TEXT = (FIXTURES / "read_result.txt").read_text()
URL = "https://claude.ai/artifact/V4iHumJLsDrW9EBEtZfqZj"
VERSION_ID = "1794664145-2330"


def _use(name: str, **input: Any) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id=f"tu-{name}", name=name, input=input)],
        model="haiku",
    )


def _use_id(use_id: str, name: str, **input: Any) -> AssistantMessage:
    return AssistantMessage(
        content=[ToolUseBlock(id=use_id, name=name, input=input)], model="haiku"
    )


def _result_of(use_id: str, text: str, *, is_error: bool = False) -> UserMessage:
    return UserMessage(
        content=[ToolResultBlock(tool_use_id=use_id, content=text, is_error=is_error)]
    )


def _init(tools: list[str]) -> SystemMessage:
    return SystemMessage(subtype="init", data={"session_id": "s", "tools": tools})


def _end() -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="s",
        total_cost_usd=0.01,
        usage=None,
        result="done",
    )


class FakeClaude:
    """A fake ``query``: one scripted stream per session kind.

    It plays the CLI's part towards the guard, as the real runtime was seen to:
    before each scripted tool call it runs the ``PreToolUse`` hook and the
    hook the production code installed (it can be
    switched off to model a hook that does not run), and a deny from either
    replaces that call's scripted result with the CLI's refusal. A call whose
    scripted result is a ``<tool_use_error>`` never reaches either (the tool
    validates first). The guard, the wiring and everything after are production
    code."""

    def __init__(
        self,
        *,
        publish_stream: Callable[[Path], list[Message]] | None = None,
        read_stream: Callable[[], list[Message]] | None = None,
        on_publish: Callable[[], None] | None = None,
        hook_layer: bool = True,
        permission_layer: bool = True,
        rewrite_for_permission: Callable[[dict[str, Any]], dict[str, Any]]
        | None = None,
        fail_after_stream: bool = False,
    ) -> None:
        self.publish_stream = publish_stream or self._default_publish
        self.read_stream = read_stream or self._default_read
        self.on_publish = on_publish
        self.hook_layer = hook_layer
        self.fail_after_stream = fail_after_stream
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    @staticmethod
    def _default_publish(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id("pub", "Artifact", action="publish", file_path=str(path)),
            _result_of("pub", PUBLISH_TEXT),
            _end(),
        ]

    @staticmethod
    def _default_read() -> list[Message]:
        return [
            _init(["Artifact"]),
            _use_id("rd", "Artifact", action="read", url=URL),
            _result_of("rd", READ_TEXT),
            _end(),
        ]

    def agent(self) -> ClaudeAgent:
        return ClaudeAgent(run_query=self)  # type: ignore[arg-type]

    async def _denial(
        self, block: ToolUseBlock, options: ClaudeAgentOptions
    ) -> str | None:
        """The reason the production layers would refuse ``block``, if any."""
        if self.hook_layer:
            for matcher in (options.hooks or {}).get("PreToolUse", []):
                for hook in matcher.hooks:
                    out = await hook(
                        {  # type: ignore[arg-type]
                            "hook_event_name": "PreToolUse",
                            "tool_name": block.name,
                            "tool_input": block.input,
                            "tool_use_id": block.id,
                        },
                        block.id,
                        None,  # type: ignore[arg-type]
                    )
                    decision = out.get("hookSpecificOutput", {})
                    if decision.get("permissionDecision") == "deny":
                        return decision.get("permissionDecisionReason", "")
        return None

    async def _session(
        self, prompt: str | AsyncIterable[dict[str, Any]], options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        if not isinstance(prompt, str):  # can_use_tool needs streaming input
            prompt = "".join([m["message"]["content"] async for m in prompt])
        self.calls.append((prompt, options))
        if prompt.startswith("Publish"):
            path = Path(prompt.split("Read tool: ", 1)[1].split("\n", 1)[0])
            if self.on_publish is not None:
                self.on_publish()
            messages = self.publish_stream(path)
        else:
            messages = self.read_stream()
        invalid = {
            b.tool_use_id
            for m in messages
            if isinstance(m, UserMessage) and isinstance(m.content, list)
            for b in m.content
            if isinstance(b, ToolResultBlock)
            and str(b.content).startswith("<tool_use_error>")
        }
        denied: dict[str, str] = {}
        for message in messages:
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock) and block.id not in invalid:
                        reason = await self._denial(block, options)
                        if reason is not None:
                            denied[block.id] = reason
            if isinstance(message, UserMessage) and isinstance(message.content, list):
                block = message.content[0]
                if isinstance(block, ToolResultBlock) and block.tool_use_id in denied:
                    yield _result_of(
                        block.tool_use_id,
                        f"PreToolUse:Artifact hook error: {denied[block.tool_use_id]}",
                        is_error=True,
                    )
                    continue
            yield message
        if self.fail_after_stream:
            raise RuntimeError("the CLI exited unexpectedly")

    def __call__(
        self,
        *,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: ClaudeAgentOptions,
    ) -> AsyncIterator[Message]:
        return self._session(prompt, options)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "report.html"
    path.write_bytes(SOURCE)
    return path


# -- parsing the recorded tool results ----------------------------------------


def test_recorded_publish_result_yields_the_identity() -> None:
    published = parse_published(PUBLISH_TEXT)
    assert published is not None
    assert published.url == URL
    assert (published.version, published.version_id) == (1, VERSION_ID)


def test_recorded_read_result_reports_owner_visibility_and_version() -> None:
    header = parse_header(READ_TEXT)
    assert header is not None
    assert header.artifact_id == "e3448f84-1ca6-4757-9dd4-3bb5616d043a"
    assert header.version_id == VERSION_ID
    assert header.owned_by_you is True
    assert header.visibility == "private"


def test_hosted_file_is_the_source_inside_a_fixed_shell() -> None:
    html = hosted_html(READ_TEXT)
    assert html is not None
    # The whole hosted response is NOT the file: a 531-byte shell surrounds it.
    assert html.encode() != SOURCE
    assert remove_shell(html) == SOURCE.decode()


def test_unrecognised_shapes_fail_closed() -> None:
    assert parse_published("Published nothing") is None
    assert parse_header("not a header\n") is None
    assert parse_header("[Artifact x (version y) — owned by you, private]") is None
    assert hosted_html("no tags here") is None
    assert remove_shell("<html>plain</html>") is None
    assert remove_shell("<!doctype html><html><head>no body marker") is None
    assert remove_shell("<!doctype html><html><head></head><body>\nx") is None


def test_header_of_someone_elses_artifact_has_no_visibility() -> None:
    other = READ_TEXT.replace("owned by you, private", "owned by someone else")
    header = parse_header(other)
    assert header is not None
    assert header.owned_by_you is False
    assert header.visibility is None


def test_source_closing_tag_inside_the_file_does_not_end_the_content() -> None:
    inner = "<p>&lt;/cowritten-artifact-html&gt; raw: </cowritten-artifact-html></p>"
    text = (
        f"[h]\n<cowritten-artifact-html>\n{inner}\n</cowritten-artifact-html>\n\ntail"
    )
    assert hosted_html(text) == inner


# -- arguments --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "content", "fragment"),
    [
        ("r.txt", b"x", ".html"),
        ("r.html", b"", "1.."),
        ("r.html", b"\xff\xfe", "UTF-8"),
        ("r.html", "a�b".encode(), "U\\+FFFD"),
    ],
)
async def test_bad_files_are_rejected_before_any_session(
    tmp_path: Path, name: str, content: bytes, fragment: str
) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    fake = FakeClaude()
    with pytest.raises(SourceError, match=fragment):
        await publish(str(path), agent=fake.agent())
    assert fake.calls == []


async def test_relative_and_missing_paths_are_rejected(tmp_path: Path) -> None:
    fake = FakeClaude()
    with pytest.raises(SourceError, match="absolute"):
        await publish("report.html", agent=fake.agent())
    with pytest.raises(SourceError, match="existing file"):
        await publish(str(tmp_path / "gone.html"), agent=fake.agent())
    with pytest.raises(SourceError, match="url must look like"):
        await read("https://example.com/artifact/x", agent=fake.agent())
    assert fake.calls == []


# -- publish ------------------------------------------------------------------------


async def test_publish_reports_a_hosted_difference(source: Path) -> None:
    changed = READ_TEXT.replace("Version 1.", "Version 9.")
    fake = FakeClaude(
        read_stream=lambda: [
            _init(["Artifact"]),
            _use_id("rd", "Artifact", action="read", url=URL),
            _result_of("rd", changed),
            _end(),
        ]
    )
    result = await publish(str(source), agent=fake.agent())
    assert result.verified is False
    assert "the hosted file differs from the local file" in result.problems
    assert result.url == URL  # the artifact exists and is not hidden


async def test_publish_does_not_confirm_a_shared_artifact(source: Path) -> None:
    shared = READ_TEXT.replace(
        "owned by you, private", "owned by you, shared with your organization"
    )
    fake = FakeClaude(
        read_stream=lambda: [
            _init(["Artifact"]),
            _use_id("rd", "Artifact", action="read", url=URL),
            _result_of("rd", shared),
            _end(),
        ]
    )
    result = await publish(str(source), agent=fake.agent())
    assert result.private_confirmed is False and result.verified is False
    assert result.readback.visibility == "shared with your organization"


async def test_unrecognised_shell_makes_no_content_claim(source: Path) -> None:
    odd = READ_TEXT.replace("<!doctype html><html><head>", "<html><head>", 1)
    fake = FakeClaude(
        read_stream=lambda: [
            _init(["Artifact"]),
            _use_id("rd", "Artifact", action="read", url=URL),
            _result_of("rd", odd),
            _end(),
        ]
    )
    result = await publish(str(source), agent=fake.agent())
    assert result.verified is False
    assert result.readback.shell_removed is False
    assert result.readback.sha256 is None


async def test_a_failed_readback_keeps_the_published_identity(source: Path) -> None:
    fake = FakeClaude(read_stream=lambda: [_init(["Artifact"]), _end()])
    result = await publish(str(source), agent=fake.agent())
    assert result.url == URL and result.verified is False
    assert any("readback failed" in p for p in result.problems)


async def test_local_file_changing_during_the_call_is_reported(source: Path) -> None:
    def change() -> None:
        source.write_bytes(SOURCE + b"<!-- x -->")

    fake = FakeClaude(on_publish=change)
    result = await publish(str(source), agent=fake.agent())
    assert result.verified is False
    assert "the local file changed while the call ran" in result.problems


async def test_missing_tool_is_reported_without_a_publish(source: Path) -> None:
    fake = FakeClaude(publish_stream=lambda path: [_init(["Read"]), _end()])
    with pytest.raises(ToolError) as caught:
        await publish(str(source), agent=fake.agent())
    assert caught.value.code == "upstream_error"
    assert caught.value.detail is not None
    assert caught.value.detail["reason"] == "tool_not_exposed"
    assert caught.value.detail["publish_attempted"] is False
    assert len(fake.calls) == 1


# -- read ------------------------------------------------------------------------------


async def test_read_compares_version_and_file_and_saves_the_hosted_file(
    source: Path, tmp_path: Path
) -> None:
    out = tmp_path / "hosted.html"
    result = await read(
        URL,
        file_path=str(source),
        expected_version_id=VERSION_ID,
        output_path=str(out),
        agent=FakeClaude().agent(),
    )
    assert result.matches_file is True and result.version_matches is True
    assert result.private is True
    assert out.read_bytes() == SOURCE  # exact bytes, shell removed
    assert result.saved_to == str(out)


async def test_read_reports_a_different_file_and_version(
    source: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other.html"
    other.write_bytes(SOURCE + b"x")
    result = await read(
        URL,
        file_path=str(other),
        expected_version_id="999-aaaa",
        agent=FakeClaude().agent(),
    )
    assert result.matches_file is False and result.version_matches is False


async def test_read_refuses_to_overwrite_and_never_reads_first(tmp_path: Path) -> None:
    existing = tmp_path / "there.html"
    existing.write_text("keep")
    fake = FakeClaude()
    with pytest.raises(SourceError, match="new file"):
        await read(URL, output_path=str(existing), agent=fake.agent())
    assert existing.read_text() == "keep" and fake.calls == []


async def test_read_of_another_url_is_not_accepted() -> None:
    fake = FakeClaude(
        read_stream=lambda: [
            _init(["Artifact"]),
            _use_id(
                "rd", "Artifact", action="read", url="https://claude.ai/artifact/Z"
            ),
            _result_of("rd", READ_TEXT),
            _end(),
        ]
    )
    with pytest.raises(ToolError, match="no readback for this URL"):
        await read(URL, agent=fake.agent())


# -- publish: authority is enforced before calls, not audited after ----------------


async def test_publish_verifies_the_hosted_bytes_in_a_separate_session(
    source: Path,
) -> None:
    fake = FakeClaude()
    result = await publish(str(source), agent=fake.agent())

    assert result.verified is True and result.problems == []
    assert result.url == URL
    assert result.artifact_id == "e3448f84-1ca6-4757-9dd4-3bb5616d043a"
    assert (result.version, result.version_id) == (1, VERSION_ID)
    assert result.private_confirmed is True
    assert result.readback.visibility == "private"
    assert result.source.sha256 == hashlib.sha256(SOURCE).hexdigest()
    assert result.readback.sha256 == result.source.sha256
    assert result.readback.bytes == len(SOURCE)
    assert result.denied_before_running == []

    (publish_prompt, publish_options), (read_prompt, read_options) = fake.calls
    for options in (publish_options, read_options):
        # Process-local, specific: the one undocumented switch and no auto-open.
        assert options.env == {
            "CLAUDE_CODE_ARTIFACT": "1",
            "CLAUDE_CODE_ARTIFACT_AUTO_OPEN": "0",
        }
        assert options.strict_mcp_config is True  # no unrelated connectors
        assert options.permission_mode == "dontAsk"
        assert options.can_use_tool is None  # hook-only: the proven boundary
        # User, project and managed policy stay in force: never suppressed.
        assert options.setting_sources is None
        # The guard is wired as a PreToolUse hook on every call.
        assert [m.matcher for m in options.hooks["PreToolUse"]] == [None]  # type: ignore[index]
        # Neutral working directory, not the file's (no project settings).
        assert options.cwd != source.parent
    assert publish_options.tools == ["Read", "Artifact"]
    assert read_options.tools == ["Artifact"]
    assert URL in read_prompt and str(source) in publish_prompt


async def test_hostile_calls_are_denied_before_they_run(source: Path) -> None:
    """Steered by hosted content or a tool result, a model might try these."""

    def stream(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id("l", "Artifact", action="list"),
            _result_of("l", "Showing 25 of 48 (EXECUTED)"),
            _use_id("s", "Artifact", action="share", url=URL),
            _result_of("s", "shared (EXECUTED)"),
            _use_id("d", "Artifact", action="delete", url=URL),
            _result_of("d", "deleted (EXECUTED)"),
            _use_id("r", "Read", file_path="/etc/passwd"),
            _result_of("r", "root:x (EXECUTED)"),
            _use_id("u", "Artifact", action="publish", file_path=str(path), url=URL),
            _result_of("u", "updated (EXECUTED)"),
            _use_id("e", "Artifact", action="publish", file_path=str(path), edits=[]),
            _result_of("e", "edited (EXECUTED)"),
            _use_id("o", "Artifact", action="publish", file_path="/etc/other.html"),
            _result_of("o", "published other (EXECUTED)"),
            _use_id("pub", "Artifact", action="publish", file_path=str(path)),
            _result_of("pub", PUBLISH_TEXT),
            _use_id("again", "Artifact", action="publish", file_path=str(path)),
            _result_of("again", PUBLISH_TEXT.replace("V4iHum", "Second")),
            _end(),
        ]

    fake = FakeClaude(publish_stream=stream)
    result = await publish(str(source), agent=fake.agent())

    assert result.url == URL and result.verified is True
    assert len(result.denied_before_running) == 8  # everything but the one publish
    joined = " ".join(result.denied_before_running)
    for fragment in (
        "'list'",
        "'share'",
        "'delete'",
        "may be read",
        "url is not permitted",
        "edits is not permitted",
        "may be published",
        "only one publish",
    ):
        assert fragment in joined, fragment
    assert len(fake.calls) == 2  # one publish session, one read session


async def test_a_read_session_has_no_write_authority_and_no_file_access() -> None:
    fake = FakeClaude(
        read_stream=lambda: [
            _init(["Artifact"]),
            _use_id("x", "Artifact", action="delete", url=URL),
            _result_of("x", "deleted (EXECUTED)"),
            _use_id("p", "Artifact", action="publish", file_path="/tmp/x.html"),
            _result_of("p", "published (EXECUTED)"),
            _use_id("f", "Read", file_path="/etc/passwd"),
            _result_of("f", "root:x (EXECUTED)"),
            _use_id("rd", "Artifact", action="read", url=URL),
            _result_of("rd", READ_TEXT),
            _end(),
        ]
    )
    result = await read(URL, agent=fake.agent())
    assert result.private is True and result.shell_removed is True


async def test_a_read_of_another_artifact_is_denied_not_used() -> None:
    fake = FakeClaude(
        read_stream=lambda: [
            _init(["Artifact"]),
            _use_id(
                "rd",
                "Artifact",
                action="read",
                url="https://claude.ai/artifact/Qm3Kz7Tn5Rv9Wx2YbDfGhj",
            ),
            _result_of("rd", READ_TEXT),
            _end(),
        ]
    )
    with pytest.raises(ToolError, match="no readback for this URL") as caught:
        await read(URL, agent=fake.agent())
    assert caught.value.detail is not None
    assert caught.value.detail["denied_by_guard"]


async def test_publish_with_a_wrong_file_is_refused_with_a_safe_retry(
    source: Path,
) -> None:
    def stream(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id("pub", "Artifact", action="publish", file_path="/etc/other.html"),
            _result_of("pub", PUBLISH_TEXT),
            _end(),
        ]

    fake = FakeClaude(publish_stream=stream)
    with pytest.raises(ToolError) as caught:
        await publish(str(source), agent=fake.agent())
    assert "denied before it ran" in caught.value.message
    assert caught.value.detail is not None
    assert caught.value.detail["retry_safe"] is True
    assert caught.value.detail["publish_allowed"] is False
    assert len(fake.calls) == 1


async def test_a_call_the_hook_never_saw_is_detected_not_enforced(source: Path) -> None:
    """Detection only: if the hook did not run, the call was NOT stopped. The
    result is marked unverified, but the effect may already have happened; this
    audit is not a substitute for the hook."""
    fake = FakeClaude(hook_layer=False)
    result = await publish(str(source), agent=fake.agent())
    assert result.verified is False
    assert any("not checked by the guard" in p for p in result.problems)
    with pytest.raises(ToolError, match="not checked by the guard"):
        await read(URL, agent=fake.agent())


async def test_a_tool_refused_call_is_not_counted_as_unguarded(source: Path) -> None:
    """CLI 2.1.318 validates input first; such a call never reaches the hook."""

    def stream(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id("bad", "Artifact", action="read", url="not-a-url"),
            _result_of(
                "bad",
                "<tool_use_error>not an artifact URL: x</tool_use_error>",
                is_error=True,
            ),
            *FakeClaude._default_publish(path)[1:],
        ]

    fake = FakeClaude(publish_stream=stream)
    result = await publish(str(source), agent=fake.agent())
    assert result.verified is True


# -- publish: outcomes that are not a clean success ---------------------------------


async def test_an_allowed_publish_that_returns_no_url_has_an_unknown_outcome(
    source: Path,
) -> None:
    def stream(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id("pub", "Artifact", action="publish", file_path=str(path)),
            _result_of("pub", "Publish refused: nope", is_error=True),
            _end(),
        ]

    fake = FakeClaude(publish_stream=stream)
    with pytest.raises(ToolError) as caught:
        await publish(str(source), agent=fake.agent())
    assert "UNKNOWN" in caught.value.message
    detail = caught.value.detail
    assert detail is not None
    assert detail["publish_attempted"] is True and detail["publish_allowed"] is True
    assert detail["retry_safe"] is False
    assert len(fake.calls) == 1  # one attempt, never retried


async def test_a_crash_before_the_result_reports_urls_seen(source: Path) -> None:
    def stream(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id("pub", "Artifact", action="publish", file_path=str(path)),
            AssistantMessage(content=[TextBlock(text=f"I made {URL}")], model="haiku"),
        ]

    fake = FakeClaude(publish_stream=stream, fail_after_stream=True)
    with pytest.raises(ToolError) as caught:
        await publish(str(source), agent=fake.agent())
    detail = caught.value.detail
    assert detail is not None and detail["retry_safe"] is False
    assert detail["urls_seen"] == [URL]
    assert "UNKNOWN" in caught.value.message
    assert "exited unexpectedly" in caught.value.message


async def test_a_crash_after_the_publish_keeps_the_receipt(source: Path) -> None:
    fake = FakeClaude(fail_after_stream=True)
    # Both sessions end by raising after delivering their results.
    result = await publish(str(source), agent=fake.agent())
    assert result.url == URL and result.version_id == VERSION_ID
    assert result.verified is False
    assert any("ended abnormally after publishing" in p for p in result.problems)
    assert result.readback.sha256 == result.source.sha256  # the readback survived too


async def test_the_local_file_vanishing_keeps_the_receipt(source: Path) -> None:
    fake = FakeClaude(on_publish=lambda: None)

    def stream(path: Path) -> list[Message]:
        path.unlink()
        return FakeClaude._default_publish(path)

    fake.publish_stream = stream
    result = await publish(str(source), agent=fake.agent())
    assert result.url == URL and result.verified is False
    assert any("could not be re-read" in p for p in result.problems)


async def test_an_expected_version_must_be_observed() -> None:
    from jake_tools.artifacts.models import Readback
    from jake_tools.artifacts.service import _compare, receipt

    unseen = Readback(url=URL, shell_removed=True, private=True, sha256="x")
    problems = _compare(
        unseen, receipt(Path("/x.html"), b"x"), version_id="1794664145-2330"
    )
    assert any("live version is None" in p for p in problems)


async def test_read_without_a_version_in_the_header_does_not_match() -> None:
    headerless = READ_TEXT.replace("(version 1794664145-2330) ", "")
    fake = FakeClaude(
        read_stream=lambda: [
            _init(["Artifact"]),
            _use_id("rd", "Artifact", action="read", url=URL),
            _result_of("rd", headerless),
            _end(),
        ]
    )
    result = await read(URL, expected_version_id=VERSION_ID, agent=fake.agent())
    assert result.version_matches is False


# -- the CLI: failed checks and uncertain writes are not exit 0 ----------------------


def _cli(
    monkeypatch: pytest.MonkeyPatch, fake: FakeClaude, *args: str
) -> click.testing.Result:
    from jake_tools.artifacts import service
    from jake_tools.cli import main

    monkeypatch.setattr(service, "ClaudeAgent", lambda: fake.agent())
    return CliRunner().invoke(main, ["artifact", *args])


def test_cli_read_succeeds_when_file_and_version_match(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _cli(
        monkeypatch,
        FakeClaude(),
        "read",
        URL,
        "--file",
        str(source),
        "--expected-version-id",
        VERSION_ID,
        "--json",
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["matches_file"] is True and body["version_matches"] is True


def test_cli_read_fails_when_the_file_differs_but_prints_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other.html"
    other.write_bytes(SOURCE + b"x")
    result = _cli(
        monkeypatch, FakeClaude(), "read", URL, "--file", str(other), "--json"
    )
    assert result.exit_code != 0
    assert '"matches_file": false' in result.output
    assert "does not match --file" in result.output


def test_cli_read_fails_when_the_expected_version_is_not_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _cli(
        monkeypatch, FakeClaude(), "read", URL, "--expected-version-id", "999-aaaa"
    )
    assert result.exit_code != 0
    assert "version_matches: False" in result.output
    assert "not '999-aaaa'" in result.output


def test_cli_read_fails_when_the_comparison_cannot_be_made(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    odd = READ_TEXT.replace("<!doctype html><html>", "<html>", 1)
    fake = FakeClaude(
        read_stream=lambda: [
            _init(["Artifact"]),
            _use_id("rd", "Artifact", action="read", url=URL),
            _result_of("rd", odd),
            _end(),
        ]
    )
    result = _cli(monkeypatch, fake, "read", URL, "--file", str(source))
    assert result.exit_code != 0  # unknown is not a pass


def test_cli_publish_error_keeps_the_uncertain_write_evidence(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id("pub", "Artifact", action="publish", file_path=str(path)),
            AssistantMessage(content=[TextBlock(text=f"made {URL}")], model="haiku"),
        ]

    fake = FakeClaude(publish_stream=crash, fail_after_stream=True)
    for flags in (["--json"], []):
        result = _cli(monkeypatch, fake, "publish", str(source), *flags)
        assert result.exit_code != 0
        assert URL in result.output
        assert "retry_safe" in result.output and "false" in result.output.lower()
        assert "UNKNOWN" in result.output
    parsed = json.loads(
        _cli(monkeypatch, fake, "publish", str(source), "--json").output.split(
            "Error: ", 1
        )[-1]
    )
    assert parsed["detail"]["urls_seen"] == [URL]
    assert parsed["detail"]["retry_safe"] is False


def test_cli_publish_after_a_crash_prints_the_receipt_and_fails(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _cli(
        monkeypatch,
        FakeClaude(fail_after_stream=True),
        "publish",
        str(source),
        "--json",
    )
    assert result.exit_code != 0
    assert URL in result.output and '"verified": false' in result.output
    assert "NOT verified" in result.output


# -- F2: the tool resolves symlinks, so the guard accepts one exact canonical path --


def test_a_symlinked_source_is_published_by_its_canonical_path(tmp_path: Path) -> None:
    from jake_tools.artifacts.service import load_source

    real = tmp_path / "real"
    real.mkdir()
    (real / "page.html").write_bytes(SOURCE)
    link = tmp_path / "link"
    link.symlink_to(real)
    path, _ = load_source(str(link / "page.html"))
    assert path == (real / "page.html").resolve()


async def test_dot_dot_through_a_symlink_is_denied_not_normalised(
    tmp_path: Path,
) -> None:
    """``/x/link/../c.html`` is lexically ``/x/c.html`` but the tool follows
    ``link`` first, so it names a different file. Only the exact string passes."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "c.html").write_bytes(b"<p>not the source</p>")
    work = tmp_path / "x"
    work.mkdir()
    (work / "c.html").write_bytes(SOURCE)
    (work / "link").symlink_to(elsewhere)
    (tmp_path / "c.html").write_bytes(b"<p>what link/.. really is</p>")
    source = work / "c.html"
    sneaky = (
        f"{work}/link/../c.html"  # lexically the source; really elsewhere/../c.html
    )
    assert os.path.normpath(sneaky) == str(source)
    assert Path(sneaky).read_bytes() != SOURCE  # the OS resolves link first

    def stream(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id("pub", "Artifact", action="publish", file_path=sneaky),
            _result_of("pub", PUBLISH_TEXT + " (EXECUTED)"),
            _end(),
        ]

    fake = FakeClaude(publish_stream=stream)
    with pytest.raises(ToolError) as caught:
        await publish(str(source), agent=fake.agent())
    assert "denied before it ran" in caught.value.message
    assert caught.value.detail is not None and caught.value.detail["retry_safe"] is True


# -- F4: a local failure after a successful publish keeps the receipt ---------------


async def test_an_oserror_during_readback_setup_keeps_url_and_never_retries(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    from jake_tools.artifacts import service

    real = tempfile.TemporaryDirectory
    made = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        made["n"] += 1
        # 1: the publish session, 2: the readback session fails.
        if made["n"] == 2:
            raise OSError(28, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr(service.tempfile, "TemporaryDirectory", flaky)
    result = await publish(str(source), agent=FakeClaude().agent())
    assert result.url == URL and result.version_id == VERSION_ID
    assert result.verified is False and result.retry_safe is False
    assert any("readback failed" in p and "space" in p for p in result.problems)


async def test_an_oserror_before_any_publish_is_a_clean_safe_retry(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jake_tools.artifacts import service

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(service.tempfile, "TemporaryDirectory", broken)
    fake = FakeClaude()
    with pytest.raises(ToolError) as caught:
        await publish(str(source), agent=fake.agent())
    assert caught.value.detail is not None
    assert caught.value.detail["retry_safe"] is True
    assert caught.value.detail["publish_attempted"] is False
    assert fake.calls == []


async def test_a_model_supplied_private_key_is_denied_before_it_runs(
    source: Path,
) -> None:
    def stream(path: Path) -> list[Message]:
        return [
            _init(["Read", "Artifact"]),
            _use_id(
                "pin",
                "Artifact",
                action="publish",
                file_path=str(path),
                __artifactReadTargetPin="x",
            ),
            _result_of("pin", PUBLISH_TEXT + " (EXECUTED)"),
            _end(),
        ]

    fake = FakeClaude(publish_stream=stream)
    with pytest.raises(ToolError) as caught:
        await publish(str(source), agent=fake.agent())
    assert "denied before it ran" in caught.value.message
    assert caught.value.detail is not None and caught.value.detail["retry_safe"] is True
