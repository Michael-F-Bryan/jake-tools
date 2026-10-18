import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    TextBlock,
)
from pydantic import BaseModel

from jake_tools.claude import (
    AgentSpec,
    ClaudeAgent,
    ClaudeAgentError,
    message_to_json,
)
from jake_tools.prompting import StructuredPrompt


class Minutes(BaseModel):
    summary: str
    actions: list[str]


class MinutesPrompt(StructuredPrompt[Minutes]):
    template = "Summarise:\n{{ notes }}"
    response_model = Minutes

    notes: str


def _result(
    *,
    structured_output: object = None,
    is_error: bool = False,
    subtype: str = "success",
    errors: list[str] | None = None,
    num_turns: int = 1,
    usage: dict[str, object] | None = None,
    total_cost_usd: float | None = None,
    result: str | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=10,
        duration_api_ms=8,
        is_error=is_error,
        num_turns=num_turns,
        session_id="session-1",
        total_cost_usd=total_cost_usd,
        usage=usage,
        result=result,
        structured_output=structured_output,
        errors=errors,
    )


class RecordingQuery:
    """A fake ``query`` that records its options and replays fixed messages."""

    def __init__(self, *messages: Message) -> None:
        self.messages = messages
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))

        async def stream() -> AsyncIterator[Message]:
            for message in self.messages:
                yield message

        return stream()

    @property
    def only_options(self) -> ClaudeAgentOptions:
        assert len(self.calls) == 1
        return self.calls[0][1]


class FailingQuery(RecordingQuery):
    """Replay messages, then reproduce the SDK's lossy stream exception."""

    def __init__(self, *messages: Message, error: str) -> None:
        super().__init__(*messages)
        self.error = error

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        stream = super().__call__(prompt=prompt, options=options)

        async def failing_stream() -> AsyncIterator[Message]:
            async for message in stream:
                yield message
            raise Exception(self.error)

        return failing_stream()


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model="claude-sonnet-5")


async def test_run_returns_concatenated_assistant_text() -> None:
    fake = RecordingQuery(_assistant("first"), _assistant("second"), _result())
    agent = ClaudeAgent(run_query=fake)

    reply = await agent.run("say something")

    assert reply.text == "first\nsecond"
    assert fake.calls[0][0] == "say something"


async def test_run_renders_a_prompt_object() -> None:
    fake = RecordingQuery(_assistant("ok"), _result())
    agent = ClaudeAgent(run_query=fake)

    await agent.run(MinutesPrompt(notes="hello there"))

    assert "hello there" in fake.calls[0][0]


async def test_run_structured_parses_structured_output() -> None:
    fake = RecordingQuery(
        _assistant("{...}"),
        _result(structured_output={"summary": "we shipped", "actions": ["ship more"]}),
    )
    agent = ClaudeAgent(run_query=fake)

    minutes, reply = await agent.run_structured(MinutesPrompt(notes="some notes"))

    assert minutes == Minutes(summary="we shipped", actions=["ship more"])
    assert reply.usage.api_calls == 1


async def test_run_structured_sends_the_response_model_schema() -> None:
    fake = RecordingQuery(_result(structured_output={"summary": "s", "actions": []}))
    agent = ClaudeAgent(run_query=fake)

    await agent.run_structured(MinutesPrompt(notes="some notes"))

    assert fake.only_options.output_format == {
        "type": "json_schema",
        "schema": Minutes.model_json_schema(),
    }


async def test_run_structured_without_structured_output_raises_with_context() -> None:
    fake = RecordingQuery(
        _assistant("sorry, no"),
        _result(
            subtype="error_during_execution",
            is_error=True,
            errors=["boom"],
            num_turns=2,
            total_cost_usd=0.5,
        ),
    )
    agent = ClaudeAgent(run_query=fake)

    with pytest.raises(ClaudeAgentError) as excinfo:
        await agent.run_structured(MinutesPrompt(notes="some notes"))

    message = str(excinfo.value)
    assert "Minutes" in message
    assert "error_during_execution" in message
    assert "boom" in message
    # The tokens already spent on this failed attempt must still be
    # chargeable, so they ride along on the exception.
    assert excinfo.value.usage.api_calls == 2
    assert excinfo.value.usage.estimated_cost_usd == 0.5


async def test_stream_without_a_result_message_raises() -> None:
    agent = ClaudeAgent(run_query=RecordingQuery(_assistant("dangling")))

    with pytest.raises(ClaudeAgentError, match="without a result"):
        await agent.run("say something")


async def test_rate_limit_message_survives_the_sdk_stream_exception() -> None:
    reset_message = "You've hit your session limit · resets 6:10pm (Australia/Perth)"
    rate_limit = AssistantMessage(
        content=[TextBlock(text=reset_message)],
        model="<synthetic>",
        error="rate_limit",
    )
    agent = ClaudeAgent(
        run_query=FailingQuery(
            rate_limit,
            error="Claude Code returned an error result: success",
        )
    )

    with pytest.raises(ClaudeAgentError) as excinfo:
        await agent.run("say something")

    assert str(excinfo.value) == f"Claude rate limit reached: {reset_message}"


def test_options_prefer_installed_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert AgentSpec().to_options().cli_path == str(cli)


def test_options_keep_sdk_discovery_without_installed_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    assert AgentSpec().to_options().cli_path is None


async def test_defaults_are_tool_less() -> None:
    fake = RecordingQuery(_result())

    await ClaudeAgent(run_query=fake).run("no tools please")

    # An empty list disables built-in tools; ``None`` would hand the agent
    # Claude Code's full default toolset.
    assert fake.only_options.tools == []
    assert fake.only_options.allowed_tools == []


async def test_granted_tools_are_also_auto_approved() -> None:
    fake = RecordingQuery(_result())
    agent = ClaudeAgent(run_query=fake)

    await agent.run(
        "read a file",
        AgentSpec(model="claude-opus-4-8", tools=("Read", "Grep")),
    )

    options = fake.only_options
    assert options.tools == ["Read", "Grep"]
    assert options.allowed_tools == ["Read", "Grep"]
    assert options.model == "claude-opus-4-8"


async def test_mcp_tools_are_approved_but_not_granted_as_builtins() -> None:
    fake = RecordingQuery(_result())
    agent = ClaudeAgent(run_query=fake)

    await agent.run(
        "search sessions",
        AgentSpec(tools=("Read",), mcp_tools=("mcp__sessions__search",)),
    )

    options = fake.only_options
    assert options.tools == ["Read"]
    assert options.allowed_tools == ["Read", "mcp__sessions__search"]


async def test_run_raises_on_an_error_result() -> None:
    fake = RecordingQuery(
        _result(
            is_error=True,
            subtype="error_during_execution",
            errors=["boom"],
            num_turns=2,
            total_cost_usd=0.5,
            usage={"input_tokens": 10, "output_tokens": 5},
        )
    )

    with pytest.raises(ClaudeAgentError) as excinfo:
        await ClaudeAgent(run_query=fake).run("go")

    message = str(excinfo.value)
    assert "error_during_execution" in message
    assert "boom" in message
    # Tokens already spent before the failure must not be lost off the
    # exception, so a caller that accounts cost can still charge them.
    assert excinfo.value.usage.api_calls == 2
    assert excinfo.value.usage.input_tokens == 10
    assert excinfo.value.usage.estimated_cost_usd == 0.5


async def test_run_raise_reports_the_subtype_when_no_errors_listed() -> None:
    fake = RecordingQuery(_result(is_error=True, subtype="error_max_turns"))

    with pytest.raises(ClaudeAgentError, match="error_max_turns"):
        await ClaudeAgent(run_query=fake).run("go")


async def test_error_result_subtype_and_text_ride_on_the_exception() -> None:
    """A caller telling a turn limit from a crash reads the field, not the prose."""
    fake = RecordingQuery(
        _result(is_error=True, subtype="error_max_budget_usd", result="partial")
    )

    with pytest.raises(ClaudeAgentError) as excinfo:
        await ClaudeAgent(run_query=fake).run("go")

    assert excinfo.value.result_subtype == "error_max_budget_usd"
    assert excinfo.value.final_text == "partial"


async def test_stream_failure_after_an_error_result_keeps_the_result() -> None:
    """The CLI exits non-zero after an error result; the SDK raises after
    yielding it. The result, not the exit, is what the caller needs."""
    fake = FailingQuery(
        _result(is_error=True, subtype="error_max_turns", num_turns=4),
        error="Claude Code returned an error result: error_max_turns",
    )

    with pytest.raises(ClaudeAgentError) as excinfo:
        await ClaudeAgent(run_query=fake).run("go")

    assert excinfo.value.result_subtype == "error_max_turns"
    assert excinfo.value.usage.api_calls == 4
    assert isinstance(excinfo.value.__cause__, Exception)


async def test_stream_failure_without_a_result_is_not_rewrapped() -> None:
    fake = FailingQuery(_assistant("partial"), error="connection lost")

    with pytest.raises(Exception, match="connection lost") as excinfo:
        await ClaudeAgent(run_query=fake).run("go")

    assert not isinstance(excinfo.value, ClaudeAgentError)


async def test_usage_is_taken_from_the_result_message() -> None:
    fake = RecordingQuery(
        _result(
            num_turns=3,
            total_cost_usd=0.25,
            usage={
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 20,
            },
        )
    )

    reply = await ClaudeAgent(run_query=fake).run("go")

    assert reply.usage.api_calls == 3
    assert reply.usage.input_tokens == 100
    assert reply.usage.output_tokens == 40
    assert reply.usage.cache_read_tokens == 900
    assert reply.usage.cache_write_tokens == 20
    assert reply.usage.total_tokens == 140
    assert reply.usage.estimated_cost_usd == 0.25


async def test_missing_usage_is_zeroed_rather_than_crashing() -> None:
    reply = await ClaudeAgent(run_query=RecordingQuery(_result())).run("go")

    assert reply.usage.total_tokens == 0
    assert reply.usage.estimated_cost_usd is None


class TestAgentSpecMerge:
    def test_none_override_keeps_the_base(self) -> None:
        base = AgentSpec(model="claude-opus-4-8", tools=("Read",))

        assert base.merge(None) == base

    def test_only_explicitly_set_fields_win(self) -> None:
        base = AgentSpec(model="claude-opus-4-8", tools=("Read", "Grep"))

        merged = base.merge(AgentSpec(model="claude-haiku-4-5"))

        # The override never mentioned tools, so the grant survives.
        assert merged.model == "claude-haiku-4-5"
        assert merged.tools == ("Read", "Grep")

    def test_an_explicit_empty_tool_grant_overrides(self) -> None:
        base = AgentSpec(tools=("Read",))

        merged = base.merge(AgentSpec(tools=()))

        assert merged.tools == ()


# --- AgentSpec isolation fields and the message observer ---------------------


def test_to_options_always_isolates_mcp_config_and_passes_through_defaults() -> None:
    options = AgentSpec().to_options()

    assert options.strict_mcp_config is True
    assert options.cwd is None
    assert options.env == {}
    assert options.setting_sources is None
    assert options.permission_mode is None


def test_to_options_forwards_cwd_env_setting_sources_and_permission_mode(
    tmp_path: Path,
) -> None:
    spec = AgentSpec(
        cwd=tmp_path,
        env={"EXAMPLE": "1"},
        setting_sources=(),
        permission_mode="dontAsk",
    )

    options = spec.to_options()

    assert options.cwd == tmp_path
    assert options.env == {"EXAMPLE": "1"}
    assert options.setting_sources == []
    assert options.permission_mode == "dontAsk"


def test_merge_only_overrides_fields_the_override_set() -> None:
    base = AgentSpec(tools=("Read",), setting_sources=(), cwd=Path("/base"))

    merged = base.merge(AgentSpec(model="other"))

    assert merged.model == "other"
    assert merged.tools == ("Read",)
    assert merged.setting_sources == ()
    assert merged.cwd == Path("/base")


async def test_reply_carries_the_result_messages_final_text() -> None:
    result = _result()
    result.result = "the final answer"
    fake = RecordingQuery(_assistant("thinking aloud"), result)
    agent = ClaudeAgent(run_query=fake)

    reply = await agent.run("go")

    assert reply.text == "thinking aloud"
    assert reply.final_text == "the final answer"


async def test_message_observer_sees_every_message_in_order() -> None:
    seen: list[Message] = []
    observer = seen.append
    fake = RecordingQuery(_assistant("one"), _assistant("two"), _result())
    agent = ClaudeAgent(run_query=fake).with_message_observer(observer)

    await agent.run("go")

    assert [type(message).__name__ for message in seen] == [
        "AssistantMessage",
        "AssistantMessage",
        "ResultMessage",
    ]
    # The observer survives the other builders.
    assert agent.for_stage("x").on_message is observer
    assert agent.with_defaults(AgentSpec()).on_message is observer


def test_message_to_json_tags_the_type_and_is_json_serialisable() -> None:
    payload = message_to_json(_assistant("hello"))

    assert payload["type"] == "AssistantMessage"
    assert payload["content"] == [{"text": "hello"}]
    assert payload["model"] == "claude-sonnet-5"
    assert json.loads(json.dumps(payload)) == payload

    result_payload = message_to_json(_result(total_cost_usd=0.5))
    assert result_payload["type"] == "ResultMessage"
    assert result_payload["total_cost_usd"] == 0.5


# --- the streaming conversation seam -------------------------------------------


class FakeStreamingClient:
    """A fake ``ClaudeSDKClient``: records what is sent, replays fixed messages.

    It stands at the SDK boundary only; ``ClaudeAgent.converse`` and
    ``Conversation`` run for real on top of it.
    """

    instances: list[FakeStreamingClient] = []

    def __init__(self, options: ClaudeAgentOptions, *messages: Message) -> None:
        self.options = options
        self.messages = messages
        self.sent: list[dict[str, object]] = []
        self.connected = False
        self.disconnected = False

    async def connect(self, prompt: object = None) -> None:
        self.connected = True

    async def query(self, prompt: object, session_id: str = "default") -> None:
        assert not isinstance(prompt, str)
        async for message in prompt:  # type: ignore[union-attr]
            self.sent.append(message)

    async def receive_messages(self) -> AsyncIterator[Message]:
        for message in self.messages:
            yield message

    async def disconnect(self) -> None:
        self.disconnected = True


def _connect_with(*messages: Message):
    made: list[FakeStreamingClient] = []

    def connect(options: ClaudeAgentOptions) -> FakeStreamingClient:
        client = FakeStreamingClient(options, *messages)
        made.append(client)
        return client

    return connect, made


async def test_converse_sends_tagged_user_messages_and_replays_them() -> None:
    from claude_agent_sdk import UserMessage

    connect, made = _connect_with(
        UserMessage(content="hello", uuid="u-1"), _result(result="hi")
    )
    agent = ClaudeAgent(connect=connect)

    async with agent.converse() as conversation:
        await conversation.send("hello", uuid="u-1")
        seen = [message async for message in conversation.messages()]

    client = made[0]
    assert client.connected and client.disconnected
    assert client.sent == [
        {
            "type": "user",
            "message": {"role": "user", "content": "hello"},
            "parent_tool_use_id": None,
            "uuid": "u-1",
        }
    ]
    # The CLI must echo user messages so a caller can see them land.
    assert client.options.extra_args == {"replay-user-messages": None}
    assert client.options.tools == []  # still genuinely tool-less by default
    assert len(seen) == 2


async def test_converse_disconnects_when_the_block_is_cancelled() -> None:
    import anyio

    connect, made = _connect_with()
    agent = ClaudeAgent(connect=connect)

    with anyio.move_on_after(0.05):
        async with agent.converse():
            await anyio.sleep(10)

    assert made[0].disconnected


async def test_converse_records_telemetry_per_result_and_observes_messages() -> None:
    from jake_tools.ai_usage import AICallTelemetry

    class Sink:
        def __init__(self) -> None:
            self.calls: list[AICallTelemetry] = []

        def record_call(self, record: AICallTelemetry) -> None:
            self.calls.append(record)

        def record_cache_hit(self, *, stage: str, model: str | None = None) -> None:
            raise AssertionError("unused")

    observed: list[Message] = []
    sink = Sink()
    connect, _ = _connect_with(
        _result(result="one", total_cost_usd=0.01),
        _result(is_error=True, subtype="error_max_turns"),
    )
    agent = (
        ClaudeAgent(connect=connect)
        .with_telemetry(sink)
        .with_message_observer(observed.append)
        .for_stage("delegated")
    )

    async with agent.converse() as conversation:
        _ = [message async for message in conversation.messages()]

    assert [call.status for call in sink.calls] == ["success", "error"]
    assert {call.stage for call in sink.calls} == {"delegated"}
    assert len(observed) == 2


def test_local_tools_are_served_in_process_and_auto_approved() -> None:
    from jake_tools.claude import LocalTool, LocalToolResult, local_tool_name

    async def handler(arguments: dict[str, object]) -> LocalToolResult:
        return LocalToolResult(text="ok")

    ask = LocalTool(
        name="ask_parent",
        description="Ask the parent.",
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}},
        handler=handler,
    )

    options = AgentSpec(tools=("Read",), local_tools=(ask,)).to_options()

    assert options.tools == ["Read"]  # built-ins unchanged
    assert options.allowed_tools == ["Read", local_tool_name("ask_parent")]
    assert local_tool_name("ask_parent") == "mcp__jake__ask_parent"
    assert isinstance(options.mcp_servers, dict)
    assert set(options.mcp_servers) == {"jake"}
    assert options.mcp_servers["jake"].get("type") == "sdk"  # in-process, not external
    assert options.strict_mcp_config is True


async def test_local_tool_handler_result_reaches_the_sdk_server() -> None:
    from mcp.types import CallToolRequest, CallToolRequestParams

    from jake_tools.claude import LocalTool, LocalToolResult

    received: list[dict[str, object]] = []

    async def handler(arguments: dict[str, object]) -> LocalToolResult:
        received.append(arguments)
        return LocalToolResult(text="no such choice", is_error=True)

    ask = LocalTool(
        name="ask_parent",
        description="Ask.",
        input_schema={"type": "object", "properties": {"question": {"type": "string"}}},
        handler=handler,
    )
    options = AgentSpec(local_tools=(ask,)).to_options()
    assert isinstance(options.mcp_servers, dict)
    server = options.mcp_servers["jake"]["instance"]  # type: ignore[typeddict-item]
    call = server.request_handlers[CallToolRequest]

    result = await call(
        CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name="ask_parent", arguments={"question": "?"}
            ),
        )
    )

    assert received == [{"question": "?"}]
    payload = result.root.model_dump()
    assert payload["isError"] is True
    assert payload["content"][0]["text"] == "no such choice"


def test_session_options_pass_through() -> None:
    fresh = AgentSpec(session_id="11111111-1111-4111-8111-111111111111").to_options()
    resumed = AgentSpec(resume="22222222-2222-4222-8222-222222222222").to_options()

    assert fresh.session_id == "11111111-1111-4111-8111-111111111111"
    assert fresh.resume is None
    assert resumed.resume == "22222222-2222-4222-8222-222222222222"


def test_describe_drops_thinking_and_flattens_tool_activity() -> None:
    from claude_agent_sdk import (
        SystemMessage,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from jake_tools.claude import describe

    assistant = describe(
        AssistantMessage(
            content=[
                ThinkingBlock(thinking="PRIVATE-THOUGHT", signature="sig"),
                TextBlock(text="Reading the file."),
                ToolUseBlock(id="t1", name="Read", input={"file_path": "a.txt"}),
            ],
            model="m",
            session_id="s",
        )
    )
    results = describe(
        UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="t1",
                    content=[{"type": "text", "text": "line"}],
                    is_error=False,
                )
            ]
        )
    )
    echoed = describe(UserMessage(content="Correction: use B", uuid="m-1"))
    init = describe(SystemMessage(subtype="init", data={"session_id": "s-1"}))
    retry = describe(
        SystemMessage(subtype="api_retry", data={"attempt": 2, "error": "overloaded"})
    )
    result = describe(_result(result="done", total_cost_usd=0.02))

    assert assistant.kind == "assistant"
    assert assistant.texts == ["Reading the file."]
    assert "PRIVATE-THOUGHT" not in assistant.model_dump_json()
    assert assistant.tool_uses[0].name == "Read"
    assert results.kind == "tool_results"
    assert results.tool_results[0].text == "line"
    assert (echoed.kind, echoed.uuid, echoed.user_text) == (
        "user_input",
        "m-1",
        "Correction: use B",
    )
    assert (init.kind, init.session_id) == ("init", "s-1")
    assert retry.kind == "notice" and retry.notice and "overloaded" in retry.notice
    assert result.result is not None
    assert result.result.text == "done"
    assert result.result.usage.estimated_cost_usd == 0.02
