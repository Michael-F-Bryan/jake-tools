from collections.abc import AsyncIterator

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
)
from jake_tools.prompting import StructuredPrompt


class Minutes(BaseModel):
    summary: str
    actions: list[str]


class MinutesPrompt(StructuredPrompt[Minutes]):
    template = "Summarise:\n{{ transcript }}"
    response_model = Minutes

    transcript: str


def _result(
    *,
    structured_output: object = None,
    is_error: bool = False,
    subtype: str = "success",
    errors: list[str] | None = None,
    num_turns: int = 1,
    usage: dict[str, object] | None = None,
    total_cost_usd: float | None = None,
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
        result=None,
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

    await agent.run(MinutesPrompt(transcript="hello there"))

    assert "hello there" in fake.calls[0][0]


async def test_run_structured_parses_structured_output() -> None:
    fake = RecordingQuery(
        _assistant("{...}"),
        _result(structured_output={"summary": "we shipped", "actions": ["ship more"]}),
    )
    agent = ClaudeAgent(run_query=fake)

    minutes, reply = await agent.run_structured(
        MinutesPrompt(transcript="a transcript")
    )

    assert minutes == Minutes(summary="we shipped", actions=["ship more"])
    assert reply.usage.api_calls == 1


async def test_run_structured_sends_the_response_model_schema() -> None:
    fake = RecordingQuery(_result(structured_output={"summary": "s", "actions": []}))
    agent = ClaudeAgent(run_query=fake)

    await agent.run_structured(MinutesPrompt(transcript="a transcript"))

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
        await agent.run_structured(MinutesPrompt(transcript="a transcript"))

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
    assert reply.usage.estimated_cost_usd == 0.0


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
