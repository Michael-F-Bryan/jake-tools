"""The single seam between jake-tools and the LLM.

Every model call in this package goes through :class:`ClaudeAgent`. It owns the
Claude Agent SDK's async surface so the rest of the codebase deals in typed
prompts in and typed replies out.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    EffortLevel,
    McpServerConfig,
    Message,
    ResultMessage,
    TextBlock,
    query,
)
from pydantic import BaseModel, ConfigDict, Field

from .ai_usage import Usage
from .prompting import Prompt, StructuredPrompt

DEFAULT_MODEL = "claude-sonnet-5"


class ClaudeAgentError(RuntimeError):
    """An agent call did not produce a usable reply.

    ``usage`` carries whatever the run accrued before it failed, so a caller
    that accounts cost can still charge it even though the call raised.
    """

    def __init__(self, message: str, *, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage if usage is not None else Usage()


class Reply(BaseModel):
    """What one agent call returned, and what it cost."""

    text: str | None = None
    usage: Usage = Field(default_factory=Usage)

    @classmethod
    def from_result(cls, result: ResultMessage, text: str | None) -> Reply:
        return cls(text=text, usage=_usage_of(result))


class AgentSpec(BaseModel):
    """How one agent call is configured.

    ``tools`` is the set of built-in tools the agent may use, and it defaults to
    nothing. That default is load-bearing: the SDK omits ``--tools`` when the
    option is ``None``, which hands the agent Claude Code's *full* default
    toolset, so :meth:`to_options` always passes an explicit list.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = DEFAULT_MODEL
    tools: tuple[str, ...] = ()
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict)
    mcp_tools: tuple[str, ...] = ()
    system_prompt: str | None = None
    max_turns: int | None = None
    effort: EffortLevel | None = None
    max_budget_usd: float | None = None

    def merge(self, override: AgentSpec | None) -> AgentSpec:
        """Layer the fields ``override`` actually set on top of this spec.

        Only explicitly-set fields win, so a caller overriding just the model
        does not silently reset the tool grant to the field default.
        """

        if override is None:
            return self
        updates = {name: getattr(override, name) for name in override.model_fields_set}
        return self.model_copy(update=updates)

    def to_options(
        self, *, output_schema: dict[str, Any] | None = None
    ) -> ClaudeAgentOptions:
        output_format = (
            None
            if output_schema is None
            else {"type": "json_schema", "schema": output_schema}
        )
        return ClaudeAgentOptions(
            model=self.model,
            tools=list(self.tools),
            # Everything granted is also auto-approved: these runs are
            # non-interactive, so an unapproved call would block on a prompt.
            allowed_tools=[*self.tools, *self.mcp_tools],
            mcp_servers=dict(self.mcp_servers),
            system_prompt=self.system_prompt,
            max_turns=self.max_turns,
            effort=self.effort,
            max_budget_usd=self.max_budget_usd,
            output_format=output_format,
        )


class QueryFn(Protocol):
    """The shape of :func:`claude_agent_sdk.query`, so tests can supply a fake."""

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]: ...


@dataclass(frozen=True)
class ClaudeAgent:
    """A high-level wrapper around the Claude Agent SDK.

    ``run_query`` is the injection seam: production passes the SDK's
    :func:`~claude_agent_sdk.query`, tests pass a fake that replays messages.
    """

    defaults: AgentSpec = field(default_factory=AgentSpec)
    run_query: QueryFn = field(default=cast(QueryFn, query))

    async def run(self, prompt: str | Prompt, spec: AgentSpec | None = None) -> Reply:
        rendered = prompt if isinstance(prompt, str) else prompt.render()
        options = self.defaults.merge(spec).to_options()
        text, result = await self._collect(rendered, options)
        if result.is_error:
            raise ClaudeAgentError(
                f"agent call to model {options.model!r} failed "
                f"(subtype={result.subtype!r}, stop_reason={result.stop_reason!r}, "
                f"errors={_error_of(result)!r})",
                usage=_usage_of(result),
            )
        return Reply.from_result(result, text)

    async def run_structured[TModel: BaseModel](
        self,
        prompt: StructuredPrompt[TModel],
        spec: AgentSpec | None = None,
    ) -> tuple[TModel, Reply]:
        response_model = cast(type[TModel], prompt.response_model)
        options = self.defaults.merge(spec).to_options(
            output_schema=response_model.model_json_schema()
        )
        text, result = await self._collect(prompt.render(), options)
        return _parse_structured(result, response_model), Reply.from_result(
            result, text
        )

    async def _collect(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> tuple[str | None, ResultMessage]:
        chunks: list[str] = []
        result: ResultMessage | None = None
        async for message in self.run_query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                chunks.extend(
                    block.text
                    for block in message.content
                    if isinstance(block, TextBlock)
                )
            elif isinstance(message, ResultMessage):
                result = message
        if result is None:
            raise ClaudeAgentError(
                f"agent stream for model {options.model!r} ended without a result"
            )
        return ("\n".join(chunks) if chunks else None), result


def _parse_structured[TModel: BaseModel](
    result: ResultMessage, model_type: type[TModel]
) -> TModel:
    payload = result.structured_output
    if payload is None:
        raise ClaudeAgentError(
            f"expected {model_type.__name__} JSON from the agent but got none "
            f"(subtype={result.subtype!r}, stop_reason={result.stop_reason!r}, "
            f"error={_error_of(result)!r})",
            usage=_usage_of(result),
        )
    return model_type.model_validate(payload)


def _error_of(result: ResultMessage) -> str | None:
    if not result.is_error:
        return None
    if result.errors:
        return "; ".join(result.errors)
    return result.subtype


def _usage_of(result: ResultMessage) -> Usage:
    raw = result.usage or {}
    return Usage(
        model=next(iter(result.model_usage), None) if result.model_usage else None,
        api_calls=result.num_turns,
        input_tokens=_count(raw, "input_tokens"),
        output_tokens=_count(raw, "output_tokens"),
        cache_read_tokens=_count(raw, "cache_read_input_tokens"),
        cache_write_tokens=_count(raw, "cache_creation_input_tokens"),
        estimated_cost_usd=result.total_cost_usd or 0.0,
    )


def _count(raw: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(raw.get(key, 0) or 0))
    except TypeError, ValueError:
        return 0
