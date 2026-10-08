"""The single seam between jake-tools and the LLM.

Every model call in this package goes through :class:`ClaudeAgent`. It owns the
Claude Agent SDK's async surface so the rest of the codebase deals in typed
prompts in and typed replies out.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    EffortLevel,
    McpServerConfig,
    Message,
    PermissionMode,
    ResultMessage,
    SettingSource,
    TextBlock,
    query,
)
from pydantic import BaseModel, ConfigDict, Field

from .ai_usage import AICallTelemetry, TelemetrySink, Usage
from .prompting import Prompt, StructuredPrompt

__all__ = [
    "DEFAULT_MODEL",
    "AgentSpec",
    "ClaudeAgent",
    "ClaudeAgentError",
    "EffortLevel",
    "Message",
    "PermissionMode",
    "Reply",
    "SettingSource",
    "message_to_json",
]

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
    """What one agent call returned, and what it cost.

    ``text`` is every assistant text block concatenated: the per-turn
    narration. ``final_text`` is the SDK result message's own ``result``
    field, the agent's final answer; a consumer reporting an outcome wants
    that, not the narration.
    """

    text: str | None = None
    final_text: str | None = None
    usage: Usage = Field(default_factory=Usage)

    @classmethod
    def from_result(cls, result: ResultMessage, text: str | None) -> Reply:
        return cls(text=text, final_text=result.result, usage=_usage_of(result))


class AgentSpec(BaseModel):
    """How one agent call is configured.

    ``tools`` is the set of built-in tools the agent may use, and it defaults to
    nothing. That default is load-bearing: the SDK omits ``--tools`` when the
    option is ``None``, which hands the agent Claude Code's *full* default
    toolset, so :meth:`to_options` always passes an explicit list.

    :meth:`to_options` always sets ``strict_mcp_config``: the only MCP servers
    an agent sees are the ones in ``mcp_servers``, never user- or
    project-level ones the CLI would otherwise pick up.

    ``setting_sources`` is ``None`` for the CLI's defaults (all filesystem
    settings); ``()`` isolates the run from every settings file. ``cwd`` and
    ``env`` are the subprocess's working directory and extra environment.
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
    cwd: Path | None = None
    env: dict[str, str] = Field(default_factory=dict)
    setting_sources: tuple[SettingSource, ...] | None = None
    permission_mode: PermissionMode | None = None

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
            strict_mcp_config=True,
            cwd=self.cwd,
            env=dict(self.env),
            setting_sources=(
                None if self.setting_sources is None else list(self.setting_sources)
            ),
            permission_mode=self.permission_mode,
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

    ``on_message`` sees every message the stream yields, in order, before it
    is interpreted; a caller that wants a durable transcript writes each one
    out (see :func:`message_to_json`). It must not raise.
    """

    defaults: AgentSpec = field(default_factory=AgentSpec)
    run_query: QueryFn = field(default=cast(QueryFn, query))
    telemetry_sink: TelemetrySink | None = None
    stage: str | None = None
    on_message: Callable[[Message], None] | None = None

    def with_telemetry(self, sink: TelemetrySink) -> ClaudeAgent:
        """Bind a durable sink without changing the query seam."""
        return dataclasses.replace(self, telemetry_sink=sink)

    def for_stage(self, stage: str) -> ClaudeAgent:
        """Carry explicit stage identity through a composed stage."""
        return dataclasses.replace(self, stage=stage)

    def with_defaults(self, defaults: AgentSpec) -> ClaudeAgent:
        return dataclasses.replace(self, defaults=defaults)

    def with_message_observer(
        self, on_message: Callable[[Message], None]
    ) -> ClaudeAgent:
        """Observe the raw message stream without changing the query seam."""
        return dataclasses.replace(self, on_message=on_message)

    async def run(
        self,
        prompt: str | Prompt,
        spec: AgentSpec | None = None,
        *,
        stage: str | None = None,
        scope: str | None = None,
    ) -> Reply:
        rendered = prompt if isinstance(prompt, str) else prompt.render()
        options = self.defaults.merge(spec).to_options()
        stage_name = stage or self.stage or "unspecified"
        try:
            text, result = await self._collect(rendered, options)
        except ClaudeAgentError as exc:
            self._record_error(stage_name, options, exc, scope=scope)
            raise
        except Exception as exc:
            self._record_error(
                stage_name,
                options,
                ClaudeAgentError(f"agent call failed before a result: {exc}"),
                scope=scope,
            )
            raise
        if result.is_error:
            error = ClaudeAgentError(
                f"agent call to model {options.model!r} failed "
                f"(subtype={result.subtype!r}, stop_reason={result.stop_reason!r}, "
                f"errors={_error_of(result)!r})",
                usage=_usage_of(result),
            )
            self._record_result(
                stage_name, options, result, status="error", scope=scope
            )
            raise error
        self._record_result(stage_name, options, result, status="success", scope=scope)
        return Reply.from_result(result, text)

    async def run_structured[TModel: BaseModel](
        self,
        prompt: StructuredPrompt[TModel],
        spec: AgentSpec | None = None,
        *,
        stage: str | None = None,
        scope: str | None = None,
    ) -> tuple[TModel, Reply]:
        response_model = cast(type[TModel], prompt.response_model)
        options = self.defaults.merge(spec).to_options(
            output_schema=response_model.model_json_schema()
        )
        stage_name = stage or self.stage or "unspecified"
        try:
            text, result = await self._collect(prompt.render(), options)
        except ClaudeAgentError as exc:
            self._record_error(stage_name, options, exc, scope=scope)
            raise
        except Exception as exc:
            self._record_error(
                stage_name,
                options,
                ClaudeAgentError(f"agent call failed before a result: {exc}"),
                scope=scope,
            )
            raise
        if result.is_error:
            error = ClaudeAgentError(
                f"expected {response_model.__name__} JSON from the agent but got an "
                f"error result (subtype={result.subtype!r}, "
                f"stop_reason={result.stop_reason!r}, errors={_error_of(result)!r})",
                usage=_usage_of(result),
            )
            self._record_result(
                stage_name, options, result, status="error", scope=scope
            )
            raise error
        try:
            parsed = _parse_structured(result, response_model)
        except Exception:
            self._record_result(
                stage_name, options, result, status="error", scope=scope
            )
            raise
        self._record_result(stage_name, options, result, status="success", scope=scope)
        return parsed, Reply.from_result(result, text)

    def record_cache_hit(self, *, stage: str | None = None) -> None:
        if self.telemetry_sink is not None:
            self.telemetry_sink.record_cache_hit(
                stage=stage or self.stage or "unspecified", model=self.defaults.model
            )

    def _record_error(
        self,
        stage: str,
        options: ClaudeAgentOptions,
        error: ClaudeAgentError,
        *,
        scope: str | None,
    ) -> None:
        if self.telemetry_sink is None:
            return
        self.telemetry_sink.record_call(
            AICallTelemetry(stage=stage, scope=scope, status="error", usage=error.usage)
        )

    def _record_result(
        self,
        stage: str,
        options: ClaudeAgentOptions,
        result: ResultMessage,
        *,
        status: Literal["success", "error"],
        scope: str | None,
    ) -> None:
        if self.telemetry_sink is None:
            return
        self.telemetry_sink.record_call(
            _telemetry_of(
                result, stage=stage, status=status, model=options.model, scope=scope
            )
        )

    async def _collect(
        self, prompt: str, options: ClaudeAgentOptions
    ) -> tuple[str | None, ResultMessage]:
        chunks: list[str] = []
        result: ResultMessage | None = None
        rate_limit_detail: str | None = None
        try:
            async for message in self.run_query(prompt=prompt, options=options):
                if self.on_message is not None:
                    self.on_message(message)
                if isinstance(message, AssistantMessage):
                    text_blocks = [
                        block.text
                        for block in message.content
                        if isinstance(block, TextBlock)
                    ]
                    chunks.extend(text_blocks)
                    if message.error == "rate_limit":
                        rate_limit_detail = "\n".join(text_blocks).strip()
                elif isinstance(message, ResultMessage):
                    result = message
        except Exception as exc:
            if rate_limit_detail is not None:
                raise _rate_limit_error(rate_limit_detail) from exc
            raise
        if rate_limit_detail is not None:
            raise _rate_limit_error(rate_limit_detail)
        if result is None:
            raise ClaudeAgentError(
                f"agent stream for model {options.model!r} ended without a result"
            )
        return ("\n".join(chunks) if chunks else None), result


def message_to_json(message: Message) -> dict[str, Any]:
    """A JSON-ready view of one SDK message, tagged with its type name.

    SDK messages are plain dataclasses, so this is a recursive ``asdict`` plus
    a ``type`` discriminator; anything the SDK leaves untyped (raw usage
    dicts, tool inputs) passes through as-is.
    """
    payload: dict[str, Any] = {"type": type(message).__name__}
    if dataclasses.is_dataclass(message):
        payload.update(dataclasses.asdict(message))
    else:  # pragma: no cover - every SDK message type is a dataclass today
        payload["repr"] = repr(message)
    return payload


def _rate_limit_error(detail: str) -> ClaudeAgentError:
    suffix = f": {detail}" if detail else ""
    return ClaudeAgentError(f"Claude rate limit reached{suffix}")


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
        model=_model_of(result),
        provider=_provider_of(result),
        model_usage=dict(result.model_usage or {}),
        api_calls=result.num_turns,
        input_tokens=_count(raw, "input_tokens"),
        output_tokens=_count(raw, "output_tokens"),
        cache_read_tokens=_count(raw, "cache_read_input_tokens"),
        cache_write_tokens=_cache_write_count(raw),
        estimated_cost_usd=result.total_cost_usd,
    )


def _model_of(result: ResultMessage) -> str | None:
    if not result.model_usage or len(result.model_usage) != 1:
        return None
    return next(iter(result.model_usage))


def _provider_of(result: ResultMessage) -> str | None:
    if not result.model_usage or len(result.model_usage) != 1:
        return None
    value = next(iter(result.model_usage.values()))
    if isinstance(value, dict):
        provider = value.get("provider")
        return provider if isinstance(provider, str) else None
    return None


def _telemetry_of(
    result: ResultMessage,
    *,
    stage: str,
    status: Literal["success", "error"],
    model: str | None,
    scope: str | None,
) -> AICallTelemetry:
    usage = _usage_of(result)
    if usage.model is None and not result.model_usage:
        usage = usage.model_copy(update={"model": model})
    cache_creation_5m_tokens = _cache_bucket(
        result.usage or {}, "ephemeral_5m_input_tokens"
    )
    cache_creation_1h_tokens = _cache_bucket(
        result.usage or {}, "ephemeral_1h_input_tokens"
    )
    return AICallTelemetry(
        stage=stage,
        scope=scope,
        status=status,
        usage=usage,
        provider=usage.provider,
        duration_ms=result.duration_ms,
        duration_api_ms=result.duration_api_ms,
        cache_creation_5m_tokens=cache_creation_5m_tokens,
        cache_creation_1h_tokens=cache_creation_1h_tokens,
    )


def _cache_bucket(raw: dict[str, Any], key: str) -> int:
    nested = raw.get("cache_creation")
    if isinstance(nested, dict):
        return _count(nested, key)
    suffix = key.removesuffix("_input_tokens")
    return _count(raw, f"cache_creation_input_tokens_{suffix}")


def _cache_write_count(raw: dict[str, Any]) -> int:
    aggregate = _count(raw, "cache_creation_input_tokens")
    if aggregate:
        return aggregate
    return _cache_bucket(raw, "ephemeral_5m_input_tokens") + _cache_bucket(
        raw, "ephemeral_1h_input_tokens"
    )


def _count(raw: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(raw.get(key, 0) or 0))
    except TypeError, ValueError:
        return 0
