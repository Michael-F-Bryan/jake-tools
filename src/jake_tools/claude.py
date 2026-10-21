"""The single seam between jake-tools and the LLM.

Every model call in this package goes through :class:`ClaudeAgent`. It owns the
Claude Agent SDK's async surface so the rest of the codebase deals in typed
prompts in and typed replies out.

Two shapes of call share one configuration (:class:`AgentSpec`):

- :meth:`ClaudeAgent.run` / :meth:`ClaudeAgent.run_structured`: one prompt in,
  one reply out, through the SDK's one-shot ``query``.
- :meth:`ClaudeAgent.converse`: a live :class:`Conversation` over the SDK's
  streaming client. The caller sends user messages whenever it likes (the CLI
  queues one sent mid-turn and adds it to the conversation after the running
  tool finishes) and reads the message stream. :func:`describe` turns each
  message into an SDK-free :class:`MessageView` that never carries thinking.

:class:`LocalTool` is an in-process tool the agent can call, served by an SDK
MCP server inside this process; it is how a caller gives the agent a narrow
capability (such as asking its parent a question) without granting any
external MCP server.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    EffortLevel,
    McpServerConfig,
    Message,
    PermissionMode,
    ResultMessage,
    SdkMcpTool,
    SettingSource,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
)
from pydantic import BaseModel, ConfigDict, Field

from .ai_usage import AICallTelemetry, TelemetrySink, Usage
from .prompting import Prompt, StructuredPrompt

__all__ = [
    "DEFAULT_MODEL",
    "LOCAL_MCP_SERVER",
    "AgentSpec",
    "ClaudeAgent",
    "ClaudeAgentError",
    "Conversation",
    "EffortLevel",
    "LocalTool",
    "LocalToolResult",
    "Message",
    "MessageView",
    "PermissionMode",
    "Reply",
    "ResultView",
    "SettingSource",
    "TOOL_NAME_PATTERN",
    "ToolResultView",
    "ToolUseView",
    "check_tool_name",
    "describe",
    "local_tool_name",
    "message_to_json",
    "permission_rule_tool",
    "unexposed_tools",
]

DEFAULT_MODEL = "claude-sonnet-5"

LOCAL_MCP_SERVER = "jake"
"""The name of the in-process MCP server that serves :class:`LocalTool` s."""


class ClaudeAgentError(RuntimeError):
    """An agent call did not produce a usable reply.

    ``usage`` carries whatever the run accrued before it failed, so a caller
    that accounts cost can still charge it even though the call raised.

    ``result_subtype`` is the SDK result message's ``subtype`` when the
    failure was reported *as a result* (``error_max_turns``,
    ``error_max_budget_usd``, ``error_during_execution``, ...) and ``None``
    when the stream failed before producing one; a caller that must tell a
    turn limit from a crash reads it rather than parsing the message.
    ``final_text`` is that result's own text, when it had one.
    """

    def __init__(
        self,
        message: str,
        *,
        usage: Usage | None = None,
        result_subtype: str | None = None,
        final_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = usage if usage is not None else Usage()
        self.result_subtype = result_subtype
        self.final_text = final_text


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


class LocalToolResult(BaseModel):
    """What a :class:`LocalTool` handler returns to the agent."""

    text: str
    is_error: bool = False


@dataclass(frozen=True)
class LocalTool:
    """An in-process tool the agent may call, and nothing else may.

    ``input_schema`` is a JSON Schema object; the handler gets the agent's
    arguments as a dict and must validate them itself (the schema guides the
    model, it is not enforced). The agent sees the tool as
    :func:`local_tool_name` ``(name)``; granting it never grants any other
    MCP server, because :meth:`AgentSpec.to_options` keeps
    ``strict_mcp_config``.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[LocalToolResult]]


def local_tool_name(name: str) -> str:
    """The name the agent uses for the :class:`LocalTool` called ``name``."""
    return f"mcp__{LOCAL_MCP_SERVER}__{name}"


TOOL_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
"""What a built-in tool name looks like. Shape only: there is no catalogue."""

_RULE_PATTERN = re.compile(
    r"(?P<tool>[A-Za-z][A-Za-z0-9_]*)\((?P<specifier>[^,\n\r]+)\)"
)

_RUNTIME_ALIASES: tuple[frozenset[str], ...] = (frozenset({"Agent", "Task"}),)
"""Names the CLI reports differently from how they were requested.

Observed with CLI 2.1.293: ``--tools Agent`` is reported as ``Task`` in the
session's ``init`` tool list."""


def check_tool_name(name: str) -> str:
    """``name`` if it is shaped like a built-in tool name, else ``ValueError``.

    The SDK passes ``tools`` to the CLI as an opaque comma-joined list and
    neither publishes a catalogue nor validates it: the CLI silently drops a
    name it does not provide (observed, CLI 2.1.293). So this checks shape
    only; what a session really exposes is its ``init`` message's tool list
    (:attr:`MessageView.tools`, :func:`unexposed_tools`).
    """
    if not TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"{name!r} is not a tool name (letters, digits and '_'; a permission "
            "rule such as 'Bash(git status)' belongs in allowed_tools)"
        )
    return name


def permission_rule_tool(rule: str) -> str:
    """The tool a permission rule (``Read``, ``Bash(git status)``) applies to.

    The grammar is the CLI's ``ToolName`` or ``ToolName(specifier)``. A
    specifier may not contain a comma or a line break: the SDK joins
    ``allowed_tools`` with commas for ``--allowedTools``, so such a rule
    would reach the CLI as two different rules. Raises ``ValueError``.
    """
    if TOOL_NAME_PATTERN.fullmatch(rule):
        return rule
    match = _RULE_PATTERN.fullmatch(rule)
    if match is None or not match["specifier"].strip():
        raise ValueError(
            f"{rule!r} is not a permission rule: use ToolName or "
            "ToolName(specifier), e.g. 'Bash(uv run pytest *)', with no comma "
            "or line break in the specifier"
        )
    return match["tool"]


def unexposed_tools(requested: Iterable[str], exposed: Iterable[str]) -> list[str]:
    """The requested built-in tools a session's ``init`` tool list lacks."""
    available = set(exposed)
    missing: list[str] = []
    for name in requested:
        names = {name}
        for group in _RUNTIME_ALIASES:
            if name in group:
                names |= group
        if not names & available:
            missing.append(name)
    return missing


class AgentSpec(BaseModel):
    """How one agent call is configured.

    ``tools`` is the set of built-in tools the session *exposes*, and it
    defaults to nothing. That default is load-bearing: the SDK omits
    ``--tools`` when the option is ``None``, which hands the agent Claude
    Code's *full* default toolset, so :meth:`to_options` always passes an
    explicit list. The CLI drops names it does not provide without an error;
    the ``init`` message says what it actually exposed.

    ``allowed_tools`` is the set of permission rules that *auto-approve*
    calls (the SDK's ``allowed_tools``): ``"Read"`` approves every ``Read``
    call, ``"Bash(git status)"`` only that command. ``None`` (the default)
    approves every exposed tool outright, which is what ``tools`` alone has
    always meant. An approval is not the only way a call runs: the CLI runs
    calls that need no approval (reads inside the working directory,
    read-only Bash commands) under any rules, and under ``dontAsk`` denies
    everything else.

    :meth:`to_options` always sets ``strict_mcp_config``: the only MCP servers
    an agent sees are the ones in ``mcp_servers``, never user- or
    project-level ones the CLI would otherwise pick up.

    ``setting_sources`` is ``None`` for the CLI's defaults (all filesystem
    settings); ``()`` isolates the run from every settings file. ``cwd`` and
    ``env`` are the subprocess's working directory and extra environment.

    ``local_tools`` are served in-process and auto-approved like ``tools``.
    ``session_id`` names the Claude session a fresh conversation creates;
    ``resume`` continues an existing one instead (the CLI keeps its ID and
    fails, rather than starting afresh, when it cannot find it).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = DEFAULT_MODEL
    tools: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] | None = None
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
    local_tools: tuple[LocalTool, ...] = ()
    session_id: str | None = None
    resume: str | None = None

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
        self,
        *,
        output_schema: dict[str, Any] | None = None,
        extra_args: dict[str, str | None] | None = None,
    ) -> ClaudeAgentOptions:
        output_format = (
            None
            if output_schema is None
            else {"type": "json_schema", "schema": output_schema}
        )
        mcp_servers: dict[str, McpServerConfig] = dict(self.mcp_servers)
        local_names: list[str] = []
        if self.local_tools:
            mcp_servers[LOCAL_MCP_SERVER] = create_sdk_mcp_server(
                LOCAL_MCP_SERVER, tools=[_sdk_tool(t) for t in self.local_tools]
            )
            local_names = [local_tool_name(t.name) for t in self.local_tools]
        rules = self.tools if self.allowed_tools is None else self.allowed_tools
        return ClaudeAgentOptions(
            # The SDK's bundled ELF may require a loader absent on Nix.
            # Prefer the installed CLI; retain SDK discovery if none exists.
            cli_path=shutil.which("claude"),
            model=self.model,
            # Exposure: bare built-in names only, never permission rules.
            tools=list(self.tools),
            # Approval: the caller's rules (by default every exposed tool),
            # plus the MCP and local tools this spec configured itself.
            allowed_tools=[*rules, *self.mcp_tools, *local_names],
            mcp_servers=mcp_servers,
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
            session_id=self.session_id,
            resume=self.resume,
            extra_args=dict(extra_args or {}),
        )


def _sdk_tool(local: LocalTool) -> SdkMcpTool[Any]:
    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        result = await local.handler(arguments)
        payload: dict[str, Any] = {"content": [{"type": "text", "text": result.text}]}
        if result.is_error:
            payload["is_error"] = True
        return payload

    return SdkMcpTool(
        name=local.name,
        description=local.description,
        input_schema=local.input_schema,
        handler=handler,
    )


class QueryFn(Protocol):
    """The shape of :func:`claude_agent_sdk.query`, so tests can supply a fake."""

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]: ...


class StreamingClient(Protocol):
    """The part of :class:`claude_agent_sdk.ClaudeSDKClient` a conversation uses."""

    async def connect(
        self, prompt: str | AsyncIterable[dict[str, Any]] | None = None
    ) -> None: ...

    async def query(
        self, prompt: str | AsyncIterable[dict[str, Any]], session_id: str = "default"
    ) -> None: ...

    def receive_messages(self) -> AsyncIterator[Message]: ...

    async def disconnect(self) -> None: ...


ConnectFn = Callable[[ClaudeAgentOptions], StreamingClient]
"""Builds the streaming client for a conversation; production uses the SDK's."""


@dataclass(frozen=True)
class ClaudeAgent:
    """A high-level wrapper around the Claude Agent SDK.

    ``run_query`` is the injection seam for one-shot calls: production passes
    the SDK's :func:`~claude_agent_sdk.query`, tests pass a fake that replays
    messages. ``connect`` is the same seam for :meth:`converse`.

    ``on_message`` sees every message the stream yields, in order, before it
    is interpreted; a caller that wants a durable transcript writes each one
    out (see :func:`message_to_json`). It must not raise.
    """

    defaults: AgentSpec = field(default_factory=AgentSpec)
    run_query: QueryFn = field(default=cast(QueryFn, query))
    connect: ConnectFn = field(default=cast(ConnectFn, ClaudeSDKClient))
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
            error = _result_error(
                f"agent call to model {options.model!r} failed", result
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
            error = _result_error(
                f"expected {response_model.__name__} JSON from the agent but got an "
                "error result",
                result,
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

    @asynccontextmanager
    async def converse(
        self,
        spec: AgentSpec | None = None,
        *,
        stage: str | None = None,
        scope: str | None = None,
        cost_baseline_usd: float | None = None,
    ) -> AsyncIterator[Conversation]:
        """Open a streaming conversation; it is closed when the block exits.

        The CLI is started with ``--replay-user-messages``, so every user
        message comes back on the stream, with the ``uuid`` it was sent with,
        at the point the CLI adds it to the conversation. That echo is the
        evidence a message reached the conversation; writing it to the CLI's
        stdin is not.

        Usage: the CLI reports token usage per prompt but ``total_cost_usd``
        as the session's running total, carried across a resume (observed
        with CLI 2.1.293). Each result is therefore recorded to telemetry
        with its own tokens and only the cost it added since the previous
        result, starting from ``cost_baseline_usd`` (the session's total
        when this conversation began). :attr:`Conversation.usage` is the
        sum of those per-turn records: this invocation's usage and cost, and
        what the telemetry totals add up to.

        Closing goes through the SDK's shielded disconnect, so the CLI is
        terminated even when the block is cancelled.
        """

        options = self.defaults.merge(spec).to_options(
            extra_args={"replay-user-messages": None}
        )
        client = self.connect(options)
        conversation = Conversation(
            _client=client,
            _agent=self,
            _options=options,
            _stage=stage or self.stage or "unspecified",
            _scope=scope,
            _session_cost_usd=cost_baseline_usd,
        )
        try:
            await client.connect()
            yield conversation
        finally:
            await client.disconnect()

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
        cost_usd: float | None = None,
    ) -> None:
        if self.telemetry_sink is None:
            return
        record = _telemetry_of(
            result, stage=stage, status=status, model=options.model, scope=scope
        )
        if cost_usd is not None:
            record = record.model_copy(
                update={
                    "usage": record.usage.model_copy(
                        update={"estimated_cost_usd": cost_usd}
                    )
                }
            )
        self.telemetry_sink.record_call(record)

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
            if result is not None and result.is_error:
                # The CLI exits non-zero after an error result on purpose, and
                # the SDK surfaces that exit as an exception *after* yielding
                # the result. The result is the informative half: keep it.
                raise _result_error(
                    f"agent call to model {options.model!r} failed", result
                ) from exc
            raise
        if rate_limit_detail is not None:
            raise _rate_limit_error(rate_limit_detail)
        if result is None:
            raise ClaudeAgentError(
                f"agent stream for model {options.model!r} ended without a result"
            )
        return ("\n".join(chunks) if chunks else None), result


@dataclass
class Conversation:
    """A live streaming conversation; obtain one from :meth:`ClaudeAgent.converse`."""

    _client: StreamingClient
    _agent: ClaudeAgent
    _options: ClaudeAgentOptions
    _stage: str
    _scope: str | None
    _session_cost_usd: float | None = None
    usage: Usage = field(default_factory=Usage)
    """This conversation's usage so far: every result's tokens, and the cost
    added since the conversation began."""

    @property
    def model(self) -> str | None:
        return self._options.model

    @property
    def session_cost_usd(self) -> float | None:
        """The session's running cost as the CLI last reported it."""
        return self._session_cost_usd

    def _turn_usage(self, result: ResultMessage) -> Usage:
        usage = _usage_of(result)
        total = result.total_cost_usd
        if total is None:
            return usage
        added = max(0.0, total - (self._session_cost_usd or 0.0))
        self._session_cost_usd = total
        return usage.model_copy(update={"estimated_cost_usd": added})

    async def send(self, text: str, *, uuid: str) -> None:
        """Write one user message to the CLI, tagged with ``uuid``.

        Returning means the CLI's stdin accepted the bytes, nothing more; the
        message is in the conversation once :func:`describe` reports a
        ``user_input`` view carrying the same ``uuid``.
        """

        async def one() -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "user",
                "message": {"role": "user", "content": text},
                "parent_tool_use_id": None,
                "uuid": uuid,
            }

        await self._client.query(one())

    async def messages(self) -> AsyncIterator[Message]:
        """Every message the CLI sends, in order, until the stream ends.

        The agent's ``on_message`` observer sees each one first. A result
        message is recorded to the telemetry sink as it arrives.
        """

        async for message in self._client.receive_messages():
            if self._agent.on_message is not None:
                self._agent.on_message(message)
            if isinstance(message, ResultMessage):
                turn = self._turn_usage(message)
                # ``model_usage`` is the CLI's own per-model breakdown; keep the
                # latest rather than summing what may be running totals.
                self.usage = (self.usage + turn).model_copy(
                    update={"model_usage": turn.model_usage}
                )
                self._agent._record_result(
                    self._stage,
                    self._options,
                    message,
                    status="error" if message.is_error else "success",
                    scope=self._scope,
                    cost_usd=turn.estimated_cost_usd,
                )
            yield message


class ToolUseView(BaseModel):
    """A tool call the agent made: which tool, with what input."""

    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultView(BaseModel):
    """A tool's result as the agent saw it, flattened to text."""

    tool_use_id: str
    text: str
    is_error: bool = False


class ResultView(BaseModel):
    """One turn's result message: how it ended, its final text and usage."""

    subtype: str
    is_error: bool
    text: str | None = None
    stop_reason: str | None = None
    errors: list[str] = Field(default_factory=list)
    num_turns: int = 0
    usage: Usage = Field(default_factory=Usage)


class MessageView(BaseModel):
    """An SDK message reduced to what a supervisor may show or act on.

    ``kind`` says which fields are meaningful. ``texts`` holds only public
    text blocks: thinking blocks are dropped here and never reach a caller.
    ``user_text``/``uuid`` on a ``user_input`` view is a user message the CLI
    replayed into the conversation. ``tools`` on an ``init`` view is every
    tool the session actually exposes, as the CLI reports it (``None`` when
    the message did not say).
    """

    kind: Literal[
        "init", "assistant", "user_input", "tool_results", "result", "notice", "other"
    ]
    session_id: str | None = None
    uuid: str | None = None
    message_id: str | None = None
    texts: list[str] = Field(default_factory=list)
    tool_uses: list[ToolUseView] = Field(default_factory=list)
    tool_results: list[ToolResultView] = Field(default_factory=list)
    user_text: str | None = None
    result: ResultView | None = None
    notice: str | None = None
    tools: list[str] | None = None


def describe(message: Message) -> MessageView:
    """The :class:`MessageView` of one SDK message."""
    if isinstance(message, AssistantMessage):
        if message.error == "rate_limit":
            detail = " ".join(
                b.text for b in message.content if isinstance(b, TextBlock)
            )
            return MessageView(
                kind="notice",
                session_id=message.session_id,
                notice=f"rate limit: {detail}".strip(),
            )
        return MessageView(
            kind="assistant",
            session_id=message.session_id,
            uuid=message.uuid,
            message_id=message.message_id,
            texts=[b.text for b in message.content if isinstance(b, TextBlock)],
            tool_uses=[
                ToolUseView(id=b.id, name=b.name, input=dict(b.input or {}))
                for b in message.content
                if isinstance(b, ToolUseBlock)
            ],
        )
    if isinstance(message, UserMessage):
        if isinstance(message.content, str):
            return MessageView(
                kind="user_input", uuid=message.uuid, user_text=message.content
            )
        results = [
            ToolResultView(
                tool_use_id=b.tool_use_id,
                text=_flatten_tool_content(b.content),
                is_error=bool(b.is_error),
            )
            for b in message.content
            if isinstance(b, ToolResultBlock)
        ]
        if results:
            return MessageView(
                kind="tool_results", uuid=message.uuid, tool_results=results
            )
        texts = [b.text for b in message.content if isinstance(b, TextBlock)]
        return MessageView(
            kind="user_input", uuid=message.uuid, user_text="\n".join(texts)
        )
    if isinstance(message, ResultMessage):
        return MessageView(
            kind="result",
            session_id=message.session_id,
            uuid=message.uuid,
            result=ResultView(
                subtype=message.subtype,
                is_error=message.is_error,
                text=message.result,
                stop_reason=message.stop_reason,
                errors=list(message.errors or []),
                num_turns=message.num_turns,
                usage=_usage_of(message),
            ),
        )
    if isinstance(message, SystemMessage):
        data = message.data or {}
        session_id = data.get("session_id")
        session = session_id if isinstance(session_id, str) else None
        if message.subtype == "init":
            tools = data.get("tools")
            return MessageView(
                kind="init",
                session_id=session,
                tools=[str(t) for t in tools] if isinstance(tools, list) else None,
            )
        if message.subtype == "api_retry":
            attempt = data.get("attempt")
            error = data.get("error") or data.get("error_status")
            return MessageView(
                kind="notice",
                session_id=session,
                notice=f"API retry (attempt {attempt}): {error}"
                if attempt is not None
                else f"API retry: {error}",
            )
        return MessageView(kind="other", session_id=session)
    return MessageView(kind="other")


def _flatten_tool_content(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        else:
            parts.append(f"[{item.get('type', 'content')}]")
    return "\n".join(parts)


def _result_error(prefix: str, result: ResultMessage) -> ClaudeAgentError:
    """A :class:`ClaudeAgentError` for a result the SDK flagged ``is_error``."""
    return ClaudeAgentError(
        f"{prefix} (subtype={result.subtype!r}, stop_reason={result.stop_reason!r}, "
        f"errors={_error_of(result)!r})",
        usage=_usage_of(result),
        result_subtype=result.subtype,
        final_text=result.result,
    )


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
