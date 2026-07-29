"""Fakes for the :class:`~jake_tools.claude.ClaudeAgent` seam.

Tests inject at ``run_query`` — the same boundary production uses — so the
wrapper's own prompt rendering, schema injection, and usage accounting stay
real. Only the subprocess call to Claude is replaced.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    TextBlock,
)

from jake_tools.claude import AgentSpec, ClaudeAgent


@dataclass
class AgentTurn:
    """One scripted agent response."""

    text: str | None = None
    structured_output: Any = None
    api_calls: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    is_error: bool = False
    subtype: str = "success"

    def messages(self) -> Iterator[Message]:
        if self.text is not None:
            yield AssistantMessage(
                content=[TextBlock(text=self.text)], model="fake-model"
            )
        yield ResultMessage(
            subtype=self.subtype,
            duration_ms=1,
            duration_api_ms=1,
            is_error=self.is_error,
            num_turns=self.api_calls,
            session_id="fake-session",
            total_cost_usd=self.cost_usd,
            usage={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            result=None,
            structured_output=self.structured_output,
        )


def structured(payload: Any, **kwargs: Any) -> AgentTurn:
    """A turn whose structured output parses into the prompt's response model."""

    return AgentTurn(structured_output=payload, **kwargs)


@dataclass
class ScriptedQuery:
    """Replays one :class:`AgentTurn` per call and records what it was asked."""

    turns: list[AgentTurn]
    prompts: list[str] = field(default_factory=list)
    options: list[ClaudeAgentOptions] = field(default_factory=list)

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.prompts.append(prompt)
        self.options.append(options)
        if not self.turns:
            raise AssertionError(
                f"agent called {len(self.prompts)} times but only "
                f"{len(self.prompts) - 1} turns were scripted"
            )
        turn = self.turns.pop(0)

        async def stream() -> AsyncIterator[Message]:
            for message in turn.messages():
                yield message

        return stream()


def fake_agent(*turns: AgentTurn, spec: AgentSpec | None = None) -> ClaudeAgent:
    """A :class:`ClaudeAgent` that replays ``turns`` instead of calling Claude."""

    return ClaudeAgent(
        defaults=spec or AgentSpec(model="fake-model"),
        run_query=ScriptedQuery(list(turns)),
    )


def scripted_query_of(agent: ClaudeAgent) -> ScriptedQuery:
    """The recorder behind a :func:`fake_agent`, for asserting on prompts."""

    assert isinstance(agent.run_query, ScriptedQuery)
    return agent.run_query
