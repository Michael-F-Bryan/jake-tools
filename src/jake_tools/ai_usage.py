"""Token and cost accounting for agent calls.

Deliberately knows nothing about the Claude Agent SDK — :mod:`jake_tools.claude`
translates an SDK result into a :class:`Usage`, and everything downstream
(manifests, summaries, cost caps) works against these plain domain types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, computed_field

if TYPE_CHECKING:
    from .claude import Reply


class Usage(BaseModel):
    """What one agent call consumed.

    ``api_calls`` counts model turns, so it is greater than one whenever the
    agent used tools.
    """

    model: str | None = None
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            model=other.model if other.model is not None else self.model,
            api_calls=self.api_calls + other.api_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


class AIStageStats(BaseModel):
    """Per-stage usage, flattened so `summary.json` stays greppable."""

    stage: str
    usage: Usage = Field(default_factory=Usage)

    @computed_field
    @property
    def model(self) -> str | None:
        return self.usage.model

    @computed_field
    @property
    def api_calls(self) -> int:
        return self.usage.api_calls

    @computed_field
    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @computed_field
    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens

    @computed_field
    @property
    def cache_read_tokens(self) -> int:
        return self.usage.cache_read_tokens

    @computed_field
    @property
    def cache_write_tokens(self) -> int:
        return self.usage.cache_write_tokens

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens

    @computed_field
    @property
    def estimated_cost_usd(self) -> float:
        return self.usage.estimated_cost_usd


class AITotals(BaseModel):
    stage_count: int = 0
    usage: Usage = Field(default_factory=Usage)

    @computed_field
    @property
    def api_calls(self) -> int:
        return self.usage.api_calls

    @computed_field
    @property
    def input_tokens(self) -> int:
        return self.usage.input_tokens

    @computed_field
    @property
    def output_tokens(self) -> int:
        return self.usage.output_tokens

    @computed_field
    @property
    def cache_read_tokens(self) -> int:
        return self.usage.cache_read_tokens

    @computed_field
    @property
    def cache_write_tokens(self) -> int:
        return self.usage.cache_write_tokens

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens

    @computed_field
    @property
    def estimated_cost_usd(self) -> float:
        return self.usage.estimated_cost_usd


def build_ai_stage_stats(stage: str, reply: Reply | None) -> AIStageStats | None:
    if reply is None:
        return None
    return AIStageStats(stage=stage, usage=reply.usage)


def build_ai_totals(stage_stats: list[AIStageStats]) -> AITotals:
    usage = Usage()
    for stage in stage_stats:
        usage = usage + stage.usage
    return AITotals(stage_count=len(stage_stats), usage=usage)
