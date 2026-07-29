"""Token and cost accounting for agent calls.

Deliberately knows nothing about the Claude Agent SDK — :mod:`jake_tools.claude`
translates an SDK result into a :class:`Usage`, and everything downstream
(manifests, summaries, cost caps) works against these plain domain types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    computed_field,
    model_serializer,
)

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
            # A mixed-model total has no single model to report; only
            # collapse to one name when both sides actually agree.
            model=self.model if self.model == other.model else None,
            api_calls=self.api_calls + other.api_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


def _flatten_usage(
    dumped: dict[str, Any], *, exclude: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Duplicate ``usage``'s scalar fields at the top level alongside it.

    `summary.json` consumers grep the flat keys, while
    :func:`jake_tools.ai_watch.tuning.run_tune` rehydrates the model from
    that same file via ``model_validate_json`` — so the nested ``usage``
    object has to survive untouched for the round trip to work.
    """
    extra = {key: value for key, value in dumped["usage"].items() if key not in exclude}
    return {**dumped, **extra}


class AIStageStats(BaseModel):
    """Per-stage usage, flattened so `summary.json` stays greppable."""

    stage: str
    usage: Usage = Field(default_factory=Usage)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _flatten_usage(handler(self))


class AITotals(BaseModel):
    stage_count: int = 0
    usage: Usage = Field(default_factory=Usage)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Unlike AIStageStats, totals never carried a top-level `model` — a
        # sum across stages has no single model to flatten.
        return _flatten_usage(handler(self), exclude=frozenset({"model"}))


def build_ai_stage_stats(stage: str, reply: Reply | None) -> AIStageStats | None:
    if reply is None:
        return None
    return AIStageStats(stage=stage, usage=reply.usage)


def build_ai_totals(stage_stats: list[AIStageStats]) -> AITotals:
    usage = Usage()
    for stage in stage_stats:
        usage = usage + stage.usage
    return AITotals(stage_count=len(stage_stats), usage=usage)
