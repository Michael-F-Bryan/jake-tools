"""Typed token and API-rate-equivalent cost accounting for agent calls.

This module deliberately knows nothing about the Claude Agent SDK or the run
cache. The SDK seam translates its ``ResultMessage`` into these domain types;
the cache persists them. ``estimated_cost_usd`` is the SDK's list-price/API-
equivalent estimate, never an invoice or subscription marginal cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    computed_field,
    model_serializer,
)

if TYPE_CHECKING:
    from .claude import Reply


COST_BASIS = (
    "list-price/API-equivalent estimate; not invoice or subscription marginal cost"
)
CallStatus = Literal["success", "error", "cache_hit"]


class Usage(BaseModel):
    """What one agent call consumed.

    ``estimated_cost_usd=None`` means the provider did not price the call. It
    must not be rendered as free when non-zero tokens were consumed. A real
    zero is reserved for explicit local/cache work and cache-hit records.
    """

    model: str | None = None
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float | None = None

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_activity(self) -> bool:
        return bool(
            self.api_calls
            or self.input_tokens
            or self.output_tokens
            or self.cache_read_tokens
            or self.cache_write_tokens
            or self.model is not None
        )

    def __add__(self, other: Usage) -> Usage:
        if not self.has_activity:
            return other
        if not other.has_activity:
            return self
        if self.estimated_cost_usd is None or other.estimated_cost_usd is None:
            estimated_cost_usd = None
        else:
            estimated_cost_usd = self.estimated_cost_usd + other.estimated_cost_usd
        return Usage(
            model=self.model if self.model == other.model else None,
            api_calls=self.api_calls + other.api_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )


def _flatten_usage(
    dumped: dict[str, Any], *, exclude: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Duplicate ``usage``'s scalar fields at the top level alongside it."""
    extra = {key: value for key, value in dumped["usage"].items() if key not in exclude}
    return {**dumped, **extra}


class AIStageStats(BaseModel):
    """Aggregated usage for one model-backed stage."""

    stage: str
    usage: Usage = Field(default_factory=Usage)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _flatten_usage(handler(self))


class AITotals(BaseModel):
    """Aggregated usage across model-backed stages."""

    stage_count: int = 0
    usage: Usage = Field(default_factory=Usage)

    @property
    def estimated_cost_usd(self) -> float | None:
        return self.usage.estimated_cost_usd

    @property
    def api_rate_equivalent_usd(self) -> float | None:
        return self.estimated_cost_usd

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return _flatten_usage(handler(self), exclude=frozenset({"model"}))


class AICallTelemetry(BaseModel):
    """One durable attempt at the ClaudeAgent seam.

    ``attempt`` is assigned by the durable sink, rather than by a caller, so
    concurrent chapter calls and retries receive a deterministic per-stage
    sequence. ``cache_hit`` is an explicit zero-cost event, not a model call.
    """

    stage: str
    attempt: int = 0
    status: CallStatus
    usage: Usage = Field(default_factory=Usage)
    provider: str | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0
    cost_basis: str | None = COST_BASIS

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
    def estimated_cost_usd(self) -> float | None:
        return self.usage.estimated_cost_usd

    @property
    def api_rate_equivalent_usd(self) -> float | None:
        return self.estimated_cost_usd

    def as_usage(self) -> Usage:
        return self.usage

    @classmethod
    def cache_hit(cls, *, stage: str, model: str | None = None) -> AICallTelemetry:
        return cls(
            stage=stage,
            status="cache_hit",
            usage=Usage(model=model, estimated_cost_usd=0.0),
            duration_ms=0,
            duration_api_ms=0,
        )


class AITelemetry(BaseModel):
    """The incrementally persisted AI telemetry document for one run."""

    schema_version: int = 1
    cost_basis: str = COST_BASIS
    calls: list[AICallTelemetry] = Field(default_factory=list)
    stages: list[AIStageStats] = Field(default_factory=list)
    totals: AITotals = Field(default_factory=AITotals)

    def append(self, record: AICallTelemetry) -> AITelemetry:
        self.calls.append(record)
        by_stage: dict[str, Usage] = {}
        for call in self.calls:
            by_stage[call.stage] = by_stage.get(call.stage, Usage()) + call.as_usage()
        self.stages = [
            AIStageStats(stage=stage, usage=usage) for stage, usage in by_stage.items()
        ]
        self.totals = AITotals(
            stage_count=len(self.stages),
            usage=build_ai_totals(self.stages).usage,
        )
        return self


class TelemetrySink(Protocol):
    """Concrete boundary used by ``ClaudeAgent`` without importing RunCache."""

    def record_call(self, record: AICallTelemetry) -> None: ...

    def record_cache_hit(self, *, stage: str, model: str | None = None) -> None: ...


def build_ai_stage_stats(stage: str, reply: Reply | None) -> AIStageStats | None:
    if reply is None:
        return None
    return AIStageStats(stage=stage, usage=reply.usage)


def build_ai_totals(stage_stats: list[AIStageStats]) -> AITotals:
    usage = Usage()
    for stage in stage_stats:
        usage = usage + stage.usage
    return AITotals(stage_count=len(stage_stats), usage=usage)
