"""Typed token and API-rate-equivalent cost accounting for agent calls.

This module deliberately knows nothing about the Claude Agent SDK or the run
cache. The SDK seam translates its ``ResultMessage`` into these domain types;
the delegated worker persists them. ``estimated_cost_usd`` is the SDK's list-price/API-
equivalent estimate, never an invoice or subscription marginal cost.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import (
    BaseModel,
    Field,
    computed_field,
)

COST_BASIS = (
    "list-price/API-equivalent estimate; not invoice or subscription marginal cost"
)
CallStatus = Literal["success", "error", "cache_hit"]


def _merge_model_usage_values(left: Any, right: Any) -> Any:
    """Merge provider usage payloads without losing repeated-model totals."""
    if isinstance(left, dict) and isinstance(right, dict):
        keys = left.keys() | right.keys()
        return {
            key: _merge_model_usage_values(left[key], right[key])
            if key in left and key in right
            else left[key]
            if key in left
            else right[key]
            for key in keys
        }
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return left + right
    if left == right:
        return left
    # Provider metadata can legitimately disagree between calls; retaining the
    # first value is safer than silently replacing earlier evidence.
    return left


def _merge_model_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return _merge_model_usage_values(left, right)


class Usage(BaseModel):
    """What one agent call consumed.

    ``estimated_cost_usd=None`` means the provider did not price the call. It
    must not be rendered as free when non-zero tokens were consumed. A real
    zero is reserved for explicit local/cache work and cache-hit records.
    """

    model: str | None = None
    provider: str | None = Field(default=None, exclude_if=lambda value: value is None)
    model_usage: dict[str, Any] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
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
            or self.provider is not None
            or bool(self.model_usage)
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
            provider=self.provider if self.provider == other.provider else None,
            model_usage=_merge_model_usage(self.model_usage, other.model_usage),
            api_calls=self.api_calls + other.api_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )


class AICallTelemetry(BaseModel):
    """One durable attempt at the ClaudeAgent seam.

    ``attempt`` is assigned by the durable sink, rather than by a caller, so
    concurrent calls and retries receive a deterministic per-stage
    sequence. ``cache_hit`` is an explicit zero-cost event, not a model call.
    """

    stage: str
    scope: str | None = None
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


class TelemetrySink(Protocol):
    """Concrete boundary used by ``ClaudeAgent`` without importing RunCache."""

    def record_call(self, record: AICallTelemetry) -> None: ...

    def record_cache_hit(self, *, stage: str, model: str | None = None) -> None: ...
