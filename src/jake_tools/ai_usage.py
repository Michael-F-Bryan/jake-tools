from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, computed_field, model_validator

if TYPE_CHECKING:
    from .hermes import Reply


class Usage(BaseModel):
    model: str | None = None
    provider: str | None = None
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            model=other.model if other.model is not None else self.model,
            provider=other.provider if other.provider is not None else self.provider,
            api_calls=self.api_calls + other.api_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )

    @classmethod
    def from_run(cls, raw: dict[str, Any]) -> Usage:
        return cls(
            model=raw.get("model"),
            provider=raw.get("provider"),
            api_calls=int(raw.get("api_calls", 0) or 0),
            input_tokens=int(raw.get("input_tokens", 0) or 0),
            output_tokens=int(raw.get("output_tokens", 0) or 0),
            cache_read_tokens=int(raw.get("cache_read_tokens", 0) or 0),
            cache_write_tokens=int(raw.get("cache_write_tokens", 0) or 0),
            reasoning_tokens=int(raw.get("reasoning_tokens", 0) or 0),
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
            total_tokens=int(raw.get("total_tokens", 0) or 0),
            estimated_cost_usd=float(raw.get("estimated_cost_usd", 0.0) or 0.0),
        )


def _usage_from_flat_fields(raw: dict[str, Any]) -> Usage:
    return Usage(
        model=raw.get("model"),
        provider=raw.get("provider"),
        api_calls=int(raw.get("api_calls", 0) or 0),
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        cache_read_tokens=int(raw.get("cache_read_tokens", 0) or 0),
        cache_write_tokens=int(raw.get("cache_write_tokens", 0) or 0),
        reasoning_tokens=int(raw.get("reasoning_tokens", 0) or 0),
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        total_tokens=int(raw.get("total_tokens", 0) or 0),
        estimated_cost_usd=float(raw.get("estimated_cost_usd", 0.0) or 0.0),
    )


class AIStageStats(BaseModel):
    stage: str
    usage: Usage = Field(default_factory=Usage)
    repair_attempted: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_usage(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "usage" in data:
            return data
        merged = dict(data)
        merged["usage"] = _usage_from_flat_fields(data)
        return merged

    @computed_field
    @property
    def model(self) -> str | None:
        return self.usage.model

    @computed_field
    @property
    def provider(self) -> str | None:
        return self.usage.provider

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
    def reasoning_tokens(self) -> int:
        return self.usage.reasoning_tokens

    @computed_field
    @property
    def prompt_tokens(self) -> int:
        return self.usage.prompt_tokens

    @computed_field
    @property
    def completion_tokens(self) -> int:
        return self.usage.completion_tokens

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

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_usage(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "usage" in data:
            return data
        merged = dict(data)
        merged["usage"] = _usage_from_flat_fields(data)
        return merged

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
    def reasoning_tokens(self) -> int:
        return self.usage.reasoning_tokens

    @computed_field
    @property
    def prompt_tokens(self) -> int:
        return self.usage.prompt_tokens

    @computed_field
    @property
    def completion_tokens(self) -> int:
        return self.usage.completion_tokens

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

    return AIStageStats(
        stage=stage,
        usage=reply.usage,
        repair_attempted=reply.usage.api_calls > 1,
    )


def build_ai_totals(stage_stats: list[AIStageStats]) -> AITotals:
    usage = Usage()
    for stage in stage_stats:
        usage = usage + stage.usage
    return AITotals(stage_count=len(stage_stats), usage=usage)
