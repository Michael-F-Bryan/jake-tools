from __future__ import annotations

from pydantic import BaseModel

from .hermes import HermesResult


class AIStageStats(BaseModel):
    stage: str
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
    repair_attempted: bool = False


class AITotals(BaseModel):
    stage_count: int = 0
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


def build_ai_stage_stats(stage: str, result: HermesResult | None) -> AIStageStats | None:
    if result is None:
        return None

    return AIStageStats(
        stage=stage,
        model=result.model,
        provider=result.provider,
        api_calls=result.api_calls,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_write_tokens=result.cache_write_tokens,
        reasoning_tokens=result.reasoning_tokens,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
        repair_attempted=result.api_calls > 1,
    )


def build_ai_totals(stage_stats: list[AIStageStats]) -> AITotals:
    totals = AITotals(stage_count=len(stage_stats))
    for stage in stage_stats:
        totals.api_calls += stage.api_calls
        totals.input_tokens += stage.input_tokens
        totals.output_tokens += stage.output_tokens
        totals.cache_read_tokens += stage.cache_read_tokens
        totals.cache_write_tokens += stage.cache_write_tokens
        totals.reasoning_tokens += stage.reasoning_tokens
        totals.prompt_tokens += stage.prompt_tokens
        totals.completion_tokens += stage.completion_tokens
        totals.total_tokens += stage.total_tokens
        totals.estimated_cost_usd += stage.estimated_cost_usd
    return totals
