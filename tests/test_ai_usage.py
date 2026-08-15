from __future__ import annotations

from jake_tools.ai_usage import AIStageStats, AITotals, Usage


def _usage() -> Usage:
    return Usage(
        model="claude-opus-4-8",
        api_calls=3,
        input_tokens=100,
        output_tokens=40,
        cache_read_tokens=900,
        cache_write_tokens=20,
        estimated_cost_usd=1.23,
    )


def test_ai_stage_stats_json_is_flattened_and_nested_identically_to_before() -> None:
    """Guard test: captured from the pre-refactor `@computed_field` output.

    Transcript manifests retain this shape, so the flattened keys and the
    nested ``usage`` object must both survive the serializer implementation.
    """
    stats = AIStageStats(stage="scout", usage=_usage())

    assert stats.model_dump(mode="json") == {
        "stage": "scout",
        "usage": {
            "model": "claude-opus-4-8",
            "api_calls": 3,
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_read_tokens": 900,
            "cache_write_tokens": 20,
            "estimated_cost_usd": 1.23,
            "total_tokens": 140,
        },
        "model": "claude-opus-4-8",
        "api_calls": 3,
        "input_tokens": 100,
        "output_tokens": 40,
        "cache_read_tokens": 900,
        "cache_write_tokens": 20,
        "estimated_cost_usd": 1.23,
        "total_tokens": 140,
    }


def test_ai_totals_json_is_flattened_and_nested_identically_to_before() -> None:
    """Guard test: captured from the pre-refactor `@computed_field` output."""
    totals = AITotals(stage_count=2, usage=_usage())

    assert totals.model_dump(mode="json") == {
        "stage_count": 2,
        "usage": {
            "model": "claude-opus-4-8",
            "api_calls": 3,
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_read_tokens": 900,
            "cache_write_tokens": 20,
            "estimated_cost_usd": 1.23,
            "total_tokens": 140,
        },
        "api_calls": 3,
        "input_tokens": 100,
        "output_tokens": 40,
        "cache_read_tokens": 900,
        "cache_write_tokens": 20,
        "estimated_cost_usd": 1.23,
        "total_tokens": 140,
    }


def test_ai_totals_round_trips_through_json() -> None:
    """Flattened keys must not shadow the nested typed usage field."""
    totals = AITotals(stage_count=2, usage=_usage())

    round_tripped = AITotals.model_validate_json(totals.model_dump_json())

    assert round_tripped == totals


def test_usage_add_keeps_model_when_both_sides_agree() -> None:
    a = Usage(model="claude-opus-4-8", input_tokens=10)
    b = Usage(model="claude-opus-4-8", input_tokens=5)

    assert (a + b).model == "claude-opus-4-8"


def test_usage_add_drops_model_when_sides_disagree() -> None:
    """A totals-style accumulation across differently-modelled stages must not
    silently misreport the total's model as whichever stage ran last."""
    a = Usage(model="claude-opus-4-8", input_tokens=10)
    b = Usage(model="claude-haiku-4-5", input_tokens=5)

    assert (a + b).model is None


def test_usage_add_drops_model_when_only_one_side_has_one() -> None:
    a = Usage(model="claude-opus-4-8", input_tokens=10)
    b = Usage(model=None, input_tokens=5)

    assert (a + b).model is None
