from __future__ import annotations

import pytest

from jake_tools.ai_usage import Usage


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


def test_usage_preserves_complete_multi_model_mapping() -> None:
    usage = Usage(
        model_usage={
            "model-a": {
                "provider": "provider-a",
                "input_tokens": 10,
                "output_tokens": 2,
                "cost_usd": 0.1,
            },
            "model-b": {
                "provider": "provider-b",
                "input_tokens": 20,
                "output_tokens": 4,
                "cost_usd": 0.2,
            },
        }
    )

    assert usage.model_usage["model-a"]["provider"] == "provider-a"
    assert usage.model_usage["model-b"]["cost_usd"] == 0.2


def test_usage_add_sums_nested_usage_for_the_same_model() -> None:
    first = Usage(
        model_usage={
            "claude-opus-4-8": {
                "provider": "anthropic",
                "inputTokens": 10,
                "outputTokens": 3,
                "cacheReadInputTokens": 4,
                "costUSD": 0.1,
            }
        }
    )
    second = Usage(
        model_usage={
            "claude-opus-4-8": {
                "provider": "anthropic",
                "inputTokens": 7,
                "outputTokens": 2,
                "cacheReadInputTokens": 6,
                "costUSD": 0.2,
            }
        }
    )

    merged = first + second

    assert merged.model_usage["claude-opus-4-8"] == {
        "provider": "anthropic",
        "inputTokens": 17,
        "outputTokens": 5,
        "cacheReadInputTokens": 10,
        "costUSD": pytest.approx(0.3),
    }
