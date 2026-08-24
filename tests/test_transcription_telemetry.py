from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from click.testing import CliRunner

from jake_tools.ai_usage import AICallTelemetry, AITelemetry, Usage
from jake_tools.claude import ClaudeAgent, ClaudeAgentError
from jake_tools.cli import main
from jake_tools.transcription.cache import RunCache
from jake_tools.transcription.chapters import run_chapterisation
from jake_tools.transcription.models import RawTranscript, Utterance
from jake_tools.transcription.speakers import RESOLVED_TRANSCRIPT_CACHE_NAME


def _result(
    *,
    payload: object = None,
    error: bool = False,
    cost: float | None = 0.25,
    usage: dict[str, object] | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype="error_during_execution" if error else "success",
        duration_ms=1234,
        duration_api_ms=987,
        is_error=error,
        num_turns=2,
        session_id="session-telemetry",
        total_cost_usd=cost,
        usage=usage
        or {
            "input_tokens": 101,
            "output_tokens": 202,
            "cache_read_input_tokens": 303,
            "cache_creation_input_tokens": 404,
        },
        structured_output=payload,
        errors=["failed"] if error else None,
        model_usage=cast(Any, {"claude-test": {"provider": "anthropic"}}),
    )


class RoutingQuery:
    def __init__(self, *results: ResultMessage) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, ClaudeAgentOptions]] = []

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.calls.append((prompt, options))
        result = self.results.pop(0)

        async def stream() -> AsyncIterator[Message]:
            yield result

        return stream()


def test_claude_agent_records_stage_usage_with_duration_and_api_cost(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path)
    query = RoutingQuery(_result(payload={"ok": True}))
    agent = ClaudeAgent(run_query=query).with_telemetry(cache.telemetry_sink("run-1"))

    # The concrete response shape is irrelevant; this exercises the seam with
    # a result carrying every usage bucket exposed by the SDK.
    asyncio.run(agent.run("prompt", stage="minutes"))

    telemetry = cache.load("run-1", "ai_telemetry", AITelemetry)
    assert telemetry is not None
    record = telemetry.calls[0]
    assert record.stage == "minutes"
    assert record.status == "success"
    assert record.model == "claude-test"
    assert record.provider == "anthropic"
    assert record.api_calls == 2
    assert record.input_tokens == 101
    assert record.output_tokens == 202
    assert record.cache_read_tokens == 303
    assert record.cache_write_tokens == 404
    assert record.duration_ms == 1234
    assert record.duration_api_ms == 987
    assert record.estimated_cost_usd == 0.25


def test_failed_claude_agent_call_is_persisted_and_chargeable(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)
    agent = ClaudeAgent(run_query=RoutingQuery(_result(error=True))).with_telemetry(
        cache.telemetry_sink("run-2")
    )

    with pytest.raises(ClaudeAgentError):
        asyncio.run(agent.run("prompt", stage="speakers"))

    telemetry = cache.load("run-2", "ai_telemetry", AITelemetry)
    assert telemetry is not None
    assert telemetry.calls[0].status == "error"
    assert telemetry.calls[0].estimated_cost_usd == 0.25
    assert telemetry.totals.estimated_cost_usd == 0.25


def test_cache_hit_records_zero_new_api_cost_and_manifest_is_content_bound(
    tmp_path: Path,
) -> None:
    import asyncio

    cache = RunCache(tmp_path)
    transcript = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0, end=70, speaker="Ada", text="one"),
            Utterance(start=70, end=140, speaker="Bob", text="two"),
        ],
    )
    cache.store("run-3", RESOLVED_TRANSCRIPT_CACHE_NAME, transcript)
    query = RoutingQuery(
        _result(
            payload={
                "chapters": [
                    {"title": "One", "start_utterance": 0},
                    {"title": "Two", "start_utterance": 1},
                ]
            }
        ),
    )
    agent = ClaudeAgent(run_query=query).with_telemetry(cache.telemetry_sink("run-3"))

    first = asyncio.run(run_chapterisation("run-3", agent=agent, cache=cache))
    second = asyncio.run(run_chapterisation("run-3", agent=agent, cache=cache))

    assert second == first
    assert len(query.calls) == 1
    telemetry = cache.load("run-3", "ai_telemetry", AITelemetry)
    assert telemetry is not None
    assert [call.status for call in telemetry.calls] == ["success", "cache_hit"]
    assert telemetry.totals.estimated_cost_usd == 0.25
    assert (tmp_path / "run-3" / "chapterise.manifest.json").exists()


def test_unpriced_nonzero_token_usage_is_not_reported_as_free() -> None:
    record = AICallTelemetry(
        stage="adapt",
        status="success",
        usage=Usage(input_tokens=10, output_tokens=2, estimated_cost_usd=None),
    )

    assert record.estimated_cost_usd is None
    assert record.input_tokens == 10


def test_cache_creation_1h_and_5m_buckets_are_preserved(tmp_path: Path) -> None:
    cache = RunCache(tmp_path)
    query = RoutingQuery(
        _result(
            payload={"ok": True},
            usage={
                "input_tokens": 1,
                "output_tokens": 2,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 5,
                    "ephemeral_1h_input_tokens": 6,
                },
            },
        )
    )
    agent = ClaudeAgent(run_query=query).with_telemetry(cache.telemetry_sink("run-5"))

    asyncio.run(agent.run("prompt", stage="minutes"))

    telemetry = cache.load("run-5", "ai_telemetry", AITelemetry)
    assert telemetry is not None
    record = telemetry.calls[0]
    assert record.cache_creation_5m_tokens == 5
    assert record.cache_creation_1h_tokens == 6
    assert record.cache_write_tokens == 11


def test_chapterise_retry_records_separate_attempts_and_sums_cost(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path)
    transcript = RawTranscript(
        clips=[],
        utterances=[
            Utterance(start=0, end=70, speaker="Ada", text="one"),
            Utterance(start=70, end=140, speaker="Bob", text="two"),
        ],
    )
    cache.store("run-6", RESOLVED_TRANSCRIPT_CACHE_NAME, transcript)
    query = RoutingQuery(
        _result(payload={"chapters": [{"title": "Only", "start_utterance": 0}]}),
        _result(
            payload={
                "chapters": [
                    {"title": "One", "start_utterance": 0},
                    {"title": "Two", "start_utterance": 1},
                ]
            },
            cost=0.5,
        ),
    )
    agent = ClaudeAgent(run_query=query).with_telemetry(cache.telemetry_sink("run-6"))

    asyncio.run(run_chapterisation("run-6", agent=agent, cache=cache))

    telemetry = cache.load("run-6", "ai_telemetry", AITelemetry)
    assert telemetry is not None
    assert [call.attempt for call in telemetry.calls] == [1, 2]
    assert telemetry.totals.estimated_cost_usd == 0.75


def test_transcript_telemetry_cli_prints_stage_records_and_totals(
    tmp_path: Path,
) -> None:
    cache = RunCache(tmp_path)
    cache.record_ai_call(
        "run-4",
        AICallTelemetry(
            stage="minutes",
            status="success",
            usage=Usage(input_tokens=10, output_tokens=2, estimated_cost_usd=0.5),
        ),
    )

    result = CliRunner().invoke(
        main,
        ["transcript", "telemetry", "--run-id", "run-4", "--cache-root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert '"stage": "minutes"' in result.output
    assert '"estimated_cost_usd": 0.5' in result.output
    assert '"totals"' in result.output
