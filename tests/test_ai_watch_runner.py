from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from jake_tools.ai_usage import Usage
from jake_tools.ai_watch.audit_models import DiscoveredRecord
from jake_tools.ai_watch.models import (
    AiWatchCommandOptions,
    CuratorDecision,
    CuratorDecisionType,
    DigestLane,
    ExtractResult,
    ObsidianRecommendation,
    RunStatus,
    ScoutOutput,
    ScoutRecommendation,
    SearchResult,
)
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.records import append_model
from jake_tools.ai_watch.runner import RunnerDeps, run_ai_watch_command
from jake_tools.ai_watch.sources import DEFAULT_SOURCE_QUERIES
from jake_tools.ai_watch.web_tools import FakeWebTools
from jake_tools.claude import Reply


class FakeStages:
    async def run_scout(self, **kwargs):
        return (
            ScoutOutput(
                tags=["agent-harnesses"],
                fit_score=5,
                novelty_score=4,
                practicality_score=5,
                noise_risk=1,
                evidence_quotes=["evaluator loops"],
                recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
                reason="Harness pattern",
                summary="Harness article",
            ),
            Reply(usage=Usage(api_calls=1)),
        )

    async def run_curate(self, **kwargs):
        return (
            CuratorDecision(
                decision=CuratorDecisionType.SURFACE,
                lane=DigestLane.MAIN_DIGEST,
                reason="Transferable harness evaluator pattern for long-running agents.",
                digest_summary="Harness write-up with evaluator loops.",
                obsidian_recommendation=ObsidianRecommendation(
                    should_create_note=True,
                    path="3 Resources/AI/Harness design.md",
                    placement_reason="AI engineering",
                ),
            ),
            Reply(usage=Usage(api_calls=1)),
        )


async def test_runner_dry_pipeline(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
            source="test",
            url="https://example.com/harness",
            title="Harness",
        ),
    )
    fake_web = FakeWebTools(
        search_results={},
        extract_results={
            "https://example.com/harness": ExtractResult(
                url="https://example.com/harness",
                title="Harness",
                content="Harness design with evaluator loops for agents.",
            )
        },
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        dry_run=True,
        max_candidates=5,
    )
    deps = RunnerDeps(web_tools=fake_web, stages=FakeStages())
    result = await run_ai_watch_command(options=options, deps=deps)
    assert result.status == RunStatus.OK
    assert paths.digest.exists()


class CostlyScoutStages:
    """Scout spends the entire cost cap on its own; curate must never run."""

    async def run_scout(self, **kwargs):
        return (
            ScoutOutput(
                tags=["agent-harnesses"],
                fit_score=5,
                novelty_score=4,
                practicality_score=5,
                noise_risk=1,
                evidence_quotes=["evaluator loops"],
                recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
                reason="Harness pattern",
                summary="Harness article",
            ),
            Reply(usage=Usage(api_calls=1, estimated_cost_usd=5.0)),
        )

    async def run_curate(self, **kwargs):
        raise AssertionError("curate must not run once the cost cap is exceeded")


def _seeded_paths(tmp_path: Path) -> AiWatchPaths:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
            source="test",
            url="https://example.com/harness",
            title="Harness",
        ),
    )
    return paths


async def test_runner_skips_curate_when_scout_alone_exceeds_cost_cap(
    tmp_path: Path,
) -> None:
    # collect() truncates and repopulates candidates.jsonl from search results,
    # so the candidate must reach fetch/scout via a search hit, not by
    # pre-seeding candidates.jsonl (collect() would wipe that seed).
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    query = DEFAULT_SOURCE_QUERIES[0].query
    fake_web = FakeWebTools(
        search_results={
            query: [
                SearchResult(
                    url="https://example.com/harness",
                    title="Harness",
                    description="harness",
                    source="test",
                )
            ]
        },
        extract_results={
            "https://example.com/harness": ExtractResult(
                url="https://example.com/harness",
                title="Harness",
                content="Harness design with evaluator loops for agents.",
            )
        },
    )
    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        dry_run=True,
        max_candidates=5,
        cost_cap_usd=1.0,
    )
    deps = RunnerDeps(web_tools=fake_web, stages=CostlyScoutStages())

    result = await run_ai_watch_command(options=options, deps=deps)

    assert result.status == RunStatus.FAIL
    curate_failure = next(f for f in result.failed_stages if f.stage == "curate")
    assert "cost cap" in curate_failure.error
    assert paths.curator_decisions.read_text(encoding="utf-8") == ""


async def test_runner_failure_records_carry_stage_and_message(tmp_path: Path) -> None:
    _seeded_paths(tmp_path)

    class RaisingWebTools:
        def search(self, query: str, *, limit: int = 5):
            del query, limit
            raise RuntimeError("search backend unavailable")

        def extract(self, urls: list[str], *, char_limit: int = 15000):
            del urls, char_limit
            return []

    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2), base_dir=tmp_path, dry_run=True
    )
    deps = RunnerDeps(web_tools=RaisingWebTools())

    result = await run_ai_watch_command(options=options, deps=deps)

    assert result.status == RunStatus.FAIL
    collect_failure = next(f for f in result.failed_stages if f.stage == "collect")
    assert collect_failure.error == "search backend unavailable"
