from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

from jake_tools.ai_usage import Usage
from jake_tools.ai_watch.audit import append_model
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
)
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.runner import RunnerDeps, run_ai_watch_command
from jake_tools.ai_watch.web_tools import FakeWebTools
from jake_tools.hermes import Reply


class FakeStages:
    def run_scout(self, **kwargs):
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

    def run_curate(self, **kwargs):
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


def test_runner_dry_pipeline(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id="sha256:test",
            timestamp="2026-07-02T00:00:00+00:00",
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
    hermes = MagicMock()
    deps = RunnerDeps(web_tools=fake_web, hermes=hermes)
    # monkeypatch stages via custom runner - use FakeStages by patching HermesAiWatchStages
    from jake_tools.ai_watch import runner as runner_module

    original = runner_module.HermesAiWatchStages
    runner_module.HermesAiWatchStages = lambda _hermes: FakeStages()  # type: ignore[misc]
    try:
        result = run_ai_watch_command(options=options, deps=deps)
    finally:
        runner_module.HermesAiWatchStages = original
    assert result.status == RunStatus.OK
    assert paths.digest.exists()
