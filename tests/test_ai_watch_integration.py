from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from jake_tools.ai_usage import Usage
from jake_tools.ai_watch.audit_models import DeliveryStatus
from jake_tools.ai_watch.delivery import (
    DISCORD_PAYLOAD_MAX_CHARS,
    FakeSender,
    build_discord_payload,
)
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
from jake_tools.ai_watch.runner import RunnerDeps, run_ai_watch_command
from jake_tools.ai_watch.sources import DEFAULT_SOURCE_QUERIES
from jake_tools.ai_watch.web_tools import FakeWebTools
from jake_tools.claude import Reply

HARNESS_URL = "https://www.anthropic.com/engineering/harness-design-long-running-apps"
HARNESS_QUERY = DEFAULT_SOURCE_QUERIES[0].query
ARTICLE_TEXT = (
    "Harness design with evaluator loops for long-running agent applications."
)


class IntegrationFakeStages:
    async def run_scout(
        self,
        *,
        article_text: str,
        metadata,
        options,
        interest_profile: str,
    ):
        del metadata, options, interest_profile
        return (
            ScoutOutput(
                tags=["agent-harnesses"],
                fit_score=5,
                novelty_score=4,
                practicality_score=5,
                noise_risk=1,
                evidence_quotes=["evaluator loops"],
                recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
                reason="Transferable harness evaluator pattern.",
                summary="Harness design article",
            ),
            Reply(usage=Usage(api_calls=1, model="fake-scout")),
        )

    async def run_curate(
        self,
        *,
        scout_record,
        article_text: str,
        metadata,
        options,
        interest_profile: str,
    ):
        del scout_record, article_text, metadata, options, interest_profile
        return (
            CuratorDecision(
                decision=CuratorDecisionType.SURFACE,
                lane=DigestLane.MAIN_DIGEST,
                reason=(
                    "Transferable evaluator-loop harness pattern for "
                    "long-running agent development."
                ),
                digest_summary="Harness write-up with evaluator loops.",
                obsidian_recommendation=ObsidianRecommendation(
                    should_create_note=True,
                    path="3 Resources/AI/Harness design for long-running apps.md",
                    placement_reason="Agent harness engineering",
                ),
            ),
            Reply(usage=Usage(api_calls=1, model="fake-curator")),
        )


async def test_integration_dry_run_full_pipeline(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    target_date = date(2026, 7, 2)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=tmp_path,
        dry_run=True,
        max_candidates=5,
        vault_path=vault,
        discord_target="discord:user123",
    )
    fake_web = FakeWebTools(
        search_results={
            HARNESS_QUERY: [
                SearchResult(
                    url=HARNESS_URL,
                    title="Harness design for long-running apps",
                    description="Agent harness engineering",
                    source="anthropic_engineering",
                )
            ]
        },
        extract_results={
            HARNESS_URL: ExtractResult(
                url=HARNESS_URL,
                title="Harness design for long-running apps",
                content=ARTICLE_TEXT,
            )
        },
    )
    sender = FakeSender()
    paths = AiWatchPaths.for_date(tmp_path, target_date)

    from jake_tools.ai_watch import runner as runner_module

    original_delivery = runner_module.run_delivery

    def delivery_with_sender(**kwargs):
        kwargs["sender"] = sender
        return original_delivery(**kwargs)

    runner_module.run_delivery = delivery_with_sender
    try:
        result = await run_ai_watch_command(
            options=options,
            deps=RunnerDeps(web_tools=fake_web, stages=IntegrationFakeStages()),
        )
    finally:
        runner_module.run_delivery = original_delivery

    assert result.status == RunStatus.OK
    assert result.surfaced_count == 1
    assert result.failed_stages == []

    for jsonl_path in (
        paths.candidates,
        paths.fetch_results,
        paths.scout_evaluations,
        paths.curator_decisions,
        paths.obsidian_sync,
        paths.delivery,
    ):
        assert jsonl_path.exists()
        assert jsonl_path.stat().st_size > 0

    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["counts"]["surfaced"] == 1

    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    assert summary["estimated_cost_usd"] >= 0

    digest_text = paths.digest.read_text(encoding="utf-8")
    assert "## Harness design for long-running apps" in digest_text
    assert "Why it matters:" in digest_text
    assert HARNESS_URL in digest_text
    assert "3 Resources/AI/" in digest_text

    payload = paths.root / "delivery-payload.txt"
    assert payload.exists()
    payload_text = payload.read_text(encoding="utf-8")
    assert payload_text == build_discord_payload(digest_text)
    assert len(payload_text) <= DISCORD_PAYLOAD_MAX_CHARS

    delivery_rows = [
        json.loads(line)
        for line in paths.delivery.read_text(encoding="utf-8").splitlines()
    ]
    assert delivery_rows[-1]["status"] == DeliveryStatus.DRY_RUN.value
    assert sender.calls == []

    archived = list(paths.articles.glob("*.md"))
    assert len(archived) == 1
    assert "evaluator loops" in archived[0].read_text(encoding="utf-8")
