from __future__ import annotations

from datetime import date
from pathlib import Path

from jake_tools.ai_usage import Usage
from jake_tools.ai_watch.audit import append_model
from jake_tools.ai_watch.audit_models import DiscoveredRecord, FetchRecord, FetchStatus
from jake_tools.ai_watch.models import (
    AiWatchCommandOptions,
    ScoutOutput,
    ScoutRecommendation,
)
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.scout import run_scout
from jake_tools.hermes import Reply


class FakeStages:
    def __init__(self, output: ScoutOutput) -> None:
        self.output = output

    def run_scout(self, **kwargs):
        return self.output, Reply(usage=Usage(api_calls=1))

    def run_curate(self, **kwargs):
        raise NotImplementedError


def test_scout_output_validates_schema() -> None:
    output = ScoutOutput(
        tags=["agent-harnesses"],
        fit_score=5,
        novelty_score=4,
        practicality_score=4,
        noise_risk=1,
        evidence_quotes=["evaluator loops"],
        recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
        reason="Harness pattern",
        summary="Harness article",
    )
    assert output.fit_score == 5


def test_run_scout_sanitizes_hallucinated_quotes(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    candidate_id = "sha256:scout"
    article = "Harness design with evaluator loops for long-running agents."
    paths.article_markdown(candidate_id).write_text(article, encoding="utf-8")
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp="2026-07-02T00:00:00+00:00",
            source="test",
            url="https://example.com/harness",
            title="Harness",
        ),
    )
    append_model(
        paths.fetch_results,
        FetchRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp="2026-07-02T00:00:00+00:00",
            status=FetchStatus.OK,
            content_path=f"articles/{candidate_id}.md",
            content_hash="sha256:abc",
        ),
    )
    output = ScoutOutput(
        tags=["agent-harnesses"],
        fit_score=5,
        novelty_score=4,
        practicality_score=4,
        noise_risk=1,
        evidence_quotes=[
            "evaluator loops",
            "The article explains evaluator loop patterns",
        ],
        recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
        reason="Harness pattern",
        summary="Harness article",
    )
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    result = run_scout(options=options, paths=paths, stages=FakeStages(output))
    assert result.evaluated == 1
    lines = paths.scout_evaluations.read_text(encoding="utf-8").splitlines()
    payload = lines[0]
    assert "evaluator loops" in payload
    assert "The article explains evaluator loop patterns" not in payload


def test_run_scout_rejects_all_hallucinated_quotes(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    candidate_id = "sha256:bad-quotes"
    article = "Harness design with evaluator loops for long-running agents."
    paths.article_markdown(candidate_id).write_text(article, encoding="utf-8")
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp="2026-07-02T00:00:00+00:00",
            source="test",
            url="https://example.com/harness",
            title="Harness",
        ),
    )
    append_model(
        paths.fetch_results,
        FetchRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp="2026-07-02T00:00:00+00:00",
            status=FetchStatus.OK,
            content_path=f"articles/{candidate_id}.md",
            content_hash="sha256:abc",
        ),
    )
    output = ScoutOutput(
        tags=["agent-harnesses"],
        fit_score=5,
        novelty_score=4,
        practicality_score=4,
        noise_risk=1,
        evidence_quotes=["Completely fabricated quote not in article."],
        recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
        reason="Harness pattern",
        summary="Harness article",
    )
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    result = run_scout(options=options, paths=paths, stages=FakeStages(output))
    assert result.evaluated == 1
    payload = paths.scout_evaluations.read_text(encoding="utf-8")
    assert "evidence_quotes" in payload
    assert "Completely fabricated quote" not in payload
