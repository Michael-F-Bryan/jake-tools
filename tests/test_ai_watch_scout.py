from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from jake_tools.ai_usage import Usage
from jake_tools.ai_watch.audit_models import (
    DiscoveredRecord,
    FetchRecord,
    FetchStatus,
    ScoutFailureRecord,
)
from jake_tools.ai_watch.models import (
    AiWatchCommandOptions,
    AiWatchStageError,
    ScoutOutput,
    ScoutRecommendation,
)
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.records import append_model, read_models
from jake_tools.ai_watch.scout import run_scout
from jake_tools.claude import Reply


class FakeStages:
    def __init__(self, output: ScoutOutput) -> None:
        self.output = output

    async def run_scout(self, **kwargs):
        return self.output, Reply(usage=Usage(api_calls=1))

    async def run_curate(self, **kwargs):
        raise NotImplementedError


class PartiallyFailingStages:
    """Raises for one specific article's text, succeeds for every other."""

    def __init__(self, output: ScoutOutput, *, fail_article_text: str) -> None:
        self.output = output
        self.fail_article_text = fail_article_text

    async def run_scout(self, *, article_text: str, **kwargs):
        del kwargs
        if article_text == self.fail_article_text:
            raise RuntimeError("malformed scout reply")
        return self.output, Reply(usage=Usage(api_calls=1))

    async def run_curate(self, **kwargs):
        raise NotImplementedError


def _seed_fetched_candidate(
    paths: AiWatchPaths, candidate_id: str, article: str
) -> None:
    paths.article_markdown(candidate_id).write_text(article, encoding="utf-8")
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
            source="test",
            url=f"https://example.com/{candidate_id}",
            title=candidate_id,
        ),
    )
    append_model(
        paths.fetch_results,
        FetchRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
            status=FetchStatus.OK,
            content_path=f"articles/{candidate_id}.md",
            content_hash="sha256:abc",
        ),
    )


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


async def test_run_scout_sanitizes_hallucinated_quotes(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    candidate_id = "sha256:scout"
    article = "Harness design with evaluator loops for long-running agents."
    paths.article_markdown(candidate_id).write_text(article, encoding="utf-8")
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
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
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
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
    result = await run_scout(options=options, paths=paths, stages=FakeStages(output))
    assert result.evaluated == 1
    lines = paths.scout_evaluations.read_text(encoding="utf-8").splitlines()
    payload = lines[0]
    assert "evaluator loops" in payload
    assert "The article explains evaluator loop patterns" not in payload


async def test_run_scout_rejects_all_hallucinated_quotes(tmp_path: Path) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    candidate_id = "sha256:bad-quotes"
    article = "Harness design with evaluator loops for long-running agents."
    paths.article_markdown(candidate_id).write_text(article, encoding="utf-8")
    append_model(
        paths.candidates,
        DiscoveredRecord(
            run_id="2026-07-02",
            candidate_id=candidate_id,
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
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
            timestamp=datetime.fromisoformat("2026-07-02T00:00:00+00:00"),
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
    result = await run_scout(options=options, paths=paths, stages=FakeStages(output))
    assert result.evaluated == 1
    payload = paths.scout_evaluations.read_text(encoding="utf-8")
    assert "evidence_quotes" in payload
    assert "Completely fabricated quote" not in payload


def _valid_output() -> ScoutOutput:
    return ScoutOutput(
        tags=["agent-harnesses"],
        fit_score=5,
        novelty_score=4,
        practicality_score=4,
        noise_risk=1,
        evidence_quotes=[],
        recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
        reason="Harness pattern",
        summary="Harness article",
    )


async def test_run_scout_records_per_candidate_failure_and_continues(
    tmp_path: Path,
) -> None:
    """One candidate's scout call raising must not discard the results
    already earned (and paid for) by every other candidate in the batch."""
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    failing_article = "Article that triggers a malformed scout reply."
    _seed_fetched_candidate(paths, "sha256:bad", failing_article)
    _seed_fetched_candidate(paths, "sha256:good", "A good, working article body.")
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    stages = PartiallyFailingStages(_valid_output(), fail_article_text=failing_article)

    result = await run_scout(options=options, paths=paths, stages=stages)

    assert result.evaluated == 1
    assert result.failed == 1
    failures = read_models(paths.scout_failures, ScoutFailureRecord)
    assert [failure.candidate_id for failure in failures] == ["sha256:bad"]
    assert "malformed scout reply" in failures[0].errors[0]


async def test_run_scout_raises_stage_error_when_every_candidate_fails(
    tmp_path: Path,
) -> None:
    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    failing_article = "The only candidate, and it fails."
    _seed_fetched_candidate(paths, "sha256:only", failing_article)
    options = AiWatchCommandOptions(target_date=date(2026, 7, 2), base_dir=tmp_path)
    stages = PartiallyFailingStages(_valid_output(), fail_article_text=failing_article)

    with pytest.raises(AiWatchStageError) as excinfo:
        await run_scout(options=options, paths=paths, stages=stages)

    assert excinfo.value.stage == "scout"
