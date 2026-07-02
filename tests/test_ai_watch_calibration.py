from __future__ import annotations

from datetime import date
from pathlib import Path

from jake_tools.ai_watch.audit_models import ScoutEvaluationRecord
from jake_tools.ai_watch.calibration import (
    DEFAULT_CALIBRATION_CASES_PATH,
    load_calibration_cases,
    load_calibration_fixture,
    resolve_calibration_extract,
)
from jake_tools.ai_watch.collect import run_collect
from jake_tools.ai_watch.models import (
    AiWatchCommandOptions,
    ExtractResult,
    ScoutRecommendation,
)
from jake_tools.ai_watch.paths import AiWatchPaths
from jake_tools.ai_watch.stages import CuratorPrompt
from jake_tools.ai_watch.state import SeenIndex
from jake_tools.ai_watch.web_tools import FakeWebTools


def test_calibration_only_collect_does_not_skip_seen_urls(tmp_path: Path) -> None:
    cases_path = tmp_path / "calibration-cases.json"
    cases_path.write_text(
        DEFAULT_CALIBRATION_CASES_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cases = load_calibration_cases(cases_path)
    assert len(cases) >= 1

    paths = AiWatchPaths.for_date(tmp_path, date(2026, 7, 2)).create()
    state = SeenIndex(paths.state_root)
    for case in cases:
        state.record_seen(
            candidate_id="sha256:seen",
            url=case.url,
            title=case.title_hint,
            source="calibration",
            content_path="articles/sha256:seen.md",
        )

    options = AiWatchCommandOptions(
        target_date=date(2026, 7, 2),
        base_dir=tmp_path,
        calibration_only=True,
        max_candidates=3,
    )
    result = run_collect(
        options=options,
        paths=paths,
        state=state,
        web_tools=FakeWebTools(),
        fixtures_path=cases_path,
    )

    assert result.skipped_seen == 0
    assert result.discovered == len(cases)


def test_curator_prompt_includes_calibration_replay_instruction() -> None:
    scout_record = ScoutEvaluationRecord(
        run_id="2026-07-02",
        candidate_id="sha256:test",
        timestamp="2026-07-02T00:00:00+00:00",
        model="gpt-5.5",
        fit_score=4,
        novelty_score=4,
        practicality_score=4,
        noise_risk=2,
        recommendation=ScoutRecommendation.PROMOTE_TO_CURATOR,
        reason="Strong fit for agent harness patterns.",
        summary="Harness design article.",
    )
    rendered = CuratorPrompt(
        interest_profile="profile",
        scout_record=scout_record,
        metadata=None,
        article_text="Article body",
        calibration_replay=True,
    ).render()

    assert "Calibration replay mode" in rendered
    assert "Do NOT return duplicate" in rendered


def test_resolve_calibration_extract_uses_fixture_when_live_extract_fails() -> None:
    cases = load_calibration_cases()
    generative_case = next(case for case in cases if case.id == "generative-ui")
    fixture = load_calibration_fixture(generative_case)
    assert fixture is not None
    assert len(fixture.content) >= 500

    resolved = resolve_calibration_extract(
        url=generative_case.url,
        title=generative_case.title_hint,
        extracted=ExtractResult(
            url=generative_case.url,
            title=generative_case.title_hint,
            content="stub",
            status="fail",
            error="404",
        ),
        calibration_cases=cases,
    )
    assert resolved.content == fixture.content
