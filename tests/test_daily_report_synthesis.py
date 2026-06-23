from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

from jake_tools.ai_usage import Usage
from jake_tools.daily_report.coordinator import DailyReportRunResult, LaneRunResult
from jake_tools.daily_report.lanes import build_lane_specs
from jake_tools.daily_report.models import (
    DailyReportLaneOptions,
    LaneName,
    LaneOutput,
    LaneSpec,
)
from jake_tools.daily_report.paths import DailyReportPaths
from jake_tools.daily_report.synthesis import (
    DailyReportSummary,
    aggregate_ai_usage,
    synthesize_daily_report,
)
from jake_tools.daily_report.validation import LaneValidationResult
from jake_tools.hermes import Reply


def make_paths_options_specs(
    tmp_path: Path,
    *,
    parent_session_id: str | None = None,
    session_db: Path | None = None,
) -> tuple[DailyReportLaneOptions, DailyReportPaths, list[LaneSpec]]:
    paths = DailyReportPaths.for_date(tmp_path, date(2026, 6, 21)).create()
    options = DailyReportLaneOptions(
        run_id="run-1",
        target_date="2026-06-21",
        parent_session_id=parent_session_id,
        session_db=session_db,
    )
    return options, paths, build_lane_specs(options, paths)


def lane_result(
    spec: LaneSpec,
    *,
    status: str = "ok",
    output: LaneOutput | None = None,
    hermes_result: Reply | None = None,
    error: str | None = None,
) -> LaneRunResult:
    return LaneRunResult(
        name=spec.name,
        status=status,  # pyright: ignore[reportArgumentType]
        artefact_path=spec.artefact_path,
        output=output,
        hermes_result=hermes_result,
        error=error,
    )


def validation(
    spec: LaneSpec, output: LaneOutput, *, ok: bool = True
) -> LaneValidationResult:
    return LaneValidationResult(
        lane=spec.name,
        status="ok" if ok else "fail",
        errors=[] if ok else ["bad lane"],
        output=output,
    )


def test_returned_lane_usage_totals_ignore_negative_costs(tmp_path: Path) -> None:
    options, paths, specs = make_paths_options_specs(tmp_path)
    first, second = specs[:2]
    run_result = DailyReportRunResult(
        run_id="run-1",
        status="ok",
        lanes={
            first.name: lane_result(
                first,
                hermes_result=Reply(
                    usage=Usage(
                        api_calls=1,
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                        estimated_cost_usd=0.25,
                    ),
                ),
            ),
            second.name: lane_result(
                second,
                hermes_result=Reply(
                    usage=Usage(
                        api_calls=2,
                        input_tokens=3,
                        output_tokens=4,
                        total_tokens=7,
                        estimated_cost_usd=-9.0,
                    ),
                ),
            ),
        },
    )

    totals = aggregate_ai_usage(options=options, run_result=run_result, paths=paths)

    assert totals.stage_count == 2
    assert totals.api_calls == 3
    assert totals.input_tokens == 13
    assert totals.output_tokens == 24
    assert totals.total_tokens == 37
    assert totals.estimated_cost_usd == 0.25


def test_parent_child_sqlite_usage_aggregation_prefers_session_family(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY,
          parent_session_id TEXT,
          model TEXT,
          provider TEXT,
          api_calls INTEGER,
          input_tokens INTEGER,
          output_tokens INTEGER,
          total_tokens INTEGER,
          estimated_cost_usd REAL
        )
        """
    )
    conn.executemany(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("parent-1", None, "m", "openrouter", 1, 1, 2, 3, 0.10),
            ("child-a", "parent-1", "m", "openrouter", 2, 10, 20, 30, 0.30),
            ("child-b", "parent-1", "m", "openrouter", 1, 7, 8, 15, -1.00),
            ("other", None, "m", "openrouter", 99, 99, 99, 99, 99.00),
        ],
    )
    conn.commit()
    conn.close()
    options, paths, specs = make_paths_options_specs(
        tmp_path, parent_session_id="parent-1", session_db=db_path
    )
    fallback = DailyReportRunResult(
        run_id="run-1",
        status="ok",
        lanes={
            specs[0].name: lane_result(
                specs[0],
                hermes_result=Reply(
                    usage=Usage(
                        api_calls=99,
                        total_tokens=99,
                        estimated_cost_usd=99.0,
                    )
                ),
            )
        },
    )

    totals = aggregate_ai_usage(options=options, run_result=fallback, paths=paths)

    assert totals.stage_count == 3
    assert totals.api_calls == 4
    assert totals.input_tokens == 18
    assert totals.output_tokens == 30
    assert totals.total_tokens == 48
    assert totals.estimated_cost_usd == 0.40


def test_report_index_contains_lane_links_statuses_and_separates_sources(
    tmp_path: Path,
) -> None:
    options, paths, specs = make_paths_options_specs(tmp_path)
    session_lane = next(
        spec for spec in specs if spec.name is LaneName.SESSION_HINDSIGHT
    )
    inbox_lane = next(spec for spec in specs if spec.name is LaneName.INBOX_TRIAGE)
    failure_lane = next(
        spec for spec in specs if spec.name is LaneName.FAILURE_PATTERNS
    )
    session_output = LaneOutput(
        markdown="session",
        findings=["worked on coordinator"],
        actions=["ship synthesis"],
    )
    inbox_output = LaneOutput(
        markdown="inbox", findings=["subject-only lead"], caveats=["envelope only"]
    )
    failure_output = LaneOutput(
        markdown="failures",
        findings=["retry pattern"],
        actions=["tighten prompt"],
        caveats=["partial evidence"],
    )
    run_result = DailyReportRunResult(
        run_id="run-1",
        status="ok",
        lanes={
            session_lane.name: lane_result(session_lane, output=session_output),
            inbox_lane.name: lane_result(inbox_lane, output=inbox_output),
            failure_lane.name: lane_result(failure_lane, output=failure_output),
        },
    )

    synthesize_daily_report(
        options=options,
        paths=paths,
        specs=[session_lane, failure_lane, inbox_lane],
        run_result=run_result,
        validation_results={
            session_lane.name: validation(session_lane, session_output),
            inbox_lane.name: validation(inbox_lane, inbox_output),
            failure_lane.name: validation(failure_lane, failure_output),
        },
    )

    report = paths.report.read_text(encoding="utf-8")
    assert "## Lane artefacts" in report
    assert "`ok` / validation `ok` — [session-hindsight]" in report
    assert str(session_lane.artefact_path) in report
    assert "## Verified target-date activity" in report
    assert "session-hindsight" in report
    assert "## Retrospective context and current-run failures" in report
    assert "failure-patterns" in report
    assert "## Envelope-only leads" in report
    assert "inbox-triage" in report
    assert "## Proposed actions" in report
    assert "session-hindsight: ship synthesis" in report
    assert "failure-patterns: tighten prompt" in report
    assert "## Caveats" in report
    assert "inbox-triage: envelope only" in report


def test_summary_json_shape_and_failed_lanes_counted(tmp_path: Path) -> None:
    options, paths, specs = make_paths_options_specs(tmp_path)
    ok_spec, failed_spec = specs[:2]
    output = LaneOutput(
        markdown="ok", findings=["finding"], actions=["action"], caveats=["caveat"]
    )
    run_result = DailyReportRunResult(
        run_id="run-1",
        status="fail",
        lanes={
            ok_spec.name: lane_result(
                ok_spec,
                output=output,
                hermes_result=Reply(usage=Usage(api_calls=1, total_tokens=3)),
            ),
            failed_spec.name: lane_result(failed_spec, status="fail", error="boom"),
        },
    )

    summary = synthesize_daily_report(
        options=options,
        paths=paths,
        specs=[ok_spec, failed_spec],
        run_result=run_result,
        validation_results={ok_spec.name: validation(ok_spec, output)},
    )

    payload = json.loads(paths.summary.read_text(encoding="utf-8"))
    parsed = DailyReportSummary.model_validate(payload)
    assert summary == parsed
    assert parsed.date == "2026-06-21"
    assert parsed.status == "fail"
    assert parsed.lane_count == 2
    assert parsed.failed_lanes == [failed_spec.name.value]
    assert parsed.findings == [f"{ok_spec.name.value}: finding"]
    assert parsed.actions == [f"{ok_spec.name.value}: action"]
    assert parsed.caveats == [f"{ok_spec.name.value}: caveat"]
    assert parsed.paths["report"] == str(paths.report)
    assert parsed.ai_totals.stage_count == 1
    assert parsed.ai_totals.total_tokens == 3
