from __future__ import annotations

import importlib
import json
from pathlib import Path

from click.testing import CliRunner

from jake_tools.cli import main
from jake_tools.cli.daily_report import daily_report
from jake_tools.daily_report.coordinator import DailyReportCommandResult
from jake_tools.daily_report.synthesis import DailyReportSummary

cli_module = importlib.import_module("jake_tools.cli.daily_report")


def _fake_result(
    status: str = "ok", failed_lanes: list[str] | None = None
) -> DailyReportCommandResult:
    summary = DailyReportSummary(
        date="2026-06-21",
        status=status,  # type: ignore[arg-type]
        paths={},
        lane_count=0,
        failed_lanes=failed_lanes or [],
    )
    return DailyReportCommandResult(
        run_id="run-1",
        status=status,  # type: ignore[arg-type]
        report_path=Path("/tmp/report.md"),
        summary_path=Path("/tmp/summary.json"),
        manifest_path=Path("/tmp/manifest.json"),
        failed_lanes=failed_lanes or [],
        summary=summary,
    )


def test_main_help_lists_daily_report() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "daily-report" in result.output


def test_daily_report_invalid_date_rejected() -> None:
    result = CliRunner().invoke(daily_report, ["--date", "21-06-2026"])

    assert result.exit_code != 0
    assert "must be YYYY-MM-DD" in result.output


def test_daily_report_json_emits_parseable_json_without_progress_chatter(
    monkeypatch,
) -> None:
    calls = []

    def fake_run_daily_report_command(*, command_options, stages):
        calls.append((command_options, stages))
        return _fake_result()

    monkeypatch.setattr(
        cli_module, "run_daily_report_command", fake_run_daily_report_command
    )

    result = CliRunner().invoke(daily_report, ["--date", "2026-06-21", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["summary_path"] == "/tmp/summary.json"
    assert result.output.strip().startswith("{")
    assert len(calls) == 1
    assert calls[0][0].target_date.isoformat() == "2026-06-21"


def test_daily_report_failure_summary_exits_nonzero(monkeypatch) -> None:
    def fake_run_daily_report_command(*, command_options, stages):
        del command_options, stages
        return _fake_result(status="fail", failed_lanes=["inbox-triage"])

    monkeypatch.setattr(
        cli_module, "run_daily_report_command", fake_run_daily_report_command
    )

    result = CliRunner().invoke(daily_report, ["--date", "2026-06-21"])

    assert result.exit_code == 1
    assert "status: fail" in result.output
    assert "failed lanes:" in result.output
    assert "- inbox-triage" in result.output
