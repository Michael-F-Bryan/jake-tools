from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..ai_usage import AIStageStats, AITotals, build_ai_stage_stats, build_ai_totals
from .coordinator import DailyReportRunResult, LaneRunResult
from .models import DailyReportLaneOptions, LaneName, LaneOutput, LaneSpec
from .paths import DailyReportPaths
from .session_accounting import (
    SessionUsageRow,
    aggregate_parent_child_from_db,
    totals_from_session_rows,
)
from .validation import LaneValidationResult

SummaryStatus = Literal["ok", "fail"]


class DailyReportSummary(BaseModel):
    date: str
    status: SummaryStatus
    paths: dict[str, str]
    lane_count: int
    failed_lanes: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    ai_totals: AITotals = Field(default_factory=AITotals)


_SOURCE_CLASS_BY_LANE: dict[LaneName, str] = {
    LaneName.SESSION_HINDSIGHT: "verified_activity",
    LaneName.MEMORY_CANDIDATES: "retrospective_current_run_failures",
    LaneName.SKILL_REVIEW: "retrospective_current_run_failures",
    LaneName.FAILURE_PATTERNS: "retrospective_current_run_failures",
    LaneName.TRANSCRIPTS_AND_DUMC: "verified_activity",
    LaneName.INBOX_TRIAGE: "envelope_leads",
}

_SOURCE_CLASS_TITLES: tuple[tuple[str, str], ...] = (
    ("verified_activity", "Verified target-date activity"),
    (
        "retrospective_current_run_failures",
        "Retrospective context and current-run failures",
    ),
    ("envelope_leads", "Envelope-only leads"),
)


def synthesize_daily_report(
    *,
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
    specs: Sequence[LaneSpec],
    run_result: DailyReportRunResult,
    validation_results: Mapping[LaneName, LaneValidationResult],
) -> DailyReportSummary:
    """Write deterministic report and summary artefacts for a daily-report run."""

    ordered_results = _ordered_lane_results(specs, run_result)
    verified = _verified_outputs(ordered_results, validation_results)
    ai_totals = aggregate_ai_usage(options=options, run_result=run_result, paths=paths)
    failed_lanes = _failed_lane_names(ordered_results, validation_results)
    status: SummaryStatus = "fail" if failed_lanes else "ok"

    summary = DailyReportSummary(
        date=options.target_date,
        status=status,
        paths=_paths_json(paths),
        lane_count=len(ordered_results),
        failed_lanes=failed_lanes,
        findings=_prefixed_items(verified, "findings"),
        actions=_prefixed_items(verified, "actions"),
        caveats=_prefixed_items(verified, "caveats"),
        ai_totals=ai_totals,
    )

    _write_report(paths.report, options, ordered_results, validation_results, verified)
    _write_summary(paths.summary, summary)
    return summary


def aggregate_ai_usage(
    *,
    options: DailyReportLaneOptions,
    run_result: DailyReportRunResult,
    paths: DailyReportPaths | None = None,
) -> AITotals:
    if options.parent_session_id:
        db_totals = aggregate_parent_child_from_db(
            options.session_db, options.parent_session_id
        )
        if db_totals is not None:
            return db_totals
        manifest_totals = _aggregate_parent_child_from_manifest(
            paths, options.parent_session_id
        )
        if manifest_totals is not None:
            return manifest_totals
    return _aggregate_from_lane_results(run_result)


def _ordered_lane_results(
    specs: Sequence[LaneSpec],
    run_result: DailyReportRunResult,
) -> list[LaneRunResult]:
    return [
        run_result.lanes[spec.name] for spec in specs if spec.name in run_result.lanes
    ]


def _verified_outputs(
    results: Sequence[LaneRunResult],
    validation_results: Mapping[LaneName, LaneValidationResult],
) -> list[tuple[LaneRunResult, LaneOutput]]:
    verified: list[tuple[LaneRunResult, LaneOutput]] = []
    for result in results:
        validation = validation_results.get(result.name)
        output = (
            validation.output
            if validation is not None and validation.ok
            else result.output
        )
        if (
            result.status == "ok"
            and validation is not None
            and validation.ok
            and output is not None
        ):
            verified.append((result, output))
    return verified


def _write_report(
    path: Path,
    options: DailyReportLaneOptions,
    results: Sequence[LaneRunResult],
    validation_results: Mapping[LaneName, LaneValidationResult],
    verified: Sequence[tuple[LaneRunResult, LaneOutput]],
) -> None:
    by_class: dict[str, list[tuple[LaneRunResult, LaneOutput]]] = {
        key: [] for key, _ in _SOURCE_CLASS_TITLES
    }
    for result, output in verified:
        by_class[_SOURCE_CLASS_BY_LANE[result.name]].append((result, output))

    lines: list[str] = [
        f"# Daily report index — {options.target_date}",
        "",
        f"Run ID: `{options.run_id}`",
        f"Status: `{_status_for_results(results)}`",
        "",
        "## Lane artefacts",
        "",
    ]
    for result in results:
        validation = validation_results.get(result.name)
        validation_status = validation.status if validation is not None else "missing"
        link = _markdown_link(result.artefact_path)
        lines.append(
            f"- `{result.status}` / validation `{validation_status}` — [{result.name.value}]({link})"
        )
    lines.append("")

    for key, title in _SOURCE_CLASS_TITLES:
        lines.extend([f"## {title}", ""])
        entries = by_class[key]
        if not entries:
            lines.extend(["- No verified lane artefacts.", ""])
            continue
        for result, output in entries:
            lines.append(f"### {result.name.value}")
            lines.append("")
            lines.append(
                f"Artefact: [{result.artefact_path.name}]({_markdown_link(result.artefact_path)})"
            )
            if output.findings:
                lines.append("")
                lines.append("Findings:")
                lines.extend(f"- {item}" for item in output.findings)
            lines.append("")

    lines.extend(["## Proposed actions", ""])
    actions = _prefixed_items(verified, "actions")
    lines.extend(
        [f"- {item}" for item in actions] or ["- No verified proposed actions."]
    )
    lines.extend(["", "## Caveats", ""])
    caveats = _prefixed_items(verified, "caveats")
    lines.extend([f"- {item}" for item in caveats] or ["- No verified caveats."])
    lines.extend(["", "## Failed lanes", ""])
    failed = [result for result in results if result.status == "fail"]
    if failed:
        for result in failed:
            error = f" — {result.error}" if result.error else ""
            lines.append(f"- `{result.name.value}`{error}")
    else:
        lines.append("- None.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(path: Path, summary: DailyReportSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _aggregate_from_lane_results(run_result: DailyReportRunResult) -> AITotals:
    stage_stats: list[AIStageStats] = []
    for name in sorted(run_result.lanes):
        stats = build_ai_stage_stats(name.value, run_result.lanes[name].hermes_result)
        if stats is None:
            continue
        if stats.usage.estimated_cost_usd < 0:
            stats.usage.estimated_cost_usd = 0.0
        stage_stats.append(stats)
    return build_ai_totals(stage_stats)


def _aggregate_parent_child_from_manifest(
    paths: DailyReportPaths | None, parent_session_id: str
) -> AITotals | None:
    if paths is None:
        return None
    manifest_path = paths.evidence / "session-manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        return None
    rows = [
        session
        for session in sessions
        if isinstance(session, dict)
        and (
            session.get("id") == parent_session_id
            or session.get("parent_session_id") == parent_session_id
        )
    ]
    if not rows:
        return None
    return totals_from_session_rows([SessionUsageRow.from_mapping(row) for row in rows])


def _prefixed_items(
    verified: Sequence[tuple[LaneRunResult, LaneOutput]],
    field_name: Literal["findings", "actions", "caveats"],
) -> list[str]:
    items: list[str] = []
    for result, output in verified:
        for item in getattr(output, field_name):
            items.append(f"{result.name.value}: {item}")
    return items


def _paths_json(paths: DailyReportPaths) -> dict[str, str]:
    return {
        "root": str(paths.root),
        "subtasks": str(paths.subtasks),
        "evidence": str(paths.evidence),
        "prompts": str(paths.prompts),
        "logs": str(paths.logs),
        "drafts": str(paths.drafts),
        "report": str(paths.report),
        "summary": str(paths.summary),
        "manifest": str(paths.manifest),
        "lane_events": str(paths.lane_events),
    }


def _status_for_results(results: Sequence[LaneRunResult]) -> SummaryStatus:
    return "fail" if any(result.status == "fail" for result in results) else "ok"


def _failed_lane_names(
    results: Sequence[LaneRunResult],
    validation_results: Mapping[LaneName, LaneValidationResult],
) -> list[str]:
    failed: list[str] = []
    for result in results:
        validation = validation_results.get(result.name)
        if result.status == "fail" or validation is None or not validation.ok:
            failed.append(result.name.value)
    return failed


def _markdown_link(path: Path) -> str:
    return str(path).replace(" ", "%20")
