from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..ai_usage import AIStageStats, AITotals, build_ai_stage_stats, build_ai_totals
from .coordinator import DailyReportRunResult, LaneRunResult
from .models import DailyReportLaneOptions, LaneName, LaneOutput, LaneSpec
from .paths import DailyReportPaths
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
        db_totals = _aggregate_parent_child_from_db(
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
        if stats.estimated_cost_usd < 0:
            stats.estimated_cost_usd = 0.0
        stage_stats.append(stats)
    return build_ai_totals(stage_stats)


def _aggregate_parent_child_from_db(
    session_db: Any, parent_session_id: str
) -> AITotals | None:
    if session_db is None:
        return None
    conn: sqlite3.Connection | None = None
    owns_connection = False
    try:
        if isinstance(session_db, sqlite3.Connection):
            conn = session_db
        else:
            conn = sqlite3.connect(Path(session_db))
            owns_connection = True
        conn.row_factory = sqlite3.Row
        columns = _table_columns(conn, "sessions")
        if not {"id", "parent_session_id"}.issubset(columns):
            return None
        select_columns = [
            _select_expr(columns, "id"),
            _select_expr(columns, "model"),
            _select_expr(columns, "provider"),
            _select_expr(columns, "api_calls", "1"),
            _select_expr(columns, "input_tokens", "0"),
            _select_expr(columns, "output_tokens", "0"),
            _select_expr(columns, "cache_read_tokens", "0"),
            _select_expr(columns, "cache_write_tokens", "0"),
            _select_expr(columns, "reasoning_tokens", "0"),
            _select_expr(columns, "prompt_tokens", "0"),
            _select_expr(columns, "completion_tokens", "0"),
            _select_expr(columns, "total_tokens", "0"),
            _select_expr(columns, "estimated_cost_usd", "0"),
        ]
        rows = conn.execute(
            f"""
            SELECT {", ".join(select_columns)}
            FROM sessions
            WHERE id = ? OR parent_session_id = ?
            ORDER BY id ASC
            """,
            (parent_session_id, parent_session_id),
        ).fetchall()
        if not rows:
            return None
        return _totals_from_session_rows([dict(row) for row in rows])
    except (OSError, sqlite3.Error, TypeError, ValueError):
        return None
    finally:
        if owns_connection and conn is not None:
            conn.close()


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
    except (OSError, json.JSONDecodeError):
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
    return _totals_from_session_rows(rows)


def _totals_from_session_rows(rows: Sequence[Mapping[str, Any]]) -> AITotals:
    stage_stats = [
        AIStageStats(
            stage=str(row.get("id") or "session"),
            model=_optional_str(row.get("model")),
            provider=_optional_str(row.get("provider")),
            api_calls=_non_negative_int(row.get("api_calls"), default=1),
            input_tokens=_non_negative_int(row.get("input_tokens")),
            output_tokens=_non_negative_int(row.get("output_tokens")),
            cache_read_tokens=_non_negative_int(row.get("cache_read_tokens")),
            cache_write_tokens=_non_negative_int(row.get("cache_write_tokens")),
            reasoning_tokens=_non_negative_int(row.get("reasoning_tokens")),
            prompt_tokens=_non_negative_int(row.get("prompt_tokens")),
            completion_tokens=_non_negative_int(row.get("completion_tokens")),
            total_tokens=_non_negative_int(row.get("total_tokens")),
            estimated_cost_usd=_non_negative_float(row.get("estimated_cost_usd")),
        )
        for row in rows
    ]
    return build_ai_totals(stage_stats)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _select_expr(columns: set[str], name: str, fallback: str = "NULL") -> str:
    if name in columns:
        return name
    return f"{fallback} AS {name}"


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


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _non_negative_int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else 0


def _non_negative_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed > 0 else 0.0
