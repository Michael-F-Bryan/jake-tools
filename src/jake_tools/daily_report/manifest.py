from __future__ import annotations

import json
from pathlib import Path

from .coordinator import DailyReportRunResult
from .manifest_models import (
    DailyReportManifest,
    manifest_lane_result,
    manifest_lane_spec,
    manifest_lane_validation,
)
from .models import DailyReportLaneOptions, LaneName, LaneSpec
from .paths import DailyReportPaths
from .validation import LaneValidationResult


def build_run_manifest(
    *,
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
    specs: list[LaneSpec],
    run_result: DailyReportRunResult,
    validation_results: dict[LaneName, LaneValidationResult],
) -> DailyReportManifest:
    status = (
        "fail"
        if run_result.status == "fail"
        or any(not result.ok for result in validation_results.values())
        else "ok"
    )
    return DailyReportManifest(
        run_id=options.run_id,
        date=options.target_date,
        parent_session_id=options.parent_session_id,
        paths=_paths_json(paths),
        lane_specs=[manifest_lane_spec(spec) for spec in specs],
        lane_results=[
            manifest_lane_result(run_result.lanes[spec.name])
            for spec in specs
            if spec.name in run_result.lanes
        ],
        validation_results=[
            manifest_lane_validation(validation_results[spec.name])
            for spec in specs
            if spec.name in validation_results
        ],
        status=status,
    )


def write_run_manifest(manifest: DailyReportManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_and_write_run_manifest(
    *,
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
    specs: list[LaneSpec],
    run_result: DailyReportRunResult,
    validation_results: dict[LaneName, LaneValidationResult],
) -> DailyReportManifest:
    manifest = build_run_manifest(
        options=options,
        paths=paths,
        specs=specs,
        run_result=run_result,
        validation_results=validation_results,
    )
    write_run_manifest(manifest, paths.manifest)
    return manifest


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
