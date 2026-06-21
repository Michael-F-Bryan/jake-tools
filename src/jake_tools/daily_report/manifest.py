from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jake_tools.daily_report.coordinator import DailyReportRunResult, LaneRunResult
from jake_tools.daily_report.models import DailyReportLaneOptions, LaneName, LaneSpec
from jake_tools.daily_report.paths import DailyReportPaths
from jake_tools.daily_report.validation import LaneValidationResult


@dataclass(frozen=True)
class DailyReportManifest:
    run_id: str
    date: str
    parent_session_id: str | None
    paths: dict[str, str]
    lane_specs: list[dict[str, Any]]
    lane_results: list[dict[str, Any]]
    validation_results: list[dict[str, Any]]
    status: str

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "date": self.date,
            "parent_session_id": self.parent_session_id,
            "paths": self.paths,
            "lane_specs": self.lane_specs,
            "lane_results": self.lane_results,
            "validation_results": self.validation_results,
            "status": self.status,
        }


def build_run_manifest(
    *,
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
    specs: list[LaneSpec],
    run_result: DailyReportRunResult,
    validation_results: dict[LaneName, LaneValidationResult],
) -> DailyReportManifest:
    status = "fail" if run_result.status == "fail" or any(not result.ok for result in validation_results.values()) else "ok"
    return DailyReportManifest(
        run_id=options.run_id,
        date=options.target_date,
        parent_session_id=options.parent_session_id,
        paths=_paths_json(paths),
        lane_specs=[_lane_spec_json(spec) for spec in specs],
        lane_results=[
            _lane_result_json(run_result.lanes[spec.name])
            for spec in specs
            if spec.name in run_result.lanes
        ],
        validation_results=[
            validation_results[spec.name].to_json()
            for spec in specs
            if spec.name in validation_results
        ],
        status=status,
    )


def write_run_manifest(manifest: DailyReportManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_json(), indent=2, sort_keys=True),
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


def _lane_spec_json(spec: LaneSpec) -> dict[str, Any]:
    return {
        "name": spec.name.value,
        "shape": spec.shape.value,
        "provider": spec.provider,
        "model": spec.model,
        "model_tier": spec.model_tier,
        "enabled_toolsets": list(spec.enabled_toolsets),
        "evidence_bundle_path": str(spec.evidence_bundle_path),
        "artefact_path": str(spec.artefact_path),
        "required_sections": list(spec.required_sections),
        "timeout_seconds": spec.timeout_seconds,
        "safety_mode": spec.safety_mode,
    }


def _lane_result_json(result: LaneRunResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": result.name.value,
        "status": result.status,
        "artefact_path": str(result.artefact_path),
        "error": result.error,
    }
    if result.output is not None:
        payload["output"] = result.output.model_dump(mode="json")
    if result.hermes_result is not None:
        payload["hermes_result"] = result.hermes_result.model_dump(mode="json")
    return payload
