from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from jake_tools.daily_report.models import DailyReportLaneOptions, LaneName, LaneOutput, LaneSpec
from jake_tools.daily_report.paths import DailyReportPaths
from jake_tools.daily_report.stages import DailyReportStages
from jake_tools.hermes import HermesResult

LaneStatus = Literal["ok", "fail"]


@dataclass(frozen=True)
class LaneRunResult:
    name: LaneName
    status: LaneStatus
    artefact_path: Path
    output: LaneOutput | None = None
    hermes_result: HermesResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class DailyReportRunResult:
    run_id: str
    status: LaneStatus
    lanes: dict[LaneName, LaneRunResult]


def run_daily_report(
    *,
    options: DailyReportLaneOptions,
    stages: DailyReportStages,
    specs: list[LaneSpec],
    paths: DailyReportPaths,
    max_workers: int = 3,
) -> DailyReportRunResult:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    paths.create()
    paths.lane_events.parent.mkdir(parents=True, exist_ok=True)
    paths.lane_events.write_text("", encoding="utf-8")

    results: dict[LaneName, LaneRunResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for spec in specs:
            _write_event(
                paths.lane_events,
                _event(options.run_id, spec.name, "start"),
            )
            futures[executor.submit(_run_one_lane, stages, options, spec)] = spec

        for future in as_completed(futures):
            spec = futures[future]
            try:
                output, hermes_result = future.result()
                _write_lane_artefact(spec.artefact_path, output)
                result = LaneRunResult(
                    name=spec.name,
                    status="ok",
                    artefact_path=spec.artefact_path,
                    output=output,
                    hermes_result=hermes_result,
                )
                _write_event(
                    paths.lane_events,
                    _event(options.run_id, spec.name, "complete", status="ok"),
                )
            except Exception as error:
                result = LaneRunResult(
                    name=spec.name,
                    status="fail",
                    artefact_path=spec.artefact_path,
                    error=f"{type(error).__name__}: {error}",
                )
                _write_event(
                    paths.lane_events,
                    _event(
                        options.run_id,
                        spec.name,
                        "complete",
                        status="fail",
                        error=result.error,
                    ),
                )
            results[spec.name] = result

    status: LaneStatus = "fail" if any(result.status == "fail" for result in results.values()) else "ok"
    return DailyReportRunResult(run_id=options.run_id, status=status, lanes=results)


def _run_one_lane(
    stages: DailyReportStages,
    options: DailyReportLaneOptions,
    spec: LaneSpec,
) -> tuple[LaneOutput, HermesResult]:
    return stages.run_lane(spec, options)


def _event(
    run_id: str,
    lane: LaneName,
    event: Literal["start", "complete"],
    *,
    status: LaneStatus | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "lane": lane.value,
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if status is not None:
        payload["status"] = status
    if error is not None:
        payload["error"] = error
    return payload


def _write_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_lane_artefact(path: Path, output: LaneOutput) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
