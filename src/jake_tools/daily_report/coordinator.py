from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from .models import DailyReportLaneOptions, LaneName, LaneOutput, LaneSpec
from .paths import DailyReportPaths
from .stages import DailyReportStages
from ..hermes import HermesResult

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


@dataclass(frozen=True)
class DailyReportCommandOptions:
    target_date: date
    provider: str = "openrouter"
    judgement_model: str = "openrouter/auto"
    evidence_model: str = "openrouter/auto"
    timezone_name: str = "Australia/Perth"
    base_dir: Path = Path("_working")
    state_db: Path = Path.home() / ".hermes" / "state.db"
    himalaya_page_size: int = 25
    run_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex


@dataclass(frozen=True)
class DailyReportCommandResult:
    run_id: str
    status: LaneStatus
    report_path: Path
    summary_path: Path
    manifest_path: Path
    failed_lanes: list[str]
    summary: Any

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "report_path": str(self.report_path),
            "summary_path": str(self.summary_path),
            "manifest_path": str(self.manifest_path),
            "failed_lanes": self.failed_lanes,
            "summary": (
                self.summary.model_dump(mode="json")
                if hasattr(self.summary, "model_dump")
                else self.summary
            ),
        }


def run_daily_report_command(
    *,
    command_options: DailyReportCommandOptions,
    stages: DailyReportStages,
    himalaya_runner: Any | None = None,
) -> DailyReportCommandResult:
    """Run the full deterministic daily-report orchestration for the Click command."""

    from .himalaya import run_himalaya_preflight
    from .lanes import build_lane_specs
    from .manifest import build_and_write_run_manifest
    from .preflight import (
        build_session_manifest,
        write_json,
        write_session_manifest,
    )
    from .synthesis import synthesize_daily_report
    from .validation import apply_validation_to_lane_result, validate_lane_artifact

    paths = DailyReportPaths.for_date(
        command_options.base_dir, command_options.target_date
    )
    paths.create()

    options = DailyReportLaneOptions(
        run_id=command_options.run_id_factory(),
        target_date=command_options.target_date.isoformat(),
        timezone_name=command_options.timezone_name,
        provider=command_options.provider,
        model=command_options.judgement_model,
        session_db=(
            command_options.state_db if command_options.state_db.exists() else None
        ),
        extra_context={"evidence_model": command_options.evidence_model},
    )

    _write_session_manifest_evidence(
        state_db=command_options.state_db,
        target_date=command_options.target_date,
        timezone_name=command_options.timezone_name,
        path=paths.evidence / "session-manifest.json",
        build_session_manifest=build_session_manifest,
        write_session_manifest=write_session_manifest,
        write_json=write_json,
    )

    himalaya_kwargs: dict[str, Any] = {"page_size": command_options.himalaya_page_size}
    if himalaya_runner is not None:
        himalaya_kwargs["runner"] = himalaya_runner
    inbox_preflight = run_himalaya_preflight(**himalaya_kwargs)
    write_json(paths.evidence / "inbox-preflight.json", inbox_preflight.to_json())
    write_json(
        paths.evidence / "inbox-envelopes.json", _inbox_envelopes_json(inbox_preflight)
    )

    specs = build_lane_specs(options, paths, preflight=inbox_preflight)
    _write_lane_evidence_bundles(
        specs=specs, options=options, paths=paths, write_json=write_json
    )
    run_result = run_daily_report(
        options=options,
        stages=stages,
        specs=specs,
        paths=paths,
        max_workers=3,
    )

    validation_results = {
        spec.name: validate_lane_artifact(
            spec,
            paths,
            session_manifest_path=paths.evidence / "session-manifest.json",
        )
        for spec in specs
    }
    validated_lanes = {
        name: apply_validation_to_lane_result(result, validation_results[name])
        for name, result in run_result.lanes.items()
    }
    validated_status: LaneStatus = (
        "fail"
        if any(result.status == "fail" for result in validated_lanes.values())
        else "ok"
    )
    validated_run_result = DailyReportRunResult(
        run_id=run_result.run_id,
        status=validated_status,
        lanes=validated_lanes,
    )

    build_and_write_run_manifest(
        options=options,
        paths=paths,
        specs=specs,
        run_result=validated_run_result,
        validation_results=validation_results,
    )
    summary = synthesize_daily_report(
        options=options,
        paths=paths,
        specs=specs,
        run_result=validated_run_result,
        validation_results=validation_results,
    )

    return DailyReportCommandResult(
        run_id=options.run_id,
        status=summary.status,
        report_path=paths.report,
        summary_path=paths.summary,
        manifest_path=paths.manifest,
        failed_lanes=list(summary.failed_lanes),
        summary=summary,
    )


def _write_session_manifest_evidence(**kwargs: Any) -> None:
    state_db: Path = kwargs["state_db"]
    if state_db.exists():
        manifest = kwargs["build_session_manifest"](
            state_db,
            kwargs["target_date"],
            timezone_name=kwargs["timezone_name"],
        )
        kwargs["write_session_manifest"](kwargs["path"], manifest)
        return
    start_epoch = end_epoch = 0.0
    try:
        from .preflight import day_epoch_bounds

        start_epoch, end_epoch = day_epoch_bounds(
            kwargs["target_date"], kwargs["timezone_name"]
        )
    except Exception:
        pass
    kwargs["write_json"](
        kwargs["path"],
        {
            "target_date": kwargs["target_date"].isoformat(),
            "timezone": kwargs["timezone_name"],
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "missing_columns": [],
            "sessions": [],
            "state_db": str(state_db),
            "state_db_present": False,
        },
    )


def _inbox_envelopes_json(preflight: Any) -> dict[str, Any]:
    return {
        "accounts": [
            {
                "account": account.account,
                "inbox_envelopes": account.inbox_envelopes,
                "sent_envelopes": account.sent_envelopes,
            }
            for account in preflight.account_preflights
        ]
    }


def _write_lane_evidence_bundles(
    *,
    specs: list[LaneSpec],
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
    write_json: Callable[[Path, dict[str, Any]], None],
) -> None:
    for spec in specs:
        source_paths = [paths.evidence / "session-manifest.json"]
        if spec.name is LaneName.INBOX_TRIAGE:
            source_paths.extend(
                [
                    paths.evidence / "inbox-preflight.json",
                    paths.evidence / "inbox-envelopes.json",
                ]
            )

        bundle: dict[str, Any] = {
            "run_id": options.run_id,
            "target_date": options.target_date,
            "timezone": options.timezone_name,
            "lane": spec.name.value,
            "code_owned": True,
            "sources": [str(path) for path in source_paths],
            "notes": [
                "Deterministic coordinator-created evidence bundle.",
                "Lane workers may only use this bundle and explicitly enabled scoped tools.",
            ],
        }

        # Inline source content for tool-less PRE_FED lanes so the model can
        # see the data without needing filesystem tools.
        for src_path in source_paths:
            stem = src_path.stem
            if not src_path.exists():
                bundle[f"inlined_{stem}"] = None
            else:
                try:
                    bundle[f"inlined_{stem}"] = json.loads(
                        src_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    bundle[f"inlined_{stem}"] = None

        write_json(spec.evidence_bundle_path, bundle)


def _clear_lane_artefacts(specs: list[LaneSpec]) -> None:
    """Remove stale lane artefacts so validation can't read prior-run output."""
    for spec in specs:
        try:
            spec.artefact_path.unlink(missing_ok=True)
        except OSError:
            pass


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
    _clear_lane_artefacts(specs)

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

    status: LaneStatus = (
        "fail" if any(result.status == "fail" for result in results.values()) else "ok"
    )
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
