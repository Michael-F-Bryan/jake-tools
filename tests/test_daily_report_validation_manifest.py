from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from jake_tools.daily_report.coordinator import DailyReportRunResult, LaneRunResult
from jake_tools.daily_report.lanes import build_lane_specs
from jake_tools.daily_report.manifest import build_and_write_run_manifest
from jake_tools.daily_report.models import DailyReportLaneOptions, LaneName, LaneOutput, LaneSpec
from jake_tools.daily_report.paths import DailyReportPaths
from jake_tools.daily_report.validation import apply_validation_to_lane_result, validate_lane_artifact
from jake_tools.hermes import HermesResult


def make_paths_and_specs(tmp_path: Path) -> tuple[DailyReportLaneOptions, DailyReportPaths, list[LaneSpec]]:
    paths = DailyReportPaths.for_date(tmp_path, date(2026, 6, 21)).create()
    options = DailyReportLaneOptions(
        run_id="run-1",
        target_date="2026-06-21",
        parent_session_id="parent-1",
    )
    specs = build_lane_specs(options, paths)
    write_session_manifest(paths, ["session-1", "parent-1"])
    return options, paths, specs


def write_session_manifest(paths: DailyReportPaths, session_ids: list[str]) -> None:
    paths.evidence.mkdir(parents=True, exist_ok=True)
    (paths.evidence / "session-manifest.json").write_text(
        json.dumps(
            {
                "target_date": "2026-06-21",
                "timezone": "Australia/Perth",
                "sessions": [{"id": session_id} for session_id in session_ids],
            }
        ),
        encoding="utf-8",
    )


def markdown_for(spec: LaneSpec) -> str:
    return "\n\n".join(f"## {heading}\nSupported detail." for heading in spec.required_sections)


def write_output(spec: LaneSpec, output: LaneOutput) -> None:
    spec.artefact_path.parent.mkdir(parents=True, exist_ok=True)
    spec.artefact_path.write_text(
        json.dumps(output.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def valid_output(spec: LaneSpec, evidence_path: Path) -> LaneOutput:
    return LaneOutput(
        markdown=markdown_for(spec),
        findings=["grounded finding"],
        actions=["proposed action"],
        caveats=["bounded caveat"],
        evidence_paths=[str(evidence_path)],
        cited_session_ids=["session-1"],
    )


def test_validation_passes_valid_lane_artefact(tmp_path: Path) -> None:
    _, paths, specs = make_paths_and_specs(tmp_path)
    spec = specs[0]
    evidence = paths.evidence / "session-hindsight-source.json"
    evidence.write_text("{}", encoding="utf-8")
    write_output(spec, valid_output(spec, evidence))

    result = validate_lane_artifact(spec, paths)

    assert result.status == "ok"
    assert result.errors == []
    assert result.output is not None


def test_validation_accepts_mild_heading_normalisation(tmp_path: Path) -> None:
    _, paths, specs = make_paths_and_specs(tmp_path)
    spec = replace(specs[0], required_sections=("Rejected/weak claims",))
    evidence = paths.evidence / "source.json"
    evidence.write_text("{}", encoding="utf-8")
    write_output(
        spec,
        LaneOutput(
            markdown="## Rejected / weak claims\nNone.",
            evidence_paths=[str(evidence)],
        ),
    )

    result = validate_lane_artifact(spec, paths)

    assert result.status == "ok"


def test_validation_catches_missing_heading(tmp_path: Path) -> None:
    _, paths, specs = make_paths_and_specs(tmp_path)
    spec = specs[0]
    evidence = paths.evidence / "source.json"
    evidence.write_text("{}", encoding="utf-8")
    write_output(
        spec,
        LaneOutput(markdown="## Summary\nOnly one section.", evidence_paths=[str(evidence)]),
    )

    result = validate_lane_artifact(spec, paths)

    assert result.status == "fail"
    assert any("required heading missing: decisions" in error for error in result.errors)


def test_validation_catches_missing_evidence(tmp_path: Path) -> None:
    _, paths, specs = make_paths_and_specs(tmp_path)
    spec = specs[0]
    write_output(spec, valid_output(spec, paths.evidence / "missing.json"))

    result = validate_lane_artifact(spec, paths)

    assert result.status == "fail"
    assert any("declared evidence missing" in error for error in result.errors)


def test_validation_catches_invented_session_id(tmp_path: Path) -> None:
    _, paths, specs = make_paths_and_specs(tmp_path)
    spec = specs[0]
    evidence = paths.evidence / "source.json"
    evidence.write_text("{}", encoding="utf-8")
    output = valid_output(spec, evidence)
    output.cited_session_ids.append("made-up-session")
    write_output(spec, output)

    result = validate_lane_artifact(spec, paths)

    assert result.status == "fail"
    assert any("made-up-session" in error for error in result.errors)


def test_validation_catches_inbox_drafting_or_resolution(tmp_path: Path) -> None:
    _, paths, specs = make_paths_and_specs(tmp_path)
    spec = next(spec for spec in specs if spec.name is LaneName.INBOX_TRIAGE)
    evidence = paths.evidence / "inbox.json"
    evidence.write_text("{}", encoding="utf-8")
    write_output(
        spec,
        LaneOutput(
            markdown=markdown_for(spec),
            actions=["Drafted a reply and resolved the thread."],
            evidence_paths=[str(evidence)],
        ),
    )

    result = validate_lane_artifact(spec, paths)

    assert result.status == "fail"
    assert any("inbox lane" in error for error in result.errors)


def test_validation_catches_tool_unavailable_without_preflight(tmp_path: Path) -> None:
    _, paths, specs = make_paths_and_specs(tmp_path)
    spec = specs[0]
    evidence = paths.evidence / "source.json"
    evidence.write_text("{}", encoding="utf-8")
    write_output(
        spec,
        LaneOutput(
            markdown=markdown_for(spec),
            caveats=["Himalaya tool unavailable."],
            evidence_paths=[str(evidence)],
        ),
    )

    result = validate_lane_artifact(spec, paths)

    assert result.status == "fail"
    assert any("preflight" in error for error in result.errors)


def test_validation_failure_can_be_represented_as_lane_status_fail_data(tmp_path: Path) -> None:
    _, paths, specs = make_paths_and_specs(tmp_path)
    spec = specs[0]
    write_output(spec, LaneOutput(markdown="{}"))
    lane_result = LaneRunResult(name=spec.name, status="ok", artefact_path=spec.artefact_path)
    validation = validate_lane_artifact(spec, paths)

    updated = apply_validation_to_lane_result(lane_result, validation)

    assert updated.status == "fail"
    assert updated.error is not None
    assert "required heading missing" in updated.error


def test_manifest_writes_expected_json(tmp_path: Path) -> None:
    options, paths, specs = make_paths_and_specs(tmp_path)
    spec = specs[0]
    evidence = paths.evidence / "source.json"
    evidence.write_text("{}", encoding="utf-8")
    output = valid_output(spec, evidence)
    write_output(spec, output)
    validation = {spec.name: validate_lane_artifact(spec, paths)}
    lane_result = LaneRunResult(
        name=spec.name,
        status="ok",
        artefact_path=spec.artefact_path,
        output=output,
        hermes_result=HermesResult(completed=True, model=spec.model, provider=spec.provider, api_calls=1),
    )
    run_result = DailyReportRunResult(
        run_id=options.run_id,
        status="ok",
        lanes={spec.name: lane_result},
    )

    manifest = build_and_write_run_manifest(
        options=options,
        paths=paths,
        specs=[spec],
        run_result=run_result,
        validation_results=validation,
    )

    payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest.status == "ok"
    assert payload["run_id"] == "run-1"
    assert payload["date"] == "2026-06-21"
    assert payload["parent_session_id"] == "parent-1"
    assert payload["paths"]["manifest"] == str(paths.manifest)
    assert payload["lane_specs"][0]["name"] == spec.name.value
    assert payload["lane_results"][0]["status"] == "ok"
    assert payload["lane_results"][0]["hermes_result"]["api_calls"] == 1
    assert payload["validation_results"] == [{"lane": spec.name.value, "status": "ok", "errors": []}]
