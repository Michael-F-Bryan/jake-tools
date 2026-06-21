from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, cast

from jake_tools.daily_report.coordinator import run_daily_report
from jake_tools.daily_report.lanes import build_lane_specs
from jake_tools.daily_report.models import DailyReportLaneOptions, LaneName, LaneOutput, LaneShape, LaneSpec
from jake_tools.daily_report.paths import DailyReportPaths
from jake_tools.daily_report.stages import HermesDailyReportStages
from jake_tools.hermes import AgentSpec, HermesResult
from jake_tools.prompting import StructuredPrompt


class FakeStages:
    def __init__(self, failing: set[LaneName] | None = None) -> None:
        self.failing = failing or set()
        self.calls: list[LaneName] = []

    def run_lane(
        self,
        spec: LaneSpec,
        options: DailyReportLaneOptions,
    ) -> tuple[LaneOutput, HermesResult]:
        del options
        self.calls.append(spec.name)
        if spec.name in self.failing:
            raise RuntimeError(f"boom {spec.name.value}")
        return (
            LaneOutput(
                markdown=f"# {spec.name.value}",
                findings=[spec.name.value],
                actions=["act"],
                caveats=["caveat"],
                evidence_paths=[str(spec.evidence_bundle_path)],
                cited_session_ids=["session-1"],
            ),
            HermesResult(completed=True, model=spec.model, provider=spec.provider, api_calls=1),
        )


def make_run(tmp_path: Path) -> tuple[DailyReportLaneOptions, DailyReportPaths, list[LaneSpec]]:
    paths = DailyReportPaths.for_date(tmp_path, date(2026, 6, 21))
    options = DailyReportLaneOptions(run_id="run-1", target_date="2026-06-21")
    specs = build_lane_specs(options, paths)
    return options, paths, specs


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_coordinator_writes_start_complete_events_and_lane_artefacts(tmp_path: Path) -> None:
    options, paths, specs = make_run(tmp_path)
    stages = FakeStages()

    result = run_daily_report(options=options, stages=stages, specs=specs[:1], paths=paths)

    assert result.status == "ok"
    events = read_events(paths.lane_events)
    assert [event["event"] for event in events] == ["start", "complete"]
    assert events[0]["lane"] == LaneName.SESSION_HINDSIGHT.value
    assert events[1]["status"] == "ok"

    artefact = json.loads(specs[0].artefact_path.read_text(encoding="utf-8"))
    assert artefact["markdown"] == "# session-hindsight"
    assert artefact["findings"] == ["session-hindsight"]


def test_coordinator_parallel_runner_calls_all_six_lanes(tmp_path: Path) -> None:
    options, paths, specs = make_run(tmp_path)
    stages = FakeStages()

    result = run_daily_report(options=options, stages=stages, specs=specs, paths=paths)

    assert result.status == "ok"
    assert set(result.lanes) == {spec.name for spec in specs}
    assert set(stages.calls) == {spec.name for spec in specs}
    assert len(read_events(paths.lane_events)) == 12
    assert all(spec.artefact_path.exists() for spec in specs)


def test_coordinator_records_lane_failure_and_still_returns_result(tmp_path: Path) -> None:
    options, paths, specs = make_run(tmp_path)
    failing_lane = specs[1].name
    stages = FakeStages(failing={failing_lane})

    result = run_daily_report(options=options, stages=stages, specs=specs, paths=paths)

    assert result.status == "fail"
    assert result.lanes[failing_lane].status == "fail"
    assert "RuntimeError: boom memory-candidates" == result.lanes[failing_lane].error
    assert not specs[1].artefact_path.exists()
    assert any(
        event["event"] == "complete"
        and event["lane"] == failing_lane.value
        and event["status"] == "fail"
        and "RuntimeError" in event["error"]
        for event in read_events(paths.lane_events)
    )
    successful = [spec for spec in specs if spec.name is not failing_lane]
    assert all(spec.artefact_path.exists() for spec in successful)


class FakeHermes:
    def __init__(self) -> None:
        self.structured_calls: list[StructuredPrompt[LaneOutput]] = []
        self.agent_calls: list[tuple[AgentSpec, StructuredPrompt[LaneOutput]]] = []

    def run_structured_with_result(
        self,
        prompt: StructuredPrompt[LaneOutput],
    ) -> tuple[LaneOutput, HermesResult]:
        self.structured_calls.append(prompt)
        return LaneOutput(markdown="prefed"), HermesResult(completed=True)

    def run_agent_structured(
        self,
        spec: AgentSpec,
        prompt: StructuredPrompt[LaneOutput],
    ) -> tuple[LaneOutput, HermesResult]:
        self.agent_calls.append((spec, prompt))
        return LaneOutput(markdown="worker"), HermesResult(completed=True)


def test_hermes_daily_report_stages_passes_worker_agent_spec(tmp_path: Path) -> None:
    paths = DailyReportPaths.for_date(tmp_path, date(2026, 6, 21))
    options = DailyReportLaneOptions(
        run_id="run-1",
        target_date="2026-06-21",
        parent_session_id="parent-1",
        session_db=tmp_path / "state.db",
        max_iterations=7,
    )
    worker = next(
        spec for spec in build_lane_specs(options, paths) if spec.shape is LaneShape.WORKER_AGENT
    )
    fake_hermes = FakeHermes()
    stages = HermesDailyReportStages(cast(Any, fake_hermes))

    output, result = stages.run_lane(worker, options)

    assert output.markdown == "worker"
    assert result.completed is True
    assert len(fake_hermes.agent_calls) == 1
    agent_spec, prompt = fake_hermes.agent_calls[0]
    assert prompt is worker.prompt
    assert agent_spec.model == worker.model
    assert agent_spec.provider == worker.provider
    assert agent_spec.enabled_toolsets == list(worker.enabled_toolsets)
    assert agent_spec.parent_session_id == "parent-1"
    assert agent_spec.session_db == tmp_path / "state.db"
    assert agent_spec.max_iterations == 7


def test_hermes_daily_report_stages_uses_prefed_structured_path(tmp_path: Path) -> None:
    options, paths, specs = make_run(tmp_path)
    prefed = next(spec for spec in specs if spec.shape is LaneShape.PRE_FED)
    fake_hermes = FakeHermes()
    stages = HermesDailyReportStages(cast(Any, fake_hermes))

    output, _ = stages.run_lane(prefed, options)

    assert output.markdown == "prefed"
    assert fake_hermes.structured_calls == [prefed.prompt]
    assert fake_hermes.agent_calls == []
