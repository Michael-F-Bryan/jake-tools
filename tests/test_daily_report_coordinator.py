from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from jake_tools.ai_usage import Usage
from jake_tools.daily_report.coordinator import (
    DailyReportCommandOptions,
    run_daily_report,
    run_daily_report_command,
)
from jake_tools.daily_report.himalaya import CommandResult
from jake_tools.daily_report.lanes import build_lane_specs
from jake_tools.daily_report.models import (
    DailyReportLaneOptions,
    LaneName,
    LaneOutput,
    LaneShape,
    LaneSpec,
)
from jake_tools.daily_report.paths import DailyReportPaths
from jake_tools.daily_report.stages import HermesDailyReportStages
from jake_tools.hermes import AgentSpec, Reply
from jake_tools.prompting import StructuredPrompt


class FakeStages:
    def __init__(self, failing: set[LaneName] | None = None) -> None:
        self.failing = failing or set()
        self.calls: list[LaneName] = []

    def run_lane(
        self,
        spec: LaneSpec,
        options: DailyReportLaneOptions,
    ) -> tuple[LaneOutput, Reply]:
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
            Reply(usage=Usage(model=spec.model, provider=spec.provider, api_calls=1)),
        )


def make_run(
    tmp_path: Path,
) -> tuple[DailyReportLaneOptions, DailyReportPaths, list[LaneSpec]]:
    paths = DailyReportPaths.for_date(tmp_path, date(2026, 6, 21))
    options = DailyReportLaneOptions(run_id="run-1", target_date="2026-06-21")
    specs = build_lane_specs(options, paths)
    return options, paths, specs


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_coordinator_writes_start_complete_events_and_lane_artefacts(
    tmp_path: Path,
) -> None:
    options, paths, specs = make_run(tmp_path)
    stages = FakeStages()

    result = run_daily_report(
        options=options, stages=stages, specs=specs[:1], paths=paths
    )

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


def test_coordinator_records_lane_failure_and_still_returns_result(
    tmp_path: Path,
) -> None:
    options, paths, specs = make_run(tmp_path)
    failing_lane = specs[1].name
    stages = FakeStages(failing={failing_lane})

    result = run_daily_report(options=options, stages=stages, specs=specs, paths=paths)

    assert result.status == "fail"
    assert result.lanes[failing_lane].status == "fail"
    assert result.lanes[failing_lane].error == "RuntimeError: boom memory-candidates"
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


def test_coordinator_clears_stale_artefacts_before_dispatch(tmp_path: Path) -> None:
    """Stale lane artefacts from a previous run must not survive dispatch."""
    options, paths, specs = make_run(tmp_path)

    # Write stale artefacts for every lane — as if a prior run left them.
    stale_content = json.dumps(
        {
            "markdown": "## Stale\ndata.",
            "findings": ["from prior run"],
            "actions": [],
            "caveats": [],
            "evidence_paths": [],
            "cited_session_ids": [],
        }
    )
    for spec in specs:
        spec.artefact_path.parent.mkdir(parents=True, exist_ok=True)
        spec.artefact_path.write_text(stale_content, encoding="utf-8")
        assert spec.artefact_path.exists()

    stages = FakeStages()
    result = run_daily_report(options=options, stages=stages, specs=specs, paths=paths)

    # All succeeded lanes should have fresh artefacts written by the coordinator.
    assert result.status == "ok"
    for spec in specs:
        artefact = json.loads(spec.artefact_path.read_text(encoding="utf-8"))
        assert artefact["findings"] == [spec.name.value], (
            f"{spec.name.value} artefact has stale data"
        )


class FakeHermes:
    def __init__(self) -> None:
        self.calls: list[tuple[StructuredPrompt[LaneOutput], AgentSpec | None]] = []

    def run_structured(
        self,
        prompt: StructuredPrompt[LaneOutput],
        spec: AgentSpec | None = None,
    ) -> tuple[LaneOutput, Reply]:
        self.calls.append((prompt, spec))
        if spec is not None and spec.enabled_toolsets:
            return LaneOutput(markdown="worker"), Reply(
                usage=Usage(model=spec.model, provider=spec.provider, api_calls=1)
            )
        return LaneOutput(markdown="prefed"), Reply(
            usage=Usage(
                model=spec.model if spec else None,
                provider=spec.provider if spec else None,
                api_calls=1,
            )
        )


class ValidatingFakeStages:
    def run_lane(
        self,
        spec: LaneSpec,
        options: DailyReportLaneOptions,
    ) -> tuple[LaneOutput, Reply]:
        del options
        markdown = "\n".join(f"## {section}" for section in spec.required_sections)
        return (
            LaneOutput(
                markdown=markdown,
                evidence_paths=[str(spec.evidence_bundle_path)],
            ),
            Reply(usage=Usage(model=spec.model, provider=spec.provider, api_calls=1)),
        )


def test_run_daily_report_command_creates_lane_evidence_bundles(tmp_path: Path) -> None:
    def fake_himalaya_runner(command: tuple[str, ...]) -> CommandResult:
        return CommandResult(args=command, returncode=1)

    result = run_daily_report_command(
        command_options=DailyReportCommandOptions(
            target_date=date(2026, 6, 21),
            base_dir=tmp_path,
            state_db=tmp_path / "missing-state.db",
            run_id_factory=lambda: "run-1",
        ),
        stages=ValidatingFakeStages(),
        himalaya_runner=fake_himalaya_runner,
    )

    assert result.status == "ok"
    evidence_dir = tmp_path / "daily-report-2026-06-21" / "evidence"
    for lane in LaneName:
        bundle_path = evidence_dir / f"{lane.value}.json"
        assert bundle_path.exists()
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        assert bundle["run_id"] == "run-1"
        assert bundle["target_date"] == "2026-06-21"
        assert bundle["lane"] == lane.value
        assert bundle["code_owned"] is True


def test_lane_prompt_renders_required_sections_as_exact_markdown_headings(
    tmp_path: Path,
) -> None:
    options, paths, specs = make_run(tmp_path)
    del options, paths
    spec = next(spec for spec in specs if spec.name is LaneName.SESSION_HINDSIGHT)

    rendered = spec.prompt.render()

    assert (
        "Required Markdown headings:\n## summary\n## decisions\n## risks\n## open threads"
        in rendered
    )
    assert (
        "must include every required section above as an exact Markdown heading"
        in rendered
    )
    assert "Set evidence_paths only to existing filesystem evidence files" in rendered
    assert (
        "Evidence unavailable from declared bundle; no body content reviewed."
        in rendered
    )


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
        spec
        for spec in build_lane_specs(options, paths)
        if spec.shape is LaneShape.WORKER_AGENT
    )
    fake_hermes = FakeHermes()
    stages = HermesDailyReportStages(fake_hermes)

    output, result = stages.run_lane(worker, options)

    assert output.markdown == "worker"
    assert result.usage.api_calls == 1
    assert len(fake_hermes.calls) == 1
    prompt, agent_spec = fake_hermes.calls[0]
    assert prompt is worker.prompt
    assert agent_spec is not None
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
    stages = HermesDailyReportStages(fake_hermes)

    output, _ = stages.run_lane(prefed, options)

    assert output.markdown == "prefed"
    assert len(fake_hermes.calls) == 1
    prompt, agent_spec = fake_hermes.calls[0]
    assert prompt is prefed.prompt
    assert agent_spec is not None
    assert agent_spec.model == prefed.model
    assert agent_spec.provider == prefed.provider
    assert agent_spec.enabled_toolsets == []
