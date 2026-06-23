from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from jake_tools.daily_report.lanes import LANE_DEFINITIONS, build_lane_specs
from jake_tools.daily_report.models import (
    DailyReportLaneOptions,
    LaneName,
    LaneShape,
    LaneSpec,
)
from jake_tools.daily_report.paths import DailyReportPaths


def specs(tmp_path: Path) -> list[LaneSpec]:
    paths = DailyReportPaths.for_date(tmp_path, date(2026, 6, 21))
    options = DailyReportLaneOptions(run_id="daily-2026-06-21", target_date="2026-06-21")
    return build_lane_specs(options, paths)


def by_name(specs: list[LaneSpec]) -> dict[LaneName, LaneSpec]:
    return {spec.name: spec for spec in specs}


def test_lane_definitions_single_table_covers_all_lanes_in_order() -> None:
    assert len(LANE_DEFINITIONS) == 6
    assert [lane.name for lane in LANE_DEFINITIONS] == [
        LaneName.SESSION_HINDSIGHT,
        LaneName.MEMORY_CANDIDATES,
        LaneName.SKILL_REVIEW,
        LaneName.FAILURE_PATTERNS,
        LaneName.TRANSCRIPTS_AND_DUMC,
        LaneName.INBOX_TRIAGE,
    ]
    assert {lane.name for lane in LANE_DEFINITIONS} == set(LaneName)


def test_build_lane_specs_returns_all_six_lanes(tmp_path: Path) -> None:
    lane_specs = specs(tmp_path)

    assert [spec.name for spec in lane_specs] == [
        LaneName.SESSION_HINDSIGHT,
        LaneName.MEMORY_CANDIDATES,
        LaneName.SKILL_REVIEW,
        LaneName.FAILURE_PATTERNS,
        LaneName.TRANSCRIPTS_AND_DUMC,
        LaneName.INBOX_TRIAGE,
    ]
    assert all(spec.evidence_bundle_path.name == f"{spec.name.value}.json" for spec in lane_specs)
    assert all(spec.artefact_path.name == f"{spec.name.value}.json" for spec in lane_specs)


def test_inbox_lane_is_prefed_envelope_only_and_toolless(tmp_path: Path) -> None:
    inbox = by_name(specs(tmp_path))[LaneName.INBOX_TRIAGE]

    assert inbox.shape is LaneShape.PRE_FED
    assert inbox.enabled_toolsets == ()
    assert inbox.safety_mode == "envelope-only"

    rendered = inbox.prompt.render()
    assert "Envelope-only inbox lane" in rendered
    assert "Do not read message bodies" in rendered
    assert "draft" in rendered
    assert "send mail" in rendered


def test_worker_lanes_have_scoped_non_empty_toolsets(tmp_path: Path) -> None:
    lane_specs = specs(tmp_path)
    worker_specs = [spec for spec in lane_specs if spec.shape is LaneShape.WORKER_AGENT]

    assert {spec.name for spec in worker_specs} == {
        LaneName.SESSION_HINDSIGHT,
        LaneName.TRANSCRIPTS_AND_DUMC,
    }
    for spec in worker_specs:
        assert spec.enabled_toolsets
        assert all(toolset not in {"default", "all", "*", "broad"} for toolset in spec.enabled_toolsets)
        assert "Do not write files" in spec.prompt.render()


def test_prompt_rendering_includes_run_date_and_evidence_path(tmp_path: Path) -> None:
    lane_specs = specs(tmp_path)

    for spec in lane_specs:
        rendered = spec.prompt.render()
        assert "daily-2026-06-21" in rendered
        assert "2026-06-21" in rendered
        assert str(spec.evidence_bundle_path) in rendered
        assert "Every cited_session_id must appear in the evidence" in rendered
        assert "Agents do not write files" in rendered


def test_no_lane_uses_default_broad_toolset(tmp_path: Path) -> None:
    for spec in specs(tmp_path):
        assert "default" not in spec.enabled_toolsets
        assert "all" not in spec.enabled_toolsets
        assert "*" not in spec.enabled_toolsets
        assert "broad" not in spec.enabled_toolsets


@pytest.mark.parametrize("toolset", ["default", "all", "*", "broad"])
def test_lane_spec_rejects_broad_toolsets(tmp_path: Path, toolset: str) -> None:
    good = specs(tmp_path)[0]

    with pytest.raises(ValueError, match="broad/default toolset"):
        LaneSpec(
            name=good.name,
            shape=good.shape,
            provider=good.provider,
            model=good.model,
            model_tier=good.model_tier,
            enabled_toolsets=(toolset,),
            prompt=good.prompt,
            evidence_bundle_path=good.evidence_bundle_path,
            artefact_path=good.artefact_path,
            required_sections=good.required_sections,
            timeout_seconds=good.timeout_seconds,
            safety_mode=good.safety_mode,
        )
