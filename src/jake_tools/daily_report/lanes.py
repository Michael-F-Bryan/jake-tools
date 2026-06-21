from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from jake_tools.daily_report.models import (
    DailyReportLaneOptions,
    LaneName,
    LaneShape,
    LaneSpec,
    ModelTier,
    SafetyMode,
)
from jake_tools.daily_report.paths import DailyReportPaths
from jake_tools.daily_report.prompts import build_prompt


_REQUIRED_SECTIONS: dict[LaneName, tuple[str, ...]] = {
    LaneName.SESSION_HINDSIGHT: ("summary", "decisions", "risks", "open threads"),
    LaneName.MEMORY_CANDIDATES: ("candidates", "rejects", "rationale"),
    LaneName.SKILL_REVIEW: ("skill updates", "missing skills", "no-op notes"),
    LaneName.FAILURE_PATTERNS: ("patterns", "impact", "mitigations"),
    LaneName.TRANSCRIPTS_AND_DUMC: ("transcripts", "DUM-C", "follow-ups"),
    LaneName.INBOX_TRIAGE: ("urgent", "follow-ups", "waiting", "caveats"),
}

_SHAPES: dict[LaneName, LaneShape] = {
    LaneName.SESSION_HINDSIGHT: LaneShape.WORKER_AGENT,
    LaneName.MEMORY_CANDIDATES: LaneShape.PRE_FED,
    LaneName.SKILL_REVIEW: LaneShape.PRE_FED,
    LaneName.FAILURE_PATTERNS: LaneShape.PRE_FED,
    LaneName.TRANSCRIPTS_AND_DUMC: LaneShape.WORKER_AGENT,
    LaneName.INBOX_TRIAGE: LaneShape.PRE_FED,
}

_TOOLSETS: dict[LaneName, tuple[str, ...]] = {
    LaneName.SESSION_HINDSIGHT: ("session_search",),
    LaneName.MEMORY_CANDIDATES: (),
    LaneName.SKILL_REVIEW: (),
    LaneName.FAILURE_PATTERNS: (),
    LaneName.TRANSCRIPTS_AND_DUMC: ("session_search", "file"),
    LaneName.INBOX_TRIAGE: (),
}

_SAFETY_MODES: dict[LaneName, SafetyMode] = {
    LaneName.SESSION_HINDSIGHT: "read-only",
    LaneName.MEMORY_CANDIDATES: "tool-less",
    LaneName.SKILL_REVIEW: "tool-less",
    LaneName.FAILURE_PATTERNS: "tool-less",
    LaneName.TRANSCRIPTS_AND_DUMC: "read-only",
    LaneName.INBOX_TRIAGE: "envelope-only",
}

_MODEL_TIERS: dict[LaneName, ModelTier] = {
    LaneName.SESSION_HINDSIGHT: "standard",
    LaneName.MEMORY_CANDIDATES: "cheap",
    LaneName.SKILL_REVIEW: "cheap",
    LaneName.FAILURE_PATTERNS: "cheap",
    LaneName.TRANSCRIPTS_AND_DUMC: "standard",
    LaneName.INBOX_TRIAGE: "cheap",
}

_LANE_ORDER: tuple[LaneName, ...] = (
    LaneName.SESSION_HINDSIGHT,
    LaneName.MEMORY_CANDIDATES,
    LaneName.SKILL_REVIEW,
    LaneName.FAILURE_PATTERNS,
    LaneName.TRANSCRIPTS_AND_DUMC,
    LaneName.INBOX_TRIAGE,
)


def build_lane_specs(
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
    preflight: Any | None = None,
) -> list[LaneSpec]:
    del preflight  # reserved for milestone 7 wiring; specs stay pure for now.
    return [_build_lane_spec(name, options, paths) for name in _LANE_ORDER]


def _build_lane_spec(
    name: LaneName,
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
) -> LaneSpec:
    evidence_bundle_path = _evidence_bundle_path(paths, name)
    artefact_path = _artefact_path(paths, name)
    model_tier = _MODEL_TIERS[name]
    lane_options = replace(options, model_tier=model_tier)
    model = _model_for_tier(lane_options)
    required_sections = _REQUIRED_SECTIONS[name]
    return LaneSpec(
        name=name,
        shape=_SHAPES[name],
        provider=lane_options.provider,
        model=model,
        model_tier=lane_options.model_tier,
        enabled_toolsets=_TOOLSETS[name],
        prompt=build_prompt(
            name=name,
            run_id=lane_options.run_id,
            target_date=lane_options.target_date,
            timezone_name=lane_options.timezone_name,
            evidence_bundle_path=str(evidence_bundle_path),
            required_sections=required_sections,
        ),
        evidence_bundle_path=evidence_bundle_path,
        artefact_path=artefact_path,
        required_sections=required_sections,
        timeout_seconds=lane_options.timeout_seconds,
        safety_mode=_SAFETY_MODES[name],
    )


def _evidence_bundle_path(paths: DailyReportPaths, name: LaneName) -> Path:
    return paths.evidence / f"{name.value}.json"


def _artefact_path(paths: DailyReportPaths, name: LaneName) -> Path:
    return paths.subtasks / f"{name.value}.json"


def _model_for_tier(options: DailyReportLaneOptions) -> str:
    if options.model_tier == "cheap":
        return options.extra_context.get("evidence_model", options.model)
    return options.model
