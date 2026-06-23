from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import (
    DailyReportLaneOptions,
    LaneName,
    LaneShape,
    LaneSpec,
    ModelTier,
    SafetyMode,
)
from .paths import DailyReportPaths
from .prompts import build_prompt


@dataclass(frozen=True)
class LaneDefinition:
    name: LaneName
    shape: LaneShape
    enabled_toolsets: tuple[str, ...]
    safety_mode: SafetyMode
    model_tier: ModelTier
    required_sections: tuple[str, ...]


LANE_DEFINITIONS: tuple[LaneDefinition, ...] = (
    LaneDefinition(
        name=LaneName.SESSION_HINDSIGHT,
        shape=LaneShape.WORKER_AGENT,
        enabled_toolsets=("session_search",),
        safety_mode="read-only",
        model_tier="standard",
        required_sections=("summary", "decisions", "risks", "open threads"),
    ),
    LaneDefinition(
        name=LaneName.MEMORY_CANDIDATES,
        shape=LaneShape.PRE_FED,
        enabled_toolsets=(),
        safety_mode="tool-less",
        model_tier="cheap",
        required_sections=("candidates", "rejects", "rationale"),
    ),
    LaneDefinition(
        name=LaneName.SKILL_REVIEW,
        shape=LaneShape.PRE_FED,
        enabled_toolsets=(),
        safety_mode="tool-less",
        model_tier="cheap",
        required_sections=("skill updates", "missing skills", "no-op notes"),
    ),
    LaneDefinition(
        name=LaneName.FAILURE_PATTERNS,
        shape=LaneShape.PRE_FED,
        enabled_toolsets=(),
        safety_mode="tool-less",
        model_tier="cheap",
        required_sections=("patterns", "impact", "mitigations"),
    ),
    LaneDefinition(
        name=LaneName.TRANSCRIPTS_AND_DUMC,
        shape=LaneShape.WORKER_AGENT,
        enabled_toolsets=("session_search", "file"),
        safety_mode="read-only",
        model_tier="standard",
        required_sections=("transcripts", "DUM-C", "follow-ups"),
    ),
    LaneDefinition(
        name=LaneName.INBOX_TRIAGE,
        shape=LaneShape.PRE_FED,
        enabled_toolsets=(),
        safety_mode="envelope-only",
        model_tier="cheap",
        required_sections=("urgent", "follow-ups", "waiting", "caveats"),
    ),
)


def build_lane_specs(
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
) -> list[LaneSpec]:
    return [_build_lane_spec(lane, options, paths) for lane in LANE_DEFINITIONS]


def _build_lane_spec(
    lane: LaneDefinition,
    options: DailyReportLaneOptions,
    paths: DailyReportPaths,
) -> LaneSpec:
    evidence_bundle_path = _evidence_bundle_path(paths, lane.name)
    artefact_path = _artefact_path(paths, lane.name)
    lane_options = options.model_copy(update={"model_tier": lane.model_tier})
    model = _model_for_tier(lane_options)
    return LaneSpec(
        name=lane.name,
        shape=lane.shape,
        provider=lane_options.provider,
        model=model,
        model_tier=lane_options.model_tier,
        enabled_toolsets=lane.enabled_toolsets,
        prompt=build_prompt(
            name=lane.name,
            run_id=lane_options.run_id,
            target_date=lane_options.target_date,
            timezone_name=lane_options.timezone_name,
            evidence_bundle_path=str(evidence_bundle_path),
            required_sections=lane.required_sections,
        ),
        evidence_bundle_path=evidence_bundle_path,
        artefact_path=artefact_path,
        required_sections=lane.required_sections,
        timeout_seconds=lane_options.timeout_seconds,
        safety_mode=lane.safety_mode,
    )


def _evidence_bundle_path(paths: DailyReportPaths, name: LaneName) -> Path:
    return paths.evidence / f"{name.value}.json"


def _artefact_path(paths: DailyReportPaths, name: LaneName) -> Path:
    return paths.subtasks / f"{name.value}.json"


def _model_for_tier(options: DailyReportLaneOptions) -> str:
    if options.model_tier == "cheap":
        return options.extra_context.get("evidence_model", options.model)
    return options.model
