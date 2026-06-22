from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..hermes import HermesResult
from .coordinator import LaneRunResult
from .models import LaneName, LaneOutput, LaneSpec
from .validation import LaneValidationResult


class ManifestLaneSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    shape: str
    provider: str
    model: str
    model_tier: str
    enabled_toolsets: list[str]
    evidence_bundle_path: str
    artefact_path: str
    required_sections: list[str]
    timeout_seconds: int
    safety_mode: str


class ManifestLaneResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    status: Literal["ok", "fail"]
    artefact_path: str
    error: str | None = None
    output: LaneOutput | None = None
    hermes_result: HermesResult | None = None


class ManifestLaneValidation(BaseModel):
    model_config = ConfigDict(frozen=True)

    lane: str
    status: str
    errors: list[str] = Field(default_factory=list)


class DailyReportManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    date: str
    parent_session_id: str | None
    paths: dict[str, str]
    lane_specs: list[ManifestLaneSpec]
    lane_results: list[ManifestLaneResult]
    validation_results: list[ManifestLaneValidation]
    status: str


def manifest_lane_spec(spec: LaneSpec) -> ManifestLaneSpec:
    return ManifestLaneSpec(
        name=spec.name.value,
        shape=spec.shape.value,
        provider=spec.provider,
        model=spec.model,
        model_tier=spec.model_tier,
        enabled_toolsets=list(spec.enabled_toolsets),
        evidence_bundle_path=str(spec.evidence_bundle_path),
        artefact_path=str(spec.artefact_path),
        required_sections=list(spec.required_sections),
        timeout_seconds=spec.timeout_seconds,
        safety_mode=spec.safety_mode,
    )


def manifest_lane_result(result: LaneRunResult) -> ManifestLaneResult:
    return ManifestLaneResult(
        name=result.name.value,
        status=result.status,
        artefact_path=str(result.artefact_path),
        error=result.error,
        output=result.output,
        hermes_result=result.hermes_result,
    )


def manifest_lane_validation(validation: LaneValidationResult) -> ManifestLaneValidation:
    return ManifestLaneValidation(
        lane=validation.lane.value,
        status=validation.status,
        errors=list(validation.errors),
    )
