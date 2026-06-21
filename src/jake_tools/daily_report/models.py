from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from jake_tools.prompting import StructuredPrompt


class LaneShape(StrEnum):
    PRE_FED = "pre-fed"
    WORKER_AGENT = "worker-agent"


class LaneName(StrEnum):
    SESSION_HINDSIGHT = "session-hindsight"
    MEMORY_CANDIDATES = "memory-candidates"
    SKILL_REVIEW = "skill-review"
    FAILURE_PATTERNS = "failure-patterns"
    TRANSCRIPTS_AND_DUMC = "transcripts-and-dumc"
    INBOX_TRIAGE = "inbox-triage"


ModelTier = Literal["cheap", "standard", "strong"]
SafetyMode = Literal["read-only", "envelope-only", "tool-less"]


class LaneOutput(BaseModel):
    markdown: str
    findings: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    evidence_paths: list[str] = Field(default_factory=list)
    cited_session_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class LaneSpec:
    name: LaneName
    shape: LaneShape
    provider: str
    model: str
    model_tier: ModelTier
    enabled_toolsets: tuple[str, ...]
    prompt: StructuredPrompt[LaneOutput]
    evidence_bundle_path: Path
    artefact_path: Path
    required_sections: tuple[str, ...]
    timeout_seconds: int
    safety_mode: SafetyMode

    def __post_init__(self) -> None:
        if self.shape is LaneShape.WORKER_AGENT and not self.enabled_toolsets:
            raise ValueError(f"{self.name.value} worker lane requires scoped toolsets")
        if self.shape is LaneShape.PRE_FED and self.name is LaneName.INBOX_TRIAGE and self.enabled_toolsets:
            raise ValueError("inbox-triage must be tool-less")
        blocked = {"default", "all", "*", "broad"}
        if any(toolset.strip().lower() in blocked for toolset in self.enabled_toolsets):
            raise ValueError(f"{self.name.value} declares a broad/default toolset")


@dataclass(frozen=True)
class DailyReportLaneOptions:
    run_id: str
    target_date: str
    timezone_name: str = "Australia/Perth"
    provider: str = "openrouter"
    model: str = "openrouter/auto"
    model_tier: ModelTier = "standard"
    timeout_seconds: int = 900
    parent_session_id: str | None = None
    session_db: Any | None = None
    max_iterations: int | None = None
    extra_context: dict[str, str] = field(default_factory=dict)
