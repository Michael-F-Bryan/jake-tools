from __future__ import annotations

from typing import Protocol

from ..hermes import AgentSpec, Reply
from ..prompting import StructuredPrompt
from .models import DailyReportLaneOptions, LaneOutput, LaneShape, LaneSpec


class DailyReportStages(Protocol):
    def run_lane(
        self,
        spec: LaneSpec,
        options: DailyReportLaneOptions,
    ) -> tuple[LaneOutput, Reply]: ...


class LaneStructuredRunner(Protocol):
    """The single Hermes capability lane stages depend on: one structured call per lane."""

    def run_structured(
        self,
        prompt: StructuredPrompt[LaneOutput],
        spec: AgentSpec | None = None,
    ) -> tuple[LaneOutput, Reply]: ...


class HermesDailyReportStages:
    def __init__(self, hermes: LaneStructuredRunner) -> None:
        self.hermes = hermes

    def run_lane(
        self,
        spec: LaneSpec,
        options: DailyReportLaneOptions,
    ) -> tuple[LaneOutput, Reply]:
        base_spec = AgentSpec(
            model=spec.model,
            provider=spec.provider,
            parent_session_id=options.parent_session_id,
            session_db=options.session_db,
            max_iterations=options.max_iterations,
        )
        if spec.shape is LaneShape.WORKER_AGENT:
            agent_spec = base_spec.model_copy(
                update={"enabled_toolsets": list(spec.enabled_toolsets)}
            )
        else:
            agent_spec = base_spec
        return self.hermes.run_structured(spec.prompt, agent_spec)
