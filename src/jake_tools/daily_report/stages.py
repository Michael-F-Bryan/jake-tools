from __future__ import annotations

from typing import Protocol

from .models import DailyReportLaneOptions, LaneOutput, LaneShape, LaneSpec
from ..hermes import AgentSpec, Hermes, Reply


class DailyReportStages(Protocol):
    def run_lane(
        self,
        spec: LaneSpec,
        options: DailyReportLaneOptions,
    ) -> tuple[LaneOutput, Reply]: ...


class HermesDailyReportStages:
    def __init__(self, hermes: Hermes) -> None:
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
