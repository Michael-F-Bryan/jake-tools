from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from jake_tools.daily_report.models import LaneName, LaneOutput
from jake_tools.prompting import StructuredPrompt


class DailyReportLanePrompt(StructuredPrompt[LaneOutput]):
    response_model: ClassVar[type[BaseModel]] = LaneOutput
    template: ClassVar[str] = """
Run ID: {{ run_id }}
Target date: {{ target_date }}
Timezone: {{ timezone_name }}
Lane: {{ lane_name }}
Evidence bundle: {{ evidence_bundle_path }}
Required sections: {{ required_sections }}
Safety rules: {{ safety_rules }}
Task: {{ task }}

Return concise Markdown plus structured findings, actions, caveats, evidence paths, and cited session IDs.
Use only evidence from the declared bundle or the explicitly enabled lane tools.
Every cited_session_id must appear in the evidence. Do not invent session IDs.
Agents do not write files; the runner writes artefacts from your structured response.
"""

    run_id: str
    target_date: str
    timezone_name: str
    lane_name: str
    evidence_bundle_path: str
    required_sections: str
    safety_rules: str
    task: str


class SessionHindsightPrompt(DailyReportLanePrompt):
    pass


class MemoryCandidatesPrompt(DailyReportLanePrompt):
    pass


class SkillReviewPrompt(DailyReportLanePrompt):
    pass


class FailurePatternsPrompt(DailyReportLanePrompt):
    pass


class TranscriptsAndDumcPrompt(DailyReportLanePrompt):
    pass


class InboxTriagePrompt(DailyReportLanePrompt):
    pass


def build_prompt(
    *,
    name: LaneName,
    run_id: str,
    target_date: str,
    timezone_name: str,
    evidence_bundle_path: str,
    required_sections: tuple[str, ...],
) -> DailyReportLanePrompt:
    prompt_type = _PROMPTS[name]
    return prompt_type(
        run_id=run_id,
        target_date=target_date,
        timezone_name=timezone_name,
        lane_name=name.value,
        evidence_bundle_path=evidence_bundle_path,
        required_sections=", ".join(required_sections),
        safety_rules=_SAFETY_RULES[name],
        task=_TASKS[name],
    )


_SAFETY_RULES: dict[LaneName, str] = {
    LaneName.SESSION_HINDSIGHT: "Read-only worker lane. Use only scoped session-search evidence. Do not write files. Cite only session IDs present in evidence.",
    LaneName.MEMORY_CANDIDATES: "Tool-less pre-fed judgement lane. Do not write files. Propose memory candidates only when supported by supplied evidence.",
    LaneName.SKILL_REVIEW: "Tool-less pre-fed judgement lane. Do not write files. Review only supplied skill/catalogue evidence.",
    LaneName.FAILURE_PATTERNS: "Tool-less pre-fed judgement lane. Do not write files. Identify repeated failures only from supplied evidence.",
    LaneName.TRANSCRIPTS_AND_DUMC: "Read-only worker lane. Use only scoped file/session evidence for transcripts and DUM-C material. Do not write files. Cite only evidence-backed session IDs.",
    LaneName.INBOX_TRIAGE: "Envelope-only inbox lane. Use supplied Himalaya envelope exports only. Do not read message bodies, draft replies, send mail, move mail, delete mail, or write files.",
}

_TASKS: dict[LaneName, str] = {
    LaneName.SESSION_HINDSIGHT: "Summarise the day's useful session hindsight: decisions, surprises, unfinished threads, and risks worth carrying forward.",
    LaneName.MEMORY_CANDIDATES: "Extract durable memory candidates and reject stale or one-off facts. Keep suggestions minimal and evidence-backed.",
    LaneName.SKILL_REVIEW: "Identify skill maintenance opportunities from the supplied catalogue and session evidence.",
    LaneName.FAILURE_PATTERNS: "Find repeated tool, workflow, or judgement failures and suggest concrete mitigations.",
    LaneName.TRANSCRIPTS_AND_DUMC: "Review transcript and DUM-C evidence for operational follow-ups, missing context, and report-worthy notes.",
    LaneName.INBOX_TRIAGE: "Triage envelope metadata into high-signal follow-ups and caveats without drafting responses or inspecting bodies.",
}

_PROMPTS: dict[LaneName, type[DailyReportLanePrompt]] = {
    LaneName.SESSION_HINDSIGHT: SessionHindsightPrompt,
    LaneName.MEMORY_CANDIDATES: MemoryCandidatesPrompt,
    LaneName.SKILL_REVIEW: SkillReviewPrompt,
    LaneName.FAILURE_PATTERNS: FailurePatternsPrompt,
    LaneName.TRANSCRIPTS_AND_DUMC: TranscriptsAndDumcPrompt,
    LaneName.INBOX_TRIAGE: InboxTriagePrompt,
}
