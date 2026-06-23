from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from ..prompting import StructuredPrompt
from .models import LaneName, LaneOutput


class DailyReportLanePrompt(StructuredPrompt[LaneOutput]):
    response_model: ClassVar[type[BaseModel]] = LaneOutput
    template: ClassVar[str] = """\
Run ID: {{ run_id }}
Target date: {{ target_date }}
Timezone: {{ timezone_name }}
Lane: {{ lane_name }}
Evidence bundle: {{ evidence_bundle_path }}
Required sections: {{ required_sections }}
Required Markdown headings:
{{ required_section_headings }}
Safety rules: {{ safety_rules }}
Task: {{ task }}
{{ inline_evidence }}

Return concise Markdown plus structured findings, actions, caveats, evidence paths, and cited session IDs.
The markdown field must include every required section above as an exact Markdown heading, one per line. Use the shown heading text exactly, including case and punctuation. Do not replace headings with prose, bullets, or synonyms.
Use only evidence from the declared bundle or the explicitly enabled lane tools.
Set evidence_paths only to existing filesystem evidence files, normally the declared evidence bundle path. Do not put tool calls, failed searches, notes, or prose in evidence_paths; mention those in findings or caveats instead.
Every cited_session_id must appear in the evidence. Do not invent session IDs.
When evidence is unavailable, say that directly, for example: "Evidence unavailable from declared bundle; no body content reviewed." Do not use TODO, TBD, lorem ipsum, or placeholder-style filler.
Agents do not write files; the runner writes artefacts from your structured response.
You MUST return a valid JSON object matching the LaneOutput schema. The markdown field is required. Never return a status, progress, or non-LaneOutput object — the runner can only interpret a valid LaneOutput. If no evidence is available, return markdown with honest "Evidence unavailable" text under each required heading and empty findings/actions/caveats. Never say "reading evidence bundle" or report progress in the output.
"""

    run_id: str
    target_date: str
    timezone_name: str
    lane_name: str
    evidence_bundle_path: str
    required_sections: str
    required_section_headings: str
    safety_rules: str
    task: str
    inline_evidence: str = ""


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

    inline_evidence = ""
    if name is LaneName.INBOX_TRIAGE:
        evidence_root = Path(evidence_bundle_path).parent
        inbox_env_path = evidence_root / "inbox-envelopes.json"
        if inbox_env_path.exists():
            try:
                data = json.loads(inbox_env_path.read_text(encoding="utf-8"))
                inline_evidence = (
                    "\nInbox envelope metadata (inline evidence):\n"
                    + json.dumps(data, indent=2)
                )
            except OSError, json.JSONDecodeError:
                inline_evidence = "\nInbox envelope metadata: (unreadable)"

    return prompt_type(
        run_id=run_id,
        target_date=target_date,
        timezone_name=timezone_name,
        lane_name=name.value,
        evidence_bundle_path=evidence_bundle_path,
        required_sections=", ".join(required_sections),
        required_section_headings="\n".join(
            f"## {section}" for section in required_sections
        ),
        safety_rules=_SAFETY_RULES[name],
        task=_TASKS[name],
        inline_evidence=inline_evidence,
    )


_SAFETY_RULES: dict[LaneName, str] = {
    LaneName.SESSION_HINDSIGHT: "Read-only worker lane. Use only scoped session-search evidence. Do not write files. Cite only session IDs present in evidence.",
    LaneName.MEMORY_CANDIDATES: "Tool-less pre-fed judgement lane. Do not write files. Propose memory candidates only when supported by supplied evidence. Every cited_session_id must be a Hermes session ID from the session manifest. Do not invent session IDs.",
    LaneName.SKILL_REVIEW: "Tool-less pre-fed judgement lane. Do not write files. Review only supplied skill/catalogue evidence.",
    LaneName.FAILURE_PATTERNS: "Tool-less pre-fed judgement lane. Do not write files. Identify repeated failures only from supplied evidence.",
    LaneName.TRANSCRIPTS_AND_DUMC: "Read-only worker lane. Use only scoped file/session evidence for transcripts and DUM-C material. Do not write files. Cite only evidence-backed session IDs.",
    LaneName.INBOX_TRIAGE: "Envelope-only inbox lane. Use supplied Himalaya envelope exports only. Do not read message bodies, draft replies, send mail, move mail, delete mail, or write files. cited_session_ids must be empty for this envelope-only lane; email message IDs are not Hermes session IDs.",
}

_TASKS: dict[LaneName, str] = {
    LaneName.SESSION_HINDSIGHT: "Summarise the day's useful session hindsight: decisions, surprises, unfinished threads, and risks worth carrying forward.",
    LaneName.MEMORY_CANDIDATES: "Extract durable memory candidates and reject stale or one-off facts. Keep suggestions minimal and evidence-backed.",
    LaneName.SKILL_REVIEW: "Identify skill maintenance opportunities from the supplied catalogue and session evidence.",
    LaneName.FAILURE_PATTERNS: "Find repeated tool, workflow, or judgement failures and suggest concrete mitigations. The markdown field is required with all required headings.",
    LaneName.TRANSCRIPTS_AND_DUMC: "Review transcript and DUM-C evidence for operational follow-ups, missing context, and report-worthy notes.",
    LaneName.INBOX_TRIAGE: "Triage envelope metadata into high-signal follow-ups and caveats without drafting responses or inspecting bodies. Do not put email message IDs in cited_session_ids.",
}

_PROMPTS: dict[LaneName, type[DailyReportLanePrompt]] = {
    LaneName.SESSION_HINDSIGHT: SessionHindsightPrompt,
    LaneName.MEMORY_CANDIDATES: MemoryCandidatesPrompt,
    LaneName.SKILL_REVIEW: SkillReviewPrompt,
    LaneName.FAILURE_PATTERNS: FailurePatternsPrompt,
    LaneName.TRANSCRIPTS_AND_DUMC: TranscriptsAndDumcPrompt,
    LaneName.INBOX_TRIAGE: InboxTriagePrompt,
}
