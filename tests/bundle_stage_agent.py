"""A prompt-aware fake for the bundle's LLM stages.

``agent_fakes.fake_agent`` replays a fixed script, which is right when a
test knows exactly what it asked for. The Phase 3-B/3-C stages cannot use
it directly: their payloads have to name turn, cluster, and participant
IDs that are minted at runtime, so a fixed script would either be
impossible to write or would name IDs the validators correctly reject.

This fake reads those IDs back out of the rendered prompt -- which is
itself a real check, since a stage that failed to include the IDs it
expects a model to cite would produce a fake with nothing to cite -- and
returns a well-formed payload for whichever stage is asking. Everything
downstream of the model boundary (validation, ledgers, gates, capability
proofs) stays completely real.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage

from jake_tools.claude import AgentSpec, ClaudeAgent

_TURN_ID_RE = re.compile(r"turn_[0-9a-f-]{36}")
_CLUSTER_ID_RE = re.compile(r"cluster_[0-9a-f-]{36}")
_PARTICIPANT_ID_RE = re.compile(r"participant_[0-9a-f-]{36}")
_SECTION_ID_RE = re.compile(r"seg_[0-9a-f-]{36}")


def _unique(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(pattern.findall(text)))


@dataclass
class StagePlan:
    """How this fake should answer each stage.

    Every field is a hook a test can override to drive one specific
    behaviour (an unfaithful polish, an invented turn ID, a proposal that
    names the wrong person) without hand-writing the other three payloads.
    """

    #: cluster index -> participant index, or None for "no candidate".
    proposals: dict[int, int | None] = field(default_factory=dict)
    #: Rewrites applied to each turn's text during correct/polish.
    correct_text: Callable[[str], str] = lambda text: text
    polish_text: Callable[[str], str] = lambda text: text
    #: Indices into the transcript at which a chapter starts.
    chapter_starts: tuple[int, ...] = (0,)
    minutes_summary: str = "The meeting covered the recorded discussion."
    finding_text: str | None = "Agreed to proceed."
    override: Callable[[str], dict[str, Any] | None] | None = None


def _proposal_payload(prompt: str, plan: StagePlan) -> dict[str, Any]:
    clusters = _unique(_CLUSTER_ID_RE, prompt)
    participants = _unique(_PARTICIPANT_ID_RE, prompt)
    turns = _unique(_TURN_ID_RE, prompt)
    proposals = []
    for index, cluster_id in enumerate(clusters):
        target = plan.proposals.get(index)
        participant_id = (
            participants[target]
            if target is not None and target < len(participants)
            else None
        )
        proposals.append(
            {
                "cluster_id": cluster_id,
                "participant_id": participant_id,
                "confidence": 0.7 if participant_id else 0.0,
                "evidence_turn_ids": turns[:1] if participant_id else [],
                "rationale": "fixture proposal from conversational context",
            }
        )
    return {"proposals": proposals}


def _turns_from_prompt(prompt: str) -> list[dict[str, Any]]:
    """The stage's own ``turns`` JSON block, parsed back out.

    Found by scanning for the first JSON array whose entries carry a
    ``turn_id`` -- the prompts render it with ``| json``, so this reads
    exactly what the stage sent rather than a re-derived guess.
    """
    for match in re.finditer(r"^\[$", prompt, re.MULTILINE):
        start = match.start()
        depth = 0
        for index in range(start, len(prompt)):
            if prompt[index] == "[":
                depth += 1
            elif prompt[index] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(prompt[start : index + 1])
                    except json.JSONDecodeError:
                        break
                    if (
                        isinstance(parsed, list)
                        and parsed
                        and isinstance(parsed[0], dict)
                        and ("turn_id" in parsed[0] or "index" in parsed[0])
                    ):
                        return parsed
                    break
    return []


def _text_payload(prompt: str, transform: Callable[[str], str]) -> dict[str, Any]:
    """The text stages address turns by window position -- so does this."""
    return {
        "turns": [
            {
                "index": turn["index"],
                "text": transform(str(turn.get("text", ""))),
                "removal_reasons": [],
            }
            for turn in _turns_from_prompt(prompt)
        ]
    }


def _chapter_payload(prompt: str, plan: StagePlan) -> dict[str, Any]:
    """The chapter stage sees ordinals, not turn IDs -- so does this."""
    turn_count = len(_turns_from_prompt(prompt))
    return {
        "chapters": [
            {
                "title": f"Chapter {position + 1}",
                "summary": "What this stretch of the conversation covered.",
                "first_turn_index": index,
            }
            for position, index in enumerate(plan.chapter_starts)
            if index < turn_count
        ]
    }


def _minutes_payload(prompt: str, plan: StagePlan) -> dict[str, Any]:
    turns = _unique(_TURN_ID_RE, prompt)
    sections = _unique(_SECTION_ID_RE, prompt)
    findings = []
    if plan.finding_text is not None and turns:
        findings.append(
            {
                "kind": "decision",
                "text": plan.finding_text,
                "evidence_turn_ids": turns[:1],
                "evidence_section_ids": [],
                "owner_participant_id": None,
                "due": "",
            }
        )
    return {
        "summary": {
            "text": plan.minutes_summary,
            "evidence_turn_ids": turns[:1],
            "evidence_section_ids": sections[:1] if sections else [],
        },
        "findings": findings,
    }


def payload_for(prompt: str, plan: StagePlan) -> dict[str, Any]:
    """Which stage is asking, and what a well-formed answer looks like."""
    if plan.override is not None:
        overridden = plan.override(prompt)
        if overridden is not None:
            return overridden
    if "one proposal per cluster" in prompt:
        return _proposal_payload(prompt, plan)
    if "Correct mis-transcriptions" in prompt:
        return _text_payload(prompt, plan.correct_text)
    if "Polish these transcript turns" in prompt:
        return _text_payload(prompt, plan.polish_text)
    if "Divide this meeting transcript into chapters" in prompt:
        return _chapter_payload(prompt, plan)
    if "Write terse minutes" in prompt:
        return _minutes_payload(prompt, plan)
    raise AssertionError(f"no fixture payload for this prompt:\n{prompt[:400]}")


@dataclass
class StageQuery:
    plan: StagePlan
    prompts: list[str] = field(default_factory=list)

    def __call__(
        self, *, prompt: str, options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        self.prompts.append(prompt)
        payload = payload_for(prompt, self.plan)

        async def stream() -> AsyncIterator[Message]:
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="fixture-session",
                total_cost_usd=None,
                usage={"input_tokens": 0, "output_tokens": 0},
                result=None,
                structured_output=payload,
            )

        return stream()


def stage_agent(plan: StagePlan | None = None) -> ClaudeAgent:
    """A ``ClaudeAgent`` that answers every bundle stage from the prompt."""
    return ClaudeAgent(
        defaults=AgentSpec(model="fixture-model"),
        run_query=StageQuery(plan or StagePlan()),
    )


def stage_query_of(agent: ClaudeAgent) -> StageQuery:
    assert isinstance(agent.run_query, StageQuery)
    return agent.run_query
