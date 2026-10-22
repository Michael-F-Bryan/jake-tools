"""The bounded, tool-free handoff a worker asks for after a ``max_turns`` stop.

Execution has already stopped when this runs: the task's conversation is
closed, its inbox is closed and its processes are swept. The worker then
resumes the same Claude session once more through :class:`ClaudeAgent`
with nothing to act with:

- ``tools=()`` (the session exposes no built-in tool), no permission rules,
  no local tools and so no in-process MCP server (``strict_mcp_config``
  keeps every other server out);
- ``max_turns=1``, and ``CLAUDE_CODE_MAX_OUTPUT_TOKENS`` caps the reply;
- a wall-clock bound of :data:`HANDOFF_TIMEOUT_SECONDS`, never past the
  task's own deadline, and ``max_budget_usd`` set to what the task has left.

The ``init`` message must report the same session and no tools, and any
tool call ends the handoff as ``failed``; neither is expected, since the
session has no tools to call. Observed against CLI 2.1.293: a session cut
off mid tool call resumes like this, keeps its earlier context and exposes
an empty tool list (``test_live_max_turns_handoff_*``).

It is never attempted after a cancel, a timeout or a budget stop, and it
is ``skipped`` (no model call) when less than :data:`HANDOFF_MIN_SECONDS`
or :data:`HANDOFF_MIN_BUDGET_USD` is left, or when what was left cannot be
established: a stop the worker enforces mid-prompt comes before Claude
reports that prompt's cost, and unknown spend is not zero spend. The CLI
checks the budget only after a reply, so a single handoff reply can
overrun what was left by its own cost; the floor and the output cap keep
that small.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import anyio

from ..ai_usage import Usage
from ..claude import AgentSpec, ClaudeAgent, describe
from .models import Handoff
from .rundir import utc_now, write_handoff_text

log = logging.getLogger(__name__)

HANDOFF_TIMEOUT_SECONDS = 120.0
HANDOFF_MIN_SECONDS = 15.0
HANDOFF_MIN_BUDGET_USD = 0.02
HANDOFF_MAX_OUTPUT_TOKENS = 4096
HANDOFF_TEXT_CHARS = 12_000

HANDOFF_PROMPT = """\
Execution of this task has stopped: it reached its max_turns limit. You \
have no tools in this reply, and you must not try to continue, fix or retry \
anything. Write a handoff for whoever picks the task up, from what you know \
in this conversation, in at most about 400 words, with these headings:

Completed: what is done, with the durable file paths and commit IDs.
Checks: each check you actually ran and its result; checks still outstanding.
Incomplete or blocked: what is unfinished and why.
Uncertain state: commands that may still have been running when you were \
stopped, and partial edits or other mutations that may be half-done.
Remaining delta: the smallest concrete set of next steps to finish.

Write "unknown" where you do not know. Do not guess, and do not claim \
anything was verified unless you saw it verified."""


async def write_handoff(
    agent: ClaudeAgent,
    run_dir: Path,
    *,
    session_id: str,
    session_cost_usd: float | None,
    seconds_left: float,
    budget_left_usd: float | None,
) -> HandoffCall:
    """Ask for the handoff: the record, the usage it added, the session total.

    ``budget_left_usd`` is ``None`` when what the task has spent is not
    known (execution stopped mid-prompt, before Claude reported that
    prompt's cost); the handoff is then skipped, not run against an
    allowance that cannot be established.

    ``agent`` carries the run's telemetry and transcript observers and its
    default spec, whose tools, rules, local tools and turn limit are all
    overridden here. Never raises for a failed call; a cancellation of the
    caller's scope propagates (the caller records it).
    """
    if seconds_left < HANDOFF_MIN_SECONDS:
        return _skipped(f"only {seconds_left:.0f}s of the task's timeout was left")
    if budget_left_usd is None:
        return _skipped(
            "the remaining budget is unknown: execution stopped mid-prompt, "
            "before Claude reported what that prompt cost"
        )
    if budget_left_usd < HANDOFF_MIN_BUDGET_USD:
        return _skipped(f"only ${budget_left_usd:.4f} of the task's budget was left")
    spec = AgentSpec(
        tools=(),
        allowed_tools=(),
        local_tools=(),
        max_turns=1,
        max_budget_usd=budget_left_usd,
        session_id=None,
        resume=session_id,
        env={"CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(HANDOFF_MAX_OUTPUT_TOKENS)},
    )
    bound = min(HANDOFF_TIMEOUT_SECONDS, seconds_left)
    started_at = utc_now()
    texts: list[str] = []
    usage: Usage | None = None
    session_total: float | None = session_cost_usd
    error: str | None = None
    result_text: str | None = None
    finished = False
    with anyio.move_on_after(bound):
        try:
            async with agent.converse(
                spec, cost_baseline_usd=session_cost_usd
            ) as conversation:
                await conversation.send(HANDOFF_PROMPT, uuid=str(uuid.uuid4()))
                async for message in conversation.messages():
                    view = describe(message)
                    if view.kind == "init":
                        if view.session_id and view.session_id != session_id:
                            error = f"Claude resumed session {view.session_id}, not {session_id}"
                            break
                        if view.tools:
                            error = f"the handoff session exposed tools: {', '.join(view.tools)}"
                            break
                        continue
                    if view.tool_uses:
                        error = "the agent attempted a tool call during the handoff"
                        break
                    if view.kind == "assistant":
                        texts.extend(view.texts)
                    if view.result is not None:
                        usage = conversation.usage
                        result_text = view.result.text
                        if view.result.is_error:
                            error = (
                                f"the handoff reply failed "
                                f"(subtype={view.result.subtype!r})"
                            )
                        finished = True
                        break
                usage = conversation.usage if usage is None else usage
                session_total = conversation.session_cost_usd
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}".strip()[:2000]
    text = result_text or ("\n".join(texts) if texts else None)
    if not finished and error is None:
        error = f"the handoff did not finish within {bound:.0f}s"
    cost = usage.estimated_cost_usd if usage is not None else None
    path = write_handoff_text(run_dir, text) if text else None
    handoff = Handoff(
        status="written" if error is None and text else "failed",
        started_at=started_at,
        finished_at=utc_now(),
        text=text[:HANDOFF_TEXT_CHARS] if text else None,
        text_chars=len(text) if text else None,
        text_path=path,
        error=error if error or text else "the handoff reply was empty",
        cost_usd=cost,
    )
    log.info("handoff %s: %s", handoff.status, handoff.error or "ok")
    return HandoffCall(handoff, usage, session_total)


@dataclass(frozen=True)
class HandoffCall:
    """What :func:`write_handoff` left: the record, the usage the call
    added (``None`` if it made none or none was reported), and the
    session's running cost after it (for the next resume's baseline)."""

    handoff: Handoff
    usage: Usage | None = None
    session_cost_usd: float | None = None


def _skipped(reason: str) -> HandoffCall:
    return HandoffCall(
        Handoff(status="skipped", error=f"no handoff was attempted: {reason}")
    )
