"""The worker: the only place a delegated task actually runs.

``python -m jake_tools.mcp worker RUN_DIR`` calls :func:`main`. The worker
reads ``spec.json`` and ``brief.md`` and holds one streaming conversation
through :meth:`ClaudeAgent.converse`:

- The brief is the first user message. Every SDK message goes to
  ``transcript.jsonl`` (raw) and, through :func:`~.events.events_from_view`,
  to ``events.jsonl`` (normalised, no thinking). ``status.json`` carries the
  latest progress line, activity time and per-turn usage.
- Steering: a pump writes each ``queued`` message in ``messages/`` to the
  CLI as it arrives, tagged with the message's UUID, and records it
  ``delivered``. The CLI adds a message sent mid-turn to the conversation
  after the running tool call returns (it never interrupts it); a message
  sent after a turn ended starts a new one. Either way its echo, matched by
  UUID, marks it ``in_conversation``.
- The conversation ends at a result message once nothing is queued or
  still on its way (:meth:`~.inbox.Inbox.close_if_idle`); an error result
  (turn or budget limit) ends it at once.
- Clarification: the agent has exactly one extra tool, ``ask_parent``, an
  in-process :class:`~jake_tools.claude.LocalTool`. Calling it records a
  pending request in ``inputs/``, sets ``input_required``, and blocks until
  ``claude_update`` answers or declines it, then returns the answer as the
  tool result, so the same turn carries on. It is not a permission
  channel: it grants nothing, and no external MCP server is ever loaded.

Isolation: the agent gets exactly the granted built-ins as ``tools`` (an
empty grant is genuinely tool-less), those plus ``ask_parent`` as
``allowed_tools``, ``permission_mode="dontAsk"`` so anything else is denied
rather than waited on, ``setting_sources=()`` so no settings file is read,
and (via ``AgentSpec.to_options``) ``strict_mcp_config``.

Sessions: a new task's conversation is created with the session ID in
``spec.json``; a resumed task passes it as ``resume``. The first ``init``
message must report that ID; anything else, or a failure before any
``init``, ends the task as ``resume_failed`` (for a resume) rather than
carrying on in a different conversation.

Termination: ``timeout_seconds`` is a wall-clock limit on the whole
conversation, time spent waiting for an answer included, enforced by an
anyio ``move_on_after``; SIGTERM cancels the same scope. Both are anyio
cancellations, which is what the SDK's transport needs to run its
stdin-EOF / SIGTERM / SIGKILL escalation on the CLI child.

Processes: the CLI runs each tool shell in its own session, outside the
worker's process group, so the worker tracks every group in its descendant
tree into ``groups.json`` while it runs (and, on Linux, is a child subreaper
so orphans of a killed CLI stay in that tree). Before recording a terminal
state it closes the inbox (settling anything left open), kills its live
descendants and every recorded group, so "the task ended" also means "its
processes are gone". If the worker itself is killed first, the server does
the same when it settles the task (see :mod:`.control`).

Liveness: the worker holds ``.worker.lock`` for its whole life; the server
reads "alive" as "that lock is held", which an exited-but-unreaped worker
cannot fake.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio

from ..ai_usage import AICallTelemetry, Usage
from ..claude import (
    AgentSpec,
    ClaudeAgent,
    Conversation,
    LocalTool,
    LocalToolResult,
    Message,
    ResultView,
    describe,
    message_to_json,
)
from .events import Event, EventLog, events_from_view, progress_line
from .inbox import MAX_CHOICES, MAX_QUESTION_CHARS, Inbox
from .models import (
    InputRequest,
    RunResult,
    RunSpec,
    RunState,
    RunStatus,
    RunTelemetry,
    TerminationReason,
)
from .rundir import (
    TELEMETRY_FILE,
    TRANSCRIPT_FILE,
    acquire_worker_lock,
    open_private,
    read_brief,
    read_process_record,
    read_spec,
    update_state,
    utc_now,
    write_final_text,
    write_model,
    write_result,
)
from .spawn import (
    become_subreaper,
    record_process,
    sweep_own_group,
    track_descendant_groups,
)

log = logging.getLogger(__name__)

STAGE = "delegated"
MAX_ERROR_CHARS = 2000
PUMP_INTERVAL_SECONDS = 0.25
ANSWER_POLL_SECONDS = 0.25
PROGRESS_FLUSH_SECONDS = 1.0
MAX_CHOICE_CHARS = 200

ASK_TOOL = "ask_parent"
ASK_DESCRIPTION = (
    "Ask the person who delegated this task a question, and wait for the "
    "answer. Use it only when you cannot continue correctly without a "
    "decision or fact that only they have; otherwise proceed and state your "
    "assumptions in your final message. It does not grant permissions or "
    "tools. Optional `choices` lists suggested answers. Blocks until "
    "answered; the answer comes back as this tool's result."
)
ASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The question, self-contained."},
        "choices": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional suggested answers.",
        },
    },
    "required": ["question"],
}

_SUBTYPE_REASONS: dict[str, TerminationReason] = {
    "error_max_turns": "max_turns",
    "error_max_budget_usd": "budget",
}


@dataclass(frozen=True)
class Outcome:
    status: RunStatus
    reason: TerminationReason
    final_text: str | None = None
    usage: Usage | None = None
    error: str | None = None


def main(run_dir: Path) -> int:
    """Run the task in ``run_dir``; the process exit status (0 = completed).

    The worker lock is taken first and held (the fd is deliberately never
    closed) until this process exits; a task whose lock is already held has
    a worker, and this process leaves without touching anything.
    """
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if acquire_worker_lock(run_dir) is None:
        log.error("another worker already holds %s; exiting", run_dir)
        return 1
    if become_subreaper():
        log.info("child subreaper enabled")
    outcome = anyio.run(run_worker, run_dir)
    return 0 if outcome.status == "completed" else 1


async def run_worker(run_dir: Path) -> Outcome:
    spec = read_spec(run_dir)
    brief = read_brief(run_dir)
    pid, pgid = os.getpid(), os.getpgid(0)
    update_state(
        run_dir,
        lambda state: state.model_copy(
            update={
                "started_at": utc_now(),
                "pid": pid,
                "pgid": pgid,
                "session_id": spec.session_id,
                "resumed_from": spec.resume_task_id,
            }
        ),
    )
    if read_process_record(run_dir) is None:
        record_process(run_dir, pid)  # the server died before it could
    log.info("task %s started (pid %d, pgid %d)", spec.task_id, pid, pgid)

    events = EventLog(run_dir)
    inbox = Inbox(run_dir, events)
    events.append(
        "started",
        detail={"session_id": spec.session_id, "resumed_from": spec.resume_task_id},
    )
    transcript = TranscriptWriter(run_dir / TRANSCRIPT_FILE)
    telemetry = TelemetryFile(run_dir / TELEMETRY_FILE)
    progress = Progress(run_dir)
    ask = AskParent(run_dir, inbox, progress)
    agent = (
        ClaudeAgent(defaults=agent_spec(spec, local_tools=(ask.tool(),)))
        .with_telemetry(telemetry)
        .with_message_observer(transcript.write)
        .for_stage(STAGE)
    )
    driver = Driver(agent, spec, brief, inbox, events, progress)

    outcome: Outcome | None = None
    try:
        with anyio.CancelScope() as run_scope:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_cancel_on_signal, run_scope)
                tg.start_soon(track_descendant_groups, run_dir)
                outcome = await driver.run()
                tg.cancel_scope.cancel()
        if run_scope.cancel_called:
            outcome = driver.partial("cancelled", "cancelled")
        if outcome is None:  # pragma: no cover - defensive; the scope says why
            outcome = driver.partial(
                "failed", "sdk_error", error="worker ended without an outcome"
            )
    except BaseException as exc:
        outcome = driver.partial("failed", "sdk_error", error=_describe(exc))
        _finish(run_dir, spec, outcome, transcript, inbox, events)
        raise
    _finish(run_dir, spec, outcome, transcript, inbox, events)
    return outcome


def agent_spec(spec: RunSpec, *, local_tools: tuple[LocalTool, ...] = ()) -> AgentSpec:
    resuming = spec.resume_task_id is not None
    return AgentSpec(
        model=spec.model,
        tools=spec.tools,
        effort=spec.effort,
        max_turns=spec.max_turns,
        max_budget_usd=spec.max_budget_usd,
        cwd=spec.working_directory,
        setting_sources=(),
        permission_mode="dontAsk",
        local_tools=local_tools,
        session_id=None if resuming else spec.session_id,
        resume=spec.session_id if resuming else None,
        # The CLI's own per-call MCP timeout must not cut a clarification
        # short; the task's timeout is the bound.
        env={"MCP_TOOL_TIMEOUT": str(int(spec.timeout_seconds * 1000) + 60_000)},
    )


class Driver:
    """One conversation: brief in, steering pumped, events out, outcome back."""

    def __init__(
        self,
        agent: ClaudeAgent,
        spec: RunSpec,
        brief: str,
        inbox: Inbox,
        events: EventLog,
        progress: Progress,
    ) -> None:
        self.agent = agent
        self.spec = spec
        self.brief = brief
        self.inbox = inbox
        self.events = events
        self.progress = progress
        self.last_result: ResultView | None = None
        self.saw_init = False
        self._send_lock = anyio.Lock()

    async def run(self) -> Outcome:
        timeout = self.spec.timeout_seconds
        with anyio.move_on_after(timeout):
            try:
                async with self.agent.converse() as conversation:
                    return await self._drive(conversation)
            except Exception as exc:
                return self._failure(exc)
        return self.partial(
            "failed", "timeout", error=f"the task exceeded its {timeout:g}s timeout"
        )

    def partial(
        self, status: RunStatus, reason: TerminationReason, *, error: str | None = None
    ) -> Outcome:
        """An outcome that keeps what the conversation produced so far."""
        last = self.last_result
        return Outcome(
            status=status,
            reason=reason,
            final_text=last.text if last is not None else None,
            usage=last.usage if last is not None else None,
            error=error,
        )

    def _failure(self, exc: BaseException) -> Outcome:
        if self.spec.resume_task_id is not None and not self.saw_init:
            return self.partial(
                "failed",
                "resume_failed",
                error=(
                    f"Claude could not resume session {self.spec.session_id} "
                    f"(it may have been deleted); see worker.stderr. {_describe(exc)}"
                ),
            )
        return self.partial("failed", "sdk_error", error=_describe(exc))

    async def _drive(self, conversation: Conversation) -> Outcome:
        await conversation.send(self.brief, uuid=str(uuid.uuid4()))
        # A pump failure (the CLI's stdin gone) surfaces from the task group
        # as an exception group, which ``run`` turns into an outcome.
        outcome: Outcome | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(self._pump, conversation)
            try:
                outcome = await self._read(conversation)
            except Exception as exc:
                outcome = self._failure(exc)
            tg.cancel_scope.cancel()
        assert outcome is not None
        return outcome

    async def _pump(self, conversation: Conversation) -> None:
        while True:
            async with self._send_lock:
                for message in self.inbox.queued():
                    attempt = self.inbox.mark_attempted(message)
                    if attempt is None:
                        continue
                    await conversation.send(attempt.text, uuid=attempt.uuid)
                    self.inbox.mark_delivered(attempt)
            self.progress.tick()
            await anyio.sleep(PUMP_INTERVAL_SECONDS)

    async def _read(self, conversation: Conversation) -> Outcome:
        async for message in conversation.messages():
            view = describe(message)
            if view.kind == "init":
                self.saw_init = True
                expected = self.spec.session_id
                if expected and view.session_id and view.session_id != expected:
                    return self.partial(
                        "failed",
                        "resume_failed" if self.spec.resume_task_id else "sdk_error",
                        error=(
                            f"Claude reported session {view.session_id}, not the "
                            f"expected {expected}; the conversation was stopped"
                        ),
                    )
                continue
            if view.kind == "user_input":
                if view.uuid:
                    self.inbox.mark_in_conversation(view.uuid)
                continue
            for kind, fields in events_from_view(view):
                self.progress.note(self.events.append(kind, **fields))
            if view.result is None:
                continue
            result = view.result
            self.last_result = result
            self.progress.record_usage(result.usage, self.spec.cost_baseline_usd)
            if result.is_error:
                reason = _SUBTYPE_REASONS.get(result.subtype, "sdk_error")
                return Outcome(
                    status="failed",
                    reason=reason,
                    final_text=result.text,
                    usage=result.usage,
                    error=_result_error(result),
                )
            async with self._send_lock:
                if self.inbox.close_if_idle():
                    return Outcome(
                        status="completed",
                        reason="finished",
                        final_text=result.text,
                        usage=result.usage,
                    )
        return self.partial(
            "failed",
            "sdk_error",
            error="the Claude stream ended without a final result",
        )


class AskParent:
    """The ``ask_parent`` local tool: record a question, block for the answer."""

    def __init__(self, run_dir: Path, inbox: Inbox, progress: Progress) -> None:
        self.run_dir = run_dir
        self.inbox = inbox
        self.progress = progress

    def tool(self) -> LocalTool:
        return LocalTool(
            name=ASK_TOOL,
            description=ASK_DESCRIPTION,
            input_schema=ASK_SCHEMA,
            handler=self.handle,
        )

    async def handle(self, arguments: dict[str, Any]) -> LocalToolResult:
        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            return LocalToolResult(
                text="`question` must be a non-empty string.", is_error=True
            )
        if len(question) > MAX_QUESTION_CHARS:
            return LocalToolResult(
                text=f"`question` must be at most {MAX_QUESTION_CHARS} characters.",
                is_error=True,
            )
        raw_choices = arguments.get("choices") or []
        if (
            not isinstance(raw_choices, list)
            or len(raw_choices) > MAX_CHOICES
            or not all(
                isinstance(c, str) and 0 < len(c) <= MAX_CHOICE_CHARS
                for c in raw_choices
            )
        ):
            return LocalToolResult(
                text=(
                    f"`choices` must be a list of at most {MAX_CHOICES} non-empty "
                    f"strings of up to {MAX_CHOICE_CHARS} characters."
                ),
                is_error=True,
            )
        request = self.inbox.ask(question, list(raw_choices))
        self.progress.flush()
        self._sync_status()
        log.info("asked parent %s", request.request_id)
        while True:
            current = self.inbox.read_input(request.request_id)
            if current is not None and current.state != "pending":
                break
            await anyio.sleep(ANSWER_POLL_SECONDS)
        self._sync_status()
        return _answer_text(current)

    def _sync_status(self) -> None:
        """``input_required`` while any request is pending, else ``working``.

        Decided under the inbox lock, so a request asked concurrently can
        never be left pending under a ``working`` status.
        """
        with self.inbox.locked():
            pending = self.inbox.pending_inputs()
            message = (
                f"Waiting for an answer: {pending[0].question.strip().splitlines()[0]}"
                if pending
                else None
            )

            def transition(state: RunState) -> RunState:
                if state.is_terminal:
                    return state
                if pending:
                    return state.model_copy(
                        update={"status": "input_required", "status_message": message}
                    )
                return state.model_copy(update={"status": "working"})

            update_state(self.run_dir, transition)


def _answer_text(request: InputRequest) -> LocalToolResult:
    if request.state == "answered":
        return LocalToolResult(text=f"Your parent answered: {request.answer}")
    if request.state == "declined":
        return LocalToolResult(
            text=(
                "Your parent declined to answer. Continue with your best "
                "judgement and say in your final message what you assumed."
            )
        )
    return LocalToolResult(
        text=f"No answer will come: {request.reason or 'the request expired'}.",
        is_error=True,
    )


class Progress:
    """Carry the latest progress line, activity time and usage to status.json.

    Writes are throttled to one per :data:`PROGRESS_FLUSH_SECONDS` for
    ordinary events; usage and explicit flushes write at once.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.status_message: str | None = None
        self.last_activity_at: datetime | None = None
        self.usage: Usage | None = None
        self.invocation_cost_usd: float | None = None
        self._last_flush = 0.0
        self._dirty = False

    def note(self, event: Event) -> None:
        if event.is_activity:
            self.last_activity_at = event.at
            self._dirty = True
        line = progress_line(event)
        if line is not None:
            self.status_message = line
            self._dirty = True
        self.tick()

    def tick(self) -> None:
        """Write pending progress once the throttle window has passed.

        Called after every event and from the worker's pump loop, so the
        last line before a long quiet tool call still reaches status.json.
        """
        if (
            self._dirty
            and time.monotonic() - self._last_flush >= PROGRESS_FLUSH_SECONDS
        ):
            self.flush()

    def record_usage(self, usage: Usage, baseline: float | None) -> None:
        self.usage = usage
        self.invocation_cost_usd = invocation_cost(usage, baseline)
        self.flush()

    def flush(self) -> None:
        def transition(state: RunState) -> RunState:
            if state.is_terminal:
                return state
            update: dict[str, Any] = {
                "last_activity_at": self.last_activity_at or state.last_activity_at,
                "usage": self.usage or state.usage,
                "invocation_cost_usd": self.invocation_cost_usd,
            }
            if state.status != "input_required" and self.status_message is not None:
                update["status_message"] = self.status_message
            return state.model_copy(update=update)

        update_state(self.run_dir, transition)
        self._last_flush = time.monotonic()
        self._dirty = False


def invocation_cost(usage: Usage | None, baseline: float | None) -> float | None:
    """This invocation's cost: the session's cumulative cost minus where it began."""
    if usage is None or usage.estimated_cost_usd is None:
        return None
    return max(0.0, usage.estimated_cost_usd - (baseline or 0.0))


def _finish(
    run_dir: Path,
    spec: RunSpec,
    outcome: Outcome,
    transcript: TranscriptWriter,
    inbox: Inbox,
    events: EventLog,
) -> None:
    """Leave nothing behind: settle the inbox, sweep processes, record the outcome.

    The sweep comes before the record so that a terminal ``status.json``
    also means the task's processes are gone (bar this one, which exits next).
    """
    transcript.close()
    inbox.close(f"the task ended ({outcome.status}/{outcome.reason})")
    swept = sweep_own_group(run_dir)
    if swept:
        log.warning("task %s: killed %d leftover process(es)", spec.task_id, swept)
    _record(run_dir, spec, outcome, events)
    log.info("task %s %s/%s", spec.task_id, outcome.status, outcome.reason)


async def _cancel_on_signal(scope: anyio.CancelScope) -> None:
    with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
        async for signum in signals:
            log.info("received %s; cancelling the run", signal.Signals(signum).name)
            scope.cancel()
            return


def _record(run_dir: Path, spec: RunSpec, outcome: Outcome, events: EventLog) -> None:
    finished_at = utc_now()
    final_path = (
        write_final_text(run_dir, outcome.final_text) if outcome.final_text else None
    )
    cost = invocation_cost(outcome.usage, spec.cost_baseline_usd)
    write_result(
        run_dir,
        RunResult(
            task_id=spec.task_id,
            status=outcome.status,
            termination_reason=outcome.reason,
            finished_at=finished_at,
            final_text=outcome.final_text,
            usage=outcome.usage,
            error=outcome.error,
            session_id=spec.session_id,
            final_text_path=final_path,
            invocation_cost_usd=cost,
        ),
    )
    summary = _summary(outcome)

    def transition(state: RunState) -> RunState:
        return state.model_copy(
            update={
                "status": outcome.status,
                "termination_reason": outcome.reason,
                "finished_at": finished_at,
                "final_text": outcome.final_text,
                "usage": outcome.usage,
                "error": outcome.error,
                "status_message": summary,
                "final_text_path": final_path,
                "invocation_cost_usd": cost,
            }
        )

    update_state(run_dir, transition)
    events.append(
        "finished",
        text=summary,
        detail={"status": outcome.status, "termination_reason": outcome.reason},
    )


def _summary(outcome: Outcome) -> str:
    if outcome.status == "completed":
        first = (outcome.final_text or "").strip().splitlines()
        return f"Finished: {first[0][:200]}" if first else "Finished"
    if outcome.status == "cancelled":
        return "Cancelled"
    return f"Failed ({outcome.reason}): {(outcome.error or '').splitlines()[0][:200]}"


def _result_error(result: ResultView) -> str:
    errors = "; ".join(result.errors) or result.subtype
    return _bounded(
        f"agent call failed (subtype={result.subtype!r}, "
        f"stop_reason={result.stop_reason!r}, errors={errors!r})"
    )


def _describe(exc: BaseException) -> str:
    """An error string for ``status.json``: type and message, bounded.

    The SDK's messages describe the CLI's exit or result subtype; they do
    not carry credentials. The bound keeps a runaway stderr dump out of the
    status file, whose readers want a sentence.
    """
    text = str(exc).strip()
    return _bounded(f"{type(exc).__name__}: {text}" if text else type(exc).__name__)


def _bounded(text: str) -> str:
    if len(text) > MAX_ERROR_CHARS:
        return text[: MAX_ERROR_CHARS - 1] + "…"
    return text


class TranscriptWriter:
    """One JSON object per SDK message, appended as it arrives and flushed."""

    def __init__(self, path: Path) -> None:
        self._handle = os.fdopen(
            open_private(path, append=False), "w", encoding="utf-8"
        )

    def write(self, message: Message) -> None:
        try:
            line = json.dumps(message_to_json(message), default=str)
            self._handle.write(line + "\n")
            self._handle.flush()
        except Exception:  # the observer must never raise into the stream
            log.exception("could not write a transcript line")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


class TelemetryFile:
    """A ``TelemetrySink`` that rewrites ``telemetry.json`` after every record."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._calls: list[AICallTelemetry] = []
        self._flush()  # the file exists even for a run that never gets a result

    def record_call(self, record: AICallTelemetry) -> None:
        self._calls.append(record.model_copy(update={"attempt": len(self._calls) + 1}))
        self._flush()

    def record_cache_hit(self, *, stage: str, model: str | None = None) -> None:
        self.record_call(AICallTelemetry.cache_hit(stage=stage, model=model))

    def _flush(self) -> None:
        write_model(self._path, RunTelemetry(calls=list(self._calls)))
