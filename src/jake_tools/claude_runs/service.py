"""The delegated-task service: one domain model behind every caller.

:class:`TaskService` is what the ordinary ``claude_*`` MCP tools call, and
what a native MCP Tasks adapter (:mod:`.tasks_protocol`) would project
from. Everything is read from the run directory and the process table, so
any server instance answers for any task.

Observation is :meth:`TaskService.poll`: a bounded wait (never longer than
:data:`MAX_WAIT_SECONDS`, which is not the task's timeout) that returns as
soon as there are events past the caller's cursor, the task needs input,
or it has ended, and otherwise at the deadline. Cancelling a poll cancels
only the wait; the worker is another process. :meth:`TaskService.transcript`
pages the same events with larger bounds. Neither scans the whole log:
both seek to the cursor's offset.
"""

from __future__ import annotations

import contextlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import anyio
from anyio import to_thread
from pydantic import BaseModel, Field

from ..ai_usage import Usage
from ..claude import EffortLevel
from .control import (
    ArgumentError,
    RunDefaults,
    StartRequest,
    cancel_task,
    reconcile,
    start_task,
)
from .events import Cursor, Event, EventLog, progress_line
from .inbox import (
    AnswerResult,
    Inbox,
    InboxClosedError,
    InputResponse,
    SteerSubmission,
)
from .models import (
    SCHEMA_VERSION,
    ClaudeStartResult,
    DeliveryState,
    InputRequest,
    RunSpec,
    RunState,
    RunStatus,
    SteerMessage,
    TerminationReason,
)
from .rundir import (
    EVENTS_FILE,
    STATUS_FILE,
    TRANSCRIPT_FILE,
    read_spec,
    read_state,
    task_dir,
)

MAX_WAIT_SECONDS = 25.0
"""Upper bound on one poll's wait: below the ~30 s per-call timeout an MCP
host must already allow for ``claude_cancel``."""
DEFAULT_WAIT_SECONDS = 20.0
POLL_DEFAULT_EVENTS = 20
POLL_MAX_EVENTS = 100
POLL_TEXT_CHARS = 1_000
POLL_MAX_BYTES = 32_000
TRANSCRIPT_DEFAULT_EVENTS = 50
TRANSCRIPT_MAX_EVENTS = 200
TRANSCRIPT_DEFAULT_TEXT_CHARS = 4_000
TRANSCRIPT_MAX_TEXT_CHARS = 50_000
TRANSCRIPT_MAX_BYTES = 200_000
FINAL_TEXT_CHARS = 20_000
MESSAGES_SHOWN = 20
SUGGESTED_POLL_INTERVAL_MS = 5_000
WAIT_TICK_SECONDS = 0.25
LIVENESS_CHECK_SECONDS = 2.0


class TaskNotRunningError(RuntimeError):
    """The task can no longer take messages (it ended, or predates steering)."""

    def __init__(self, task_id: str, status: RunStatus, reason: str) -> None:
        super().__init__(reason)
        self.task_id = task_id
        self.status = status
        self.reason = reason


class Limits(BaseModel):
    """The hard limits this invocation runs under (never changed mid-run)."""

    model: str
    effort: EffortLevel | None
    tools: tuple[str, ...]
    """The built-in tools requested for the session to expose."""
    allowed_tools: tuple[str, ...]
    """The permission rules that auto-approve calls (every exposed tool,
    when the request gave none). Calls the runtime needs no approval for,
    such as reads inside the working directory and read-only Bash commands,
    run without a rule; everything else is denied."""
    max_turns: int
    timeout_seconds: float
    max_budget_usd: float
    allow_questions: bool
    deadline_at: datetime | None
    """``started_at + timeout_seconds``: the wall clock keeps running while
    the task waits for an answer."""


class SteerStatus(BaseModel):
    """A steering message's delivery state, without its text."""

    message_id: str
    state: DeliveryState
    submitted_at: datetime
    delivered_at: datetime | None = None
    in_conversation_at: datetime | None = None
    reason: str | None = None

    @classmethod
    def of(cls, message: SteerMessage) -> SteerStatus:
        return cls(
            message_id=message.message_id,
            state=message.state,
            submitted_at=message.submitted_at,
            delivered_at=message.delivered_at,
            in_conversation_at=message.in_conversation_at,
            reason=message.reason,
        )


class TaskResult(BaseModel):
    """What a finished task left: its answer, outcome and where the rest is.

    ``final_text`` is the agent's final answer (its claim, not verified
    fact), cut to :data:`FINAL_TEXT_CHARS` with ``final_text_truncated``;
    ``final_text_path`` holds it in full. On a limit, error or cancel it is
    the last completed turn's answer, if any, and ``last_narration`` is the
    agent's latest public message, as recovery context. ``resumable`` means
    a Claude session is on record for ``claude_start(resume_task_id=...)``;
    only the session's latest task is accepted there (an older one gets
    ``resume_unavailable`` naming the latest), and Claude Code may still
    have deleted the session (its own retention applies).
    """

    outcome: RunStatus
    termination_reason: TerminationReason | None
    final_text: str | None = None
    final_text_chars: int | None = None
    final_text_truncated: bool = False
    final_text_path: Path | None = None
    last_narration: str | None = None
    error: str | None = None
    session_id: str | None = None
    resumable: bool = False
    usage: Usage | None = None
    invocation_cost_usd: float | None = None
    transcript_path: Path
    events_path: Path | None = None


class TaskView(BaseModel):
    """A task's state, as every observation reports it.

    ``status`` is the task lifecycle (``working``, ``input_required``,
    ``completed``, ``failed``, ``cancelled``); ``termination_reason`` is the
    worker outcome. ``pending_inputs`` are the open clarification requests,
    to answer with ``claude_update``. ``legacy_run`` marks a run created
    before events existed: it has no event log, only its raw transcript.
    """

    task_id: str
    status: RunStatus
    termination_reason: TerminationReason | None = None
    status_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    last_activity_at: datetime | None = None
    last_updated_at: datetime
    cancel_requested_at: datetime | None = None
    session_id: str | None = None
    resumed_from: str | None = None
    usage: Usage | None = None
    invocation_cost_usd: float | None = None
    invocation_cost_exact: bool = True
    """False for a resume whose parent ended mid-turn: the session may have
    spent more than its last result reported, so this invocation's cost
    (measured from that report) may be overstated."""
    session_cost_usd: float | None = None
    limits: Limits
    pending_inputs: list[InputRequest] = Field(default_factory=list)
    messages: list[SteerStatus] = Field(default_factory=list)
    messages_total: int = 0
    result: TaskResult | None = None
    run_dir: Path
    legacy_run: bool = False
    suggested_poll_interval_ms: int = SUGGESTED_POLL_INTERVAL_MS


WakeReason = Literal["events", "input_required", "terminal", "deadline"]


class PollResult(TaskView):
    """:class:`TaskView` plus the events after the caller's cursor.

    Pass ``next_cursor`` to the next poll to get only newer events;
    ``has_more`` means more were already written than this page holds.
    ``events_truncated`` means some event text was cut for this page
    (``claude_transcript`` returns more of it). ``wake_reason`` says why
    the poll returned.
    """

    events: list[Event]
    next_cursor: str
    has_more: bool
    events_truncated: bool
    wake_reason: WakeReason
    waited_seconds: float


class TranscriptPage(BaseModel):
    """A page of normalised events, larger than a poll's, plus the raw path."""

    task_id: str
    status: RunStatus
    events: list[Event]
    next_cursor: str
    has_more: bool
    truncated: bool
    raw_transcript_path: Path
    legacy_run: bool = False


class SteerResult(BaseModel):
    """Where a steering message is now. ``replayed``: an earlier identical call."""

    task_id: str
    message_id: str
    state: DeliveryState
    replayed: bool
    submitted_at: datetime
    delivered_at: datetime | None = None
    in_conversation_at: datetime | None = None
    reason: str | None = None
    task_status: RunStatus


class UpdateResult(BaseModel):
    """The outcome per responded request, and what is still pending."""

    task_id: str
    results: list[AnswerResult]
    task_status: RunStatus
    pending_inputs: list[InputRequest]


class TaskService:
    """Delegated tasks under one runs directory."""

    def __init__(self, runs_dir: Path, defaults: RunDefaults) -> None:
        self.runs_dir = runs_dir
        self.defaults = defaults

    # --- lifecycle --------------------------------------------------------------

    def start(self, request: StartRequest) -> ClaudeStartResult:
        return start_task(self.runs_dir, request, self.defaults)

    def status(self, task_id: str) -> RunState:
        return reconcile(task_dir(self.runs_dir, task_id))

    async def cancel(self, task_id: str) -> RunState:
        return await cancel_task(self.runs_dir, task_id)

    # --- observation ------------------------------------------------------------

    def view(self, task_id: str) -> TaskView:
        run_dir = task_dir(self.runs_dir, task_id)
        return build_view(run_dir, reconcile(run_dir))

    async def poll(
        self,
        task_id: str,
        *,
        cursor: str | None = None,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        limit: int = POLL_DEFAULT_EVENTS,
    ) -> PollResult:
        """Wait up to ``wait_seconds`` for something worth reporting.

        Raises :class:`~.events.InvalidCursorError` for a bad cursor and
        :class:`~.control.ArgumentError` for out-of-range bounds, before
        waiting.
        """
        if not 0 <= wait_seconds <= MAX_WAIT_SECONDS:
            raise ArgumentError(
                "wait_seconds", f"must be between 0 and {MAX_WAIT_SECONDS:g}"
            )
        if not 1 <= limit <= POLL_MAX_EVENTS:
            raise ArgumentError("limit", f"must be between 1 and {POLL_MAX_EVENTS}")
        run_dir = task_dir(self.runs_dir, task_id)
        log = EventLog(run_dir)
        position = Cursor.parse(cursor, task_id) if cursor else Cursor.start(task_id)
        log.page(position, limit=1, max_text_chars=0, max_bytes=0)  # validate now
        started = time.monotonic()
        deadline = started + wait_seconds
        last_check = float("-inf")
        state = read_state(run_dir)
        while True:
            now = time.monotonic()
            if now - last_check >= LIVENESS_CHECK_SECONDS:
                state = await to_thread.run_sync(reconcile, run_dir)
                last_check = now
            else:
                state = read_state(run_dir)
            page = log.page(
                position,
                limit=limit,
                max_text_chars=POLL_TEXT_CHARS,
                max_bytes=POLL_MAX_BYTES,
            )
            reason: WakeReason | None = None
            if page.events:
                reason = "events"
            elif state.is_terminal:
                reason = "terminal"
            elif state.status == "input_required":
                reason = "input_required"
            elif now >= deadline:
                reason = "deadline"
            if reason is not None:
                view = await to_thread.run_sync(build_view, run_dir, state)
                view = with_page_progress(view, state, page.events)
                return PollResult(
                    **view.model_dump(),
                    events=page.events,
                    next_cursor=page.next_cursor,
                    has_more=page.has_more,
                    events_truncated=page.truncated,
                    wake_reason=reason,
                    waited_seconds=round(time.monotonic() - started, 3),
                )
            await anyio.sleep(min(WAIT_TICK_SECONDS, max(0.0, deadline - now)))

    def transcript(
        self,
        task_id: str,
        *,
        cursor: str | None = None,
        limit: int = TRANSCRIPT_DEFAULT_EVENTS,
        max_text_chars: int = TRANSCRIPT_DEFAULT_TEXT_CHARS,
    ) -> TranscriptPage:
        if not 1 <= limit <= TRANSCRIPT_MAX_EVENTS:
            raise ArgumentError(
                "limit", f"must be between 1 and {TRANSCRIPT_MAX_EVENTS}"
            )
        if not 1 <= max_text_chars <= TRANSCRIPT_MAX_TEXT_CHARS:
            raise ArgumentError(
                "max_text_chars", f"must be between 1 and {TRANSCRIPT_MAX_TEXT_CHARS}"
            )
        run_dir = task_dir(self.runs_dir, task_id)
        state = read_state(run_dir)
        position = Cursor.parse(cursor, task_id) if cursor else Cursor.start(task_id)
        page = EventLog(run_dir).page(
            position,
            limit=limit,
            max_text_chars=max_text_chars,
            max_bytes=TRANSCRIPT_MAX_BYTES,
        )
        return TranscriptPage(
            task_id=task_id,
            status=state.status,
            events=page.events,
            next_cursor=page.next_cursor,
            has_more=page.has_more,
            truncated=page.truncated,
            raw_transcript_path=run_dir / TRANSCRIPT_FILE,
            legacy_run=_is_legacy(read_spec(run_dir)),
        )

    # --- steering and clarification ---------------------------------------------

    def steer(self, task_id: str, submission: SteerSubmission) -> SteerResult:
        """Queue ``submission`` for delivery, idempotently by ``message_id``.

        Raises :class:`~.inbox.ConflictError` for a reused ID with another
        message and :class:`TaskNotRunningError` when the task cannot take
        one. A retry of a stored message always reports its current state.
        """
        run_dir = task_dir(self.runs_dir, task_id)
        state = reconcile(run_dir)
        if _is_legacy(read_spec(run_dir)):
            raise TaskNotRunningError(
                task_id,
                state.status,
                "this task was started by a version without steering; it cannot take messages",
            )
        try:
            message, replayed = Inbox(run_dir).submit(submission)
        except InboxClosedError as exc:
            raise TaskNotRunningError(
                task_id,
                state.status,
                f"the task is {state.status} and no longer accepts messages",
            ) from exc
        return SteerResult(
            task_id=task_id,
            message_id=message.message_id,
            state=message.state,
            replayed=replayed,
            submitted_at=message.submitted_at,
            delivered_at=message.delivered_at,
            in_conversation_at=message.in_conversation_at,
            reason=message.reason,
            task_status=read_state(run_dir).status,
        )

    def respond(
        self, task_id: str, responses: dict[str, InputResponse]
    ) -> UpdateResult:
        """Answer or decline pending clarification requests, keyed by ID."""
        run_dir = task_dir(self.runs_dir, task_id)
        reconcile(run_dir)
        inbox = Inbox(run_dir)
        results = inbox.respond(responses)
        return UpdateResult(
            task_id=task_id,
            results=results,
            task_status=read_state(run_dir).status,
            pending_inputs=inbox.pending_inputs(),
        )


def with_page_progress(
    view: TaskView, state: RunState, events: list[Event]
) -> TaskView:
    """``view`` brought up to date with the events returned alongside it.

    The worker appends an event before its (throttled) progress write, and a
    poll reads state before the page, so a page can hold events newer than
    the snapshot. A returned event's activity time or progress line is used
    only when strictly newer than what the snapshot already holds, so an old
    cursor's page never rolls progress back. A question or an end owns the
    summary and is never replaced.
    """
    update: dict[str, object] = {}
    activity = [e.at for e in events if e.is_activity]
    if activity and (
        view.last_activity_at is None or max(activity) > view.last_activity_at
    ):
        update["last_activity_at"] = max(activity)
    if view.status == "working":
        for event in reversed(events):
            line = progress_line(event)
            if line is None:
                continue
            if state.status_message_at is None or event.at > state.status_message_at:
                update["status_message"] = line
            break
    return view.model_copy(update=update) if update else view


def build_view(run_dir: Path, state: RunState) -> TaskView:
    """The :class:`TaskView` of a run directory whose state is ``state``."""
    spec = read_spec(run_dir)
    inbox = Inbox(run_dir)
    messages = inbox.messages()
    deadline = None
    if state.started_at is not None:
        deadline = datetime.fromtimestamp(
            state.started_at.timestamp() + spec.timeout_seconds, tz=UTC
        )
    return TaskView(
        task_id=state.task_id,
        status=state.status,
        termination_reason=state.termination_reason,
        status_message=state.status_message,
        created_at=state.created_at,
        started_at=state.started_at,
        finished_at=state.finished_at,
        last_activity_at=state.last_activity_at,
        last_updated_at=_last_updated(run_dir, state),
        cancel_requested_at=state.cancel_requested_at,
        session_id=state.session_id or spec.session_id,
        resumed_from=state.resumed_from or spec.resume_task_id,
        usage=state.usage,
        invocation_cost_usd=state.invocation_cost_usd,
        invocation_cost_exact=spec.cost_baseline_exact,
        session_cost_usd=state.session_cost_usd,
        limits=Limits(
            model=spec.model,
            effort=spec.effort,
            tools=spec.tools,
            allowed_tools=spec.approval_rules,
            max_turns=spec.max_turns,
            timeout_seconds=spec.timeout_seconds,
            max_budget_usd=spec.max_budget_usd,
            allow_questions=spec.allow_questions,
            deadline_at=deadline,
        ),
        pending_inputs=[r for r in inbox.inputs() if r.state == "pending"],
        messages=[SteerStatus.of(m) for m in messages[-MESSAGES_SHOWN:]],
        messages_total=len(messages),
        result=_result(run_dir, spec, state) if state.is_terminal else None,
        run_dir=run_dir,
        legacy_run=_is_legacy(spec),
    )


def _result(run_dir: Path, spec: RunSpec, state: RunState) -> TaskResult:
    text = state.final_text
    events_path = run_dir / EVENTS_FILE
    narration = None
    if state.status != "completed" and events_path.exists():
        narration = _last_narration(EventLog(run_dir))
    session_id = state.session_id or spec.session_id
    return TaskResult(
        outcome=state.status,
        termination_reason=state.termination_reason,
        final_text=text[:FINAL_TEXT_CHARS] if text is not None else None,
        final_text_chars=len(text) if text is not None else None,
        final_text_truncated=text is not None and len(text) > FINAL_TEXT_CHARS,
        final_text_path=state.final_text_path,
        last_narration=narration,
        error=state.error,
        session_id=session_id,
        resumable=session_id is not None
        and state.termination_reason != "resume_failed",
        usage=state.usage,
        invocation_cost_usd=state.invocation_cost_usd,
        transcript_path=run_dir / TRANSCRIPT_FILE,
        events_path=events_path if events_path.exists() else None,
    )


def _last_narration(log: EventLog) -> str | None:
    """The latest ``assistant_text`` near the end of the log, bounded."""
    found = next(
        (e.text for e in reversed(log.tail(256_000)) if e.kind == "assistant_text"),
        None,
    )
    if found is not None and len(found) > POLL_TEXT_CHARS:
        found = found[:POLL_TEXT_CHARS] + "…"
    return found


def _last_updated(run_dir: Path, state: RunState) -> datetime:
    stamps = [state.created_at.timestamp()]
    for name in (STATUS_FILE, EVENTS_FILE):
        with contextlib.suppress(FileNotFoundError):
            stamps.append((run_dir / name).stat().st_mtime)
    return datetime.fromtimestamp(max(stamps), tz=UTC)


def _is_legacy(spec: RunSpec) -> bool:
    return spec.schema_version < SCHEMA_VERSION
