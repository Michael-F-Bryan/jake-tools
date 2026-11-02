"""The delegated-task service: one domain model behind every caller.

:class:`TaskService` is what the ordinary ``claude_*`` MCP tools call, and
what a native MCP Tasks adapter (:mod:`.tasks_protocol`) would project
from. Everything is read from the run directory and the process table, so
any server instance answers for any task.

Supervision is :meth:`TaskService.poll`: a bounded wait (never longer than
:data:`MAX_WAIT_SECONDS`, which is not the task's timeout) that returns a
compact :class:`~.observe.Observation` as soon as something needs the
caller (see :mod:`.observe` for the wake rules and observation tokens), and
otherwise at the deadline. Cancelling a poll cancels only the wait; the
worker is another process. History is :meth:`TaskService.transcript`,
paged by its own cursor; detail and full result text are
:meth:`TaskService.detail` and :meth:`TaskService.result_text`.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import anyio
from anyio import to_thread
from pydantic import BaseModel, Field

from ..ai_usage import Usage
from .control import (
    ArgumentError,
    RunDefaults,
    StartRequest,
    cancel_task,
    reconcile,
    start_task,
)
from .events import Cursor, Event, EventLog, InvalidCursorError
from .inbox import (
    AnswerResult,
    Inbox,
    InboxClosedError,
    InputResponse,
    SteerSubmission,
)
from .models import (
    SCHEMA_VERSION,
    TASK_ID_PATTERN,
    ClaudeStartResult,
    DeliveryState,
    Handoff,
    InputRequest,
    PermissionDenial,
    RunConfig,
    RunSpec,
    RunState,
    RunStatus,
    SteerMessage,
    TerminationReason,
)
from .observe import (
    QUESTION_CHARS,
    QUESTIONS_SHOWN,
    InvalidTokenError,
    Observation,
    PollUntil,
    Question,
    Retrieval,
    Scanner,
    TokenStatus,
    WakeReason,
    bounded_progress,
    decode_token,
    diagnostics_of,
    encode_token,
    known_denials,
    outcome_of,
    recent_activity,
    result_route,
    wake_reason,
)
from .rundir import (
    EVENTS_FILE,
    STATUS_FILE,
    TRANSCRIPT_FILE,
    read_brief,
    read_spec,
    read_state,
    task_dir,
)

MAX_WAIT_SECONDS = 120.0
"""Upper bound on one poll's wait. Hermes allows an MCP call 300 s by
default (``mcp_servers.<name>.timeout`` / ``timeouts.mcp.tool_call``); the
wait stays well inside that, and the caller chooses it."""
DEFAULT_WAIT_SECONDS = 30.0
NARRATION_CHARS = 1_000
TRANSCRIPT_DEFAULT_EVENTS = 50
TRANSCRIPT_MAX_EVENTS = 200
TRANSCRIPT_DEFAULT_TEXT_CHARS = 4_000
TRANSCRIPT_MAX_TEXT_CHARS = 50_000
TRANSCRIPT_MAX_BYTES = 200_000
FINAL_TEXT_CHARS = 20_000
"""How much final text the native Tasks result carries inline."""
DETAIL_TEXT_CHARS = 2_000
"""How much final text ``claude_status``'s detail view carries inline."""
RESULT_CHUNK_DEFAULT = 15_000
RESULT_CHUNK_MAX = 30_000
"""Result chunks stay well under the 50 000 characters past which Hermes
replaces a tool result with a preview and a host-side file."""
MESSAGES_SHOWN = 20
SUGGESTED_POLL_INTERVAL_MS = 5_000
WAIT_TICK_SECONDS = 0.25
LIVENESS_CHECK_SECONDS = 2.0
LIST_DEFAULT_TASKS = 20
LIST_MAX_TASKS = 100
LIST_MAX_SCANNED = 500
LABEL_CHARS = 120


class TaskNotRunningError(RuntimeError):
    """The task can no longer take messages (it ended, or predates steering)."""

    def __init__(self, task_id: str, status: RunStatus, reason: str) -> None:
        super().__init__(reason)
        self.task_id = task_id
        self.status = status
        self.reason = reason


class Limits(RunConfig):
    """The configuration and hard limits this invocation runs under (never
    changed mid-run): :class:`~.models.RunConfig` plus what is known once it
    runs.

    Calls the runtime needs no approval for, such as reads inside the
    working directory and read-only Bash commands, run without a rule;
    with ``permission_mode="dontAsk"`` every other call no rule approves is
    denied (reported in ``permission_denials``), never prompted.
    """

    deadline_at: datetime | None = None
    """``started_at + timeout_seconds``: the wall clock keeps running while
    the task waits for an answer."""
    exposed_tools: list[str] | None = None
    """What the session's ``init`` listed (runtime evidence); ``None`` until
    the session started, or for runs that predate the record."""


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
    fact), cut with ``final_text_truncated`` (``retrieve`` is the
    ``claude_status`` call that returns all of it; ``final_text_path`` is
    the server-local file, supplementary evidence). On a limit, error or
    cancel it is the last completed turn's answer, if any, and
    ``last_narration`` is the agent's latest public message, as recovery
    context. ``resumable`` means a Claude session is on record for
    ``claude_start(resume_task_id=...)``; only the session's latest task is
    accepted there (an older one gets ``resume_unavailable`` naming the
    latest), and Claude Code may still have deleted the session (its own
    retention applies).

    ``handoff`` is set only after a ``max_turns`` stop: the worker's own
    tool-free account of what is done and what is left (see
    :class:`~.models.Handoff`). It does not change ``outcome`` or
    ``termination_reason``, and ``final_text``/``last_narration`` are kept
    whatever became of it. ``diagnostics`` are mechanically detected
    inconsistencies (see :func:`~.observe.diagnostics_of`).
    """

    outcome: RunStatus
    termination_reason: TerminationReason | None = None
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
    cost_complete: bool | None = None
    handoff: Handoff | None = None
    diagnostics: list[str] = Field(default_factory=list)
    retrieve: Retrieval | None = None
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
    turns_used: int | None = None
    """Model turns this invocation has taken, as the worker counts them to
    enforce ``limits.max_turns``; ``None`` for runs that predate it."""
    limits: Limits
    permission_denials: list[PermissionDenial] = Field(default_factory=list)
    """The latest calls the runtime refused, oldest first: the requested
    operation and the runtime's own message where it gave one. Compare
    with ``limits`` for the effective policy; no reason is inferred."""
    permission_denials_total: int = 0
    pending_inputs: list[InputRequest] = Field(default_factory=list)
    messages: list[SteerStatus] = Field(default_factory=list)
    messages_total: int = 0
    result: TaskResult | None = None
    run_dir: Path
    legacy_run: bool = False


MAX_POLL_TASKS = 16


class PollTask(BaseModel):
    """One watched task, with its own observation token."""

    task_id: str
    token: str | None = None


class MultiObservation(BaseModel):
    """Every watched task's snapshot, in request order, under one deadline.

    ``ready`` names the tasks that woke the poll (empty at the deadline);
    every task's ``observation_token`` is current, so pass each back.
    """

    tasks: list[Observation]
    ready: list[str]
    wake_reason: Literal["ready", "deadline"]
    waited_seconds: float


ResultPart = Literal["final_text", "handoff", "error"]


class ResultText(BaseModel):
    """One chunk of a finished task's text, for retrieval over MCP.

    ``text`` is ``total_chars`` long in full; this chunk starts at
    ``offset``, and ``next_offset`` is where the next one starts (``None``
    at the end). ``available`` is false when the task has not ended or left
    no such text. ``path`` is the server-local file, if any: supplementary,
    since the caller may be on another machine.
    """

    task_id: str
    status: RunStatus
    termination_reason: TerminationReason | None = None
    part: ResultPart
    available: bool
    text: str | None = None
    offset: int = 0
    total_chars: int | None = None
    next_offset: int | None = None
    path: Path | None = None


CancelTermination = Literal[
    "confirmed", "unconfirmed", "already_ended", "ended_before_cancel"
]


class CancelResult(BaseModel):
    """What ``claude_cancel`` established.

    ``termination`` is ``confirmed`` when every process of the task was seen
    gone and the task recorded ``cancelled``; ``unconfirmed`` when some could
    not be confirmed dead (``status`` is ``failed``, ``error`` says why);
    ``already_ended`` when the task had ended before the call; and
    ``ended_before_cancel`` when it reached its own end while the cancel ran
    (its outcome stands). ``rollback`` is always ``none``: nothing the task
    already changed is undone.
    """

    task_id: str
    status: RunStatus
    termination_reason: TerminationReason | None = None
    termination: CancelTermination
    cancel_requested_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    rollback: Literal["none"] = "none"


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


class TaskSummary(BaseModel):
    """One task in a listing: enough to pick it, not its full record.

    ``label`` is the brief's first non-blank line, cut to
    :data:`LABEL_CHARS`; ``claude_poll`` returns the rest. ``resumable``
    has :class:`TaskResult`'s meaning, and only the session's latest task
    is accepted by ``claude_start(resume_task_id=...)``.
    """

    task_id: str
    label: str
    status: RunStatus
    termination_reason: TerminationReason | None = None
    status_message: str | None = None
    created_at: datetime
    finished_at: datetime | None = None
    last_activity_at: datetime | None = None
    session_id: str | None = None
    resumed_from: str | None = None
    resumable: bool = False
    handoff_status: str | None = None
    legacy_run: bool = False


class UnreadableTask(BaseModel):
    """A run directory whose records could not be read; listed, not fatal."""

    task_id: str
    error: str


class TaskList(BaseModel):
    """A page of tasks, newest first by task ID: to the second of creation,
    then (for tasks created in the same second) by the ID's random part.

    Pass ``next_before`` as ``before`` for the next (older) page; it is
    ``None`` once every run directory has been seen. ``scanned`` is how
    many run directories this page read; a status filter can return fewer
    than ``limit`` tasks with more still to scan.
    """

    tasks: list[TaskSummary]
    unreadable: list[UnreadableTask] = Field(default_factory=list)
    next_before: str | None = None
    scanned: int


class TaskService:
    """Delegated tasks under one runs directory."""

    def __init__(
        self,
        runs_dir: Path,
        defaults: RunDefaults,
        worker_environ: Mapping[str, str] | None = None,
    ) -> None:
        """``worker_environ`` replaces this process's environment as the base
        a worker's environment is derived from. An in-process host whose own
        environment holds unrelated credentials (Hermes) passes an allowlist;
        the MCP server, launched with a filtered environment, passes nothing.
        """
        self.runs_dir = runs_dir
        self.defaults = defaults
        self.worker_environ = worker_environ

    # --- lifecycle --------------------------------------------------------------

    def start(self, request: StartRequest) -> ClaudeStartResult:
        started = start_task(self.runs_dir, request, self.defaults, self.worker_environ)
        run_dir = task_dir(self.runs_dir, started.task_id)
        return started.model_copy(
            update={
                "observation_token": encode_token(Cursor.start(started.task_id)),
                "config": RunConfig.of(read_spec(run_dir)),
            }
        )

    def status(self, task_id: str) -> RunState:
        return reconcile(task_dir(self.runs_dir, task_id))

    def detail(self, task_id: str) -> TaskView:
        """Everything about a task now, with its final text cut to an excerpt."""
        run_dir = task_dir(self.runs_dir, task_id)
        return build_view(run_dir, reconcile(run_dir), text_chars=DETAIL_TEXT_CHARS)

    def result_text(
        self,
        task_id: str,
        *,
        part: ResultPart = "final_text",
        offset: int = 0,
        max_chars: int = RESULT_CHUNK_DEFAULT,
    ) -> ResultText:
        """A chunk of a finished task's final text, handoff or error."""
        if offset < 0:
            raise ArgumentError("offset", "must be 0 or more")
        if not 1 <= max_chars <= RESULT_CHUNK_MAX:
            raise ArgumentError(
                "max_chars", f"must be between 1 and {RESULT_CHUNK_MAX}"
            )
        run_dir = task_dir(self.runs_dir, task_id)
        state = reconcile(run_dir)
        text, path = _result_source(state, part)
        base = ResultText(
            task_id=task_id,
            status=state.status,
            termination_reason=state.termination_reason,
            part=part,
            available=state.is_terminal and text is not None,
            path=path,
        )
        if not base.available or text is None:
            return base
        if offset > len(text):
            raise ArgumentError("offset", f"must be at most {len(text)}")
        end = offset + max_chars
        return base.model_copy(
            update={
                "text": text[offset:end],
                "offset": offset,
                "total_chars": len(text),
                "next_offset": end if end < len(text) else None,
            }
        )

    async def cancel(self, task_id: str) -> CancelResult:
        run_dir = task_dir(self.runs_dir, task_id)
        before = reconcile(run_dir)
        state = await cancel_task(self.runs_dir, task_id)
        termination: CancelTermination
        if before.is_terminal:
            termination = "already_ended"
        elif state.status == "cancelled":
            termination = "confirmed"
        elif state.termination_reason == "cancelled":
            termination = "unconfirmed"
        else:
            termination = "ended_before_cancel"
        return CancelResult(
            task_id=task_id,
            status=state.status,
            termination_reason=state.termination_reason,
            termination=termination,
            cancel_requested_at=state.cancel_requested_at,
            finished_at=state.finished_at,
            error=state.error,
        )

    # --- observation ------------------------------------------------------------

    def list(
        self,
        *,
        before: str | None = None,
        limit: int = LIST_DEFAULT_TASKS,
        status: RunStatus | None = None,
    ) -> TaskList:
        """Tasks newest first, from run directories alone (no index).

        Reads at most ``limit`` matching tasks and :data:`LIST_MAX_SCANNED`
        directories. Unfinished tasks are reconciled, so a dead worker is
        reported as such. Raises :class:`~.control.ArgumentError`.
        """
        if not 1 <= limit <= LIST_MAX_TASKS:
            raise ArgumentError("limit", f"must be between 1 and {LIST_MAX_TASKS}")
        if before is not None and not TASK_ID_PATTERN.fullmatch(before):
            raise ArgumentError("before", "must be a task_id from an earlier page")
        names = sorted(
            (
                entry.name
                for entry in (self.runs_dir.iterdir() if self.runs_dir.is_dir() else [])
                if TASK_ID_PATTERN.fullmatch(entry.name) and entry.is_dir()
            ),
            reverse=True,
        )
        if before is not None:
            names = [name for name in names if name < before]
        tasks: list[TaskSummary] = []
        unreadable: list[UnreadableTask] = []
        scanned = 0
        for name in names:
            if len(tasks) >= limit or scanned >= LIST_MAX_SCANNED:
                break
            scanned += 1
            try:
                summary = _summary(self.runs_dir / name)
            except Exception as exc:  # a malformed record must not hide the rest
                unreadable.append(
                    UnreadableTask(
                        task_id=name, error=f"{type(exc).__name__}: {exc}"[:300]
                    )
                )
                continue
            if status is None or summary.status == status:
                tasks.append(summary)
        return TaskList(
            tasks=tasks,
            unreadable=unreadable,
            next_before=names[scanned - 1] if scanned < len(names) else None,
            scanned=scanned,
        )

    def view(self, task_id: str) -> TaskView:
        run_dir = task_dir(self.runs_dir, task_id)
        return build_view(run_dir, reconcile(run_dir))

    async def poll(
        self,
        task_id: str,
        *,
        token: str | None = None,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        until: PollUntil = "attention",
    ) -> Observation:
        """Wait up to ``wait_seconds`` for something that needs the caller.

        Raises :class:`~.control.ArgumentError` for an out-of-range wait and
        :class:`~.rundir.UnknownTaskError`, before waiting. A bad token is
        not an error: the observation resynchronises (see :mod:`.observe`).
        """
        _check_wait(wait_seconds)
        watch = _Watch(task_dir(self.runs_dir, task_id), token)
        result = await _wait_for([watch], wait_seconds, until)
        return result.tasks[0]

    async def poll_many(
        self,
        tasks: list[PollTask],
        *,
        wait_seconds: float = DEFAULT_WAIT_SECONDS,
        until: PollUntil = "attention",
    ) -> MultiObservation:
        """Watch several tasks under one deadline, without affecting workers."""
        if not 1 <= len(tasks) <= MAX_POLL_TASKS:
            raise ArgumentError("tasks", f"must contain 1-{MAX_POLL_TASKS} tasks")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ArgumentError("tasks", "task IDs must be unique")
        _check_wait(wait_seconds)
        watches = [
            _Watch(task_dir(self.runs_dir, task.task_id), task.token) for task in tasks
        ]
        return await _wait_for(watches, wait_seconds, until)

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


def _check_wait(wait_seconds: float) -> None:
    if not 0 <= wait_seconds <= MAX_WAIT_SECONDS:
        raise ArgumentError(
            "wait_seconds", f"must be between 0 and {MAX_WAIT_SECONDS:g}"
        )


class _Watch:
    """One task under observation: its baseline, scanner and latest state."""

    def __init__(self, run_dir: Path, token: str | None) -> None:
        self.run_dir = run_dir
        self.task_id = run_dir.name
        self.log = EventLog(run_dir)
        self.spec = read_spec(run_dir)
        self.state = read_state(run_dir)
        self.token_status: TokenStatus = "none"
        self.resync_reason: str | None = None
        start = None
        if token is not None:
            try:
                start = decode_token(token, self.task_id)
                self.log.validate(start)
                self.token_status = "current"
            except (InvalidTokenError, InvalidCursorError) as exc:
                start = None
                self.token_status = "resynchronised"
                self.resync_reason = str(exc).replace("cursor", "token")
        if start is None:
            start = self.log.end()
        self.scanner = Scanner(
            self.log, start, known_denials(self.state.permission_denials, start.seq)
        )

    async def refresh(self, *, reconcile_now: bool) -> None:
        self.scanner.advance()
        if reconcile_now:
            self.state = await to_thread.run_sync(reconcile, self.run_dir)
        else:
            self.state = read_state(self.run_dir)

    def reason(self, until: PollUntil) -> WakeReason | None:
        pending = 1 if self.state.status == "input_required" else 0
        found = wake_reason(self.state, pending, self.scanner, until)
        if found is None and self.token_status == "resynchronised":
            return "resynchronised"
        return found

    def observe(self, reason: WakeReason, waited: float) -> Observation:
        # The position first, then the state: the snapshot is never older
        # than the token it comes with.
        position = self.scanner.advance()
        state = read_state(self.run_dir)
        pending = Inbox(self.run_dir).pending_inputs()
        scanned = self.token_status == "current" or self.scanner.events > 0
        result = None
        if state.is_terminal:
            session_id = state.session_id or self.spec.session_id
            narration = None
            if state.status != "completed" and self.log.exists():
                narration = _last_narration(self.log)
            result = outcome_of(
                self.run_dir,
                state,
                final_text=state.final_text,
                last_narration=narration,
                session_id=session_id,
                resumable=_resumable(state, session_id),
                diagnostics=diagnostics_of(state, state.final_text),
            )
        running, line, newer_activity = (
            (None, None, None)
            if state.is_terminal
            else recent_activity(self.log, state)
        )
        return Observation(
            task_id=self.task_id,
            status=state.status,
            termination_reason=state.termination_reason,
            wake_reason=reason,
            observation_token=encode_token(position),
            token_status=self.token_status,
            resync_reason=self.resync_reason,
            waited_seconds=round(waited, 3),
            progress=bounded_progress(line or state.status_message),
            last_activity_at=newer_activity or state.last_activity_at,
            running_tool=running,
            turns_used=state.turns_used,
            max_turns=self.spec.max_turns,
            cancel_requested_at=state.cancel_requested_at,
            pending_questions=[Question.of(r) for r in pending[:QUESTIONS_SHOWN]],
            pending_questions_total=len(pending),
            pending_questions_retrieve=Retrieval(
                tool="claude_status",
                arguments={"task_id": self.task_id, "view": "detail"},
            )
            if len(pending) > QUESTIONS_SHOWN
            or any(len(r.question) > QUESTION_CHARS for r in pending)
            else None,
            permission_denials_total=state.permission_denials_total,
            changes=self.scanner.changes() if scanned else None,
            result=result,
        )


async def _wait_for(
    watches: list[_Watch], wait_seconds: float, until: PollUntil
) -> MultiObservation:
    started = time.monotonic()
    deadline = started + wait_seconds
    last_check = float("-inf")
    while True:
        now = time.monotonic()
        reconcile_now = now - last_check >= LIVENESS_CHECK_SECONDS
        if reconcile_now:
            last_check = now
        reasons: list[WakeReason | None] = []
        for watch in watches:
            await watch.refresh(reconcile_now=reconcile_now)
            reasons.append(watch.reason(until))
        if any(reasons) or now >= deadline:
            break
        await anyio.sleep(min(WAIT_TICK_SECONDS, max(0.0, deadline - now)))
    waited = time.monotonic() - started
    observations = [
        await to_thread.run_sync(watch.observe, reason or "deadline", waited)
        for watch, reason in zip(watches, reasons, strict=True)
    ]
    ready = [w.task_id for w, r in zip(watches, reasons, strict=True) if r]
    return MultiObservation(
        tasks=observations,
        ready=ready,
        wake_reason="ready" if ready else "deadline",
        waited_seconds=round(waited, 3),
    )


def _resumable(state: RunState, session_id: str | None) -> bool:
    return session_id is not None and state.termination_reason != "resume_failed"


def _result_source(state: RunState, part: ResultPart) -> tuple[str | None, Path | None]:
    """The full text of one result part, and its file when there is one."""
    if part == "error":
        return state.error, None
    if part == "handoff":
        handoff = state.handoff
        if handoff is None:
            return None, None
        path = handoff.text_path
        if path is not None and path.is_file():
            return path.read_text(encoding="utf-8"), path
        return handoff.text, path
    path = state.final_text_path
    if path is not None and path.is_file():
        return path.read_text(encoding="utf-8"), path
    return state.final_text, path


def build_view(
    run_dir: Path, state: RunState, *, text_chars: int = FINAL_TEXT_CHARS
) -> TaskView:
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
        turns_used=state.turns_used,
        permission_denials=state.permission_denials,
        permission_denials_total=state.permission_denials_total,
        limits=Limits(
            **RunConfig.of(spec).model_dump(),
            deadline_at=deadline,
            exposed_tools=state.exposed_tools,
        ),
        pending_inputs=[r for r in inbox.inputs() if r.state == "pending"],
        messages=[SteerStatus.of(m) for m in messages[-MESSAGES_SHOWN:]],
        messages_total=len(messages),
        result=_result(run_dir, spec, state, text_chars) if state.is_terminal else None,
        run_dir=run_dir,
        legacy_run=_is_legacy(spec),
    )


def _result(
    run_dir: Path, spec: RunSpec, state: RunState, text_chars: int
) -> TaskResult:
    text = state.final_text
    events_path = run_dir / EVENTS_FILE
    narration = None
    if state.status != "completed" and events_path.exists():
        narration = _last_narration(EventLog(run_dir))
    session_id = state.session_id or spec.session_id
    truncated = text is not None and len(text) > text_chars
    return TaskResult(
        outcome=state.status,
        termination_reason=state.termination_reason,
        final_text=text[:text_chars] if text is not None else None,
        final_text_chars=len(text) if text is not None else None,
        final_text_truncated=truncated,
        final_text_path=state.final_text_path,
        last_narration=narration,
        error=state.error,
        session_id=session_id,
        resumable=_resumable(state, session_id),
        usage=state.usage,
        invocation_cost_usd=state.invocation_cost_usd,
        cost_complete=state.cost_complete,
        handoff=state.handoff,
        diagnostics=diagnostics_of(state, text),
        retrieve=result_route(state.task_id, "final_text") if truncated else None,
        transcript_path=run_dir / TRANSCRIPT_FILE,
        events_path=events_path if events_path.exists() else None,
    )


def _summary(run_dir: Path) -> TaskSummary:
    spec = read_spec(run_dir)
    state = read_state(run_dir)
    if not state.is_terminal:
        state = reconcile(run_dir)
    lines = [line.strip() for line in read_brief(run_dir).splitlines()]
    label = next((line for line in lines if line), "")
    if len(label) > LABEL_CHARS:
        label = label[: LABEL_CHARS - 1] + "…"
    session_id = state.session_id or spec.session_id
    return TaskSummary(
        task_id=state.task_id,
        label=label,
        status=state.status,
        termination_reason=state.termination_reason,
        status_message=state.status_message,
        created_at=state.created_at,
        finished_at=state.finished_at,
        last_activity_at=state.last_activity_at,
        session_id=session_id,
        resumed_from=state.resumed_from or spec.resume_task_id,
        resumable=state.is_terminal and _resumable(state, session_id),
        handoff_status=state.handoff.status if state.handoff else None,
        legacy_run=_is_legacy(spec),
    )


def _last_narration(log: EventLog) -> str | None:
    """The latest ``assistant_text`` near the end of the log, bounded."""
    found = next(
        (e.text for e in reversed(log.tail(256_000)) if e.kind == "assistant_text"),
        None,
    )
    if found is not None and len(found) > NARRATION_CHARS:
        found = found[:NARRATION_CHARS] + "…"
    return found


def _last_updated(run_dir: Path, state: RunState) -> datetime:
    stamps = [state.created_at.timestamp()]
    for name in (STATUS_FILE, EVENTS_FILE):
        with contextlib.suppress(FileNotFoundError):
            stamps.append((run_dir / name).stat().st_mtime)
    return datetime.fromtimestamp(max(stamps), tz=UTC)


def _is_legacy(spec: RunSpec) -> bool:
    return spec.schema_version < SCHEMA_VERSION
