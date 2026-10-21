"""Start, status, cancel and capacity for delegated tasks.

These are the state-changing operations behind the ``claude_*`` MCP tools,
minus the transport (:mod:`.service` adds the reads and steering). Each one works from disk plus the process table, never from
server memory, so a freshly started server answers for tasks an earlier
one began, and several servers can share one runs directory.

Liveness is "the worker holds its lock" (see :func:`~.spawn.worker_alive`),
falling back to "the recorded PID is running, not a zombie, and started
when we recorded it" for the instant between the spawn and the worker
taking the lock. A zombie worker therefore reads as dead, and a PID the OS
has since reused is never mistaken for ours or signalled.

:func:`start_task` holds ``<runs_dir>/.runs.lock`` across the capacity
check, the directory creation, the spawn and the recording of the PID, so
two servers cannot both pass the check and no reader sees a half-started
task. It is synchronous, so the same holds for two calls on one server.
The same lock makes two more checks race-free: a ``request_id`` maps to at
most one task (``<runs_dir>/.requests/``), and no two live tasks write the
same Claude session.
:func:`cancel_task` is async because it waits on processes; every wait is
an anyio sleep.

Every way a task ends closes its inbox (:meth:`~.inbox.Inbox.close`), so an
accepted steering message or a pending question is always settled, never
left looking live.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from ..claude import EffortLevel, permission_rule_tool
from ..config import Config
from .events import EventLog
from .inbox import Inbox
from .models import (
    CALLER_ID_PATTERN,
    SCHEMA_VERSION,
    ClaudeStartResult,
    PermissionRule,
    RequestRecord,
    RunSpec,
    RunState,
    RunStatus,
    TerminationReason,
    ToolName,
)
from .procs import process_table
from .rundir import (
    DIR_MODE,
    REQUESTS_DIR,
    TRANSCRIPT_FILE,
    create_run_dir,
    ensure_runs_dir,
    list_task_dirs,
    new_task_id,
    read_process_record,
    read_result,
    read_spec,
    read_state,
    runs_lock,
    task_dir,
    update_state,
    utc_now,
    write_model,
)
from .spawn import (
    record_process,
    settle_leftovers,
    spawn_worker,
    terminate_task,
    worker_alive,
)

MAX_LIVE_TASKS = 2
"""At most this many run directories may have a live worker."""

STARTUP_GRACE = timedelta(seconds=60)
"""How long a task may exist without ``process.json`` before it reads as dead.

Under the runs lock a reader never sees that window on a healthy server; it
only remains when the server died between creating the directory and
spawning, and then the task must not hold a capacity slot forever.
"""


class CapacityExceededError(RuntimeError):
    def __init__(self, live: int, limit: int) -> None:
        super().__init__(
            f"{live} delegated task(s) already running; the limit is {limit}"
        )
        self.live = live
        self.limit = limit


class WorkerSpawnError(RuntimeError):
    """The run directory exists but the worker process could not be started."""


class ArgumentError(ValueError):
    """A request argument that is well-formed but unusable; names the argument."""

    def __init__(self, argument: str, message: str) -> None:
        super().__init__(f"{argument}: {message}")
        self.argument = argument
        self.message = message


class RequestConflictError(RuntimeError):
    """``request_id`` already started a task with a different request."""

    def __init__(self, request_id: str, task_id: str) -> None:
        super().__init__(
            f"request_id {request_id!r} already started task {task_id} with a "
            "different request"
        )
        self.request_id = request_id
        self.task_id = task_id


class SessionBusyError(RuntimeError):
    """The Claude session to resume is being written by a live task."""

    def __init__(self, session_id: str, task_id: str) -> None:
        super().__init__(
            f"Claude session {session_id} is in use by task {task_id}, which is "
            "still running; wait for it to end or cancel it"
        )
        self.session_id = session_id
        self.task_id = task_id


class ResumeUnavailableError(RuntimeError):
    """The task to resume cannot be resumed (no session, or a mismatch)."""

    def __init__(
        self, task_id: str, reason: str, *, latest_task_id: str | None = None
    ) -> None:
        super().__init__(f"task {task_id} cannot be resumed: {reason}")
        self.task_id = task_id
        self.reason = reason
        self.latest_task_id = latest_task_id


class RunDefaults(BaseModel):
    """The configured defaults a :class:`StartRequest` falls back to."""

    model_config = ConfigDict(frozen=True)

    model: str
    effort: EffortLevel | None = None
    max_turns: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    max_budget_usd: float = Field(gt=0)

    @classmethod
    def from_config(cls, config: Config) -> RunDefaults:
        return cls(
            model=config.claude_model.value,
            effort=config.claude_effort.value,
            max_turns=config.claude_max_turns.value,
            timeout_seconds=config.claude_timeout_seconds.value,
            max_budget_usd=config.claude_max_budget_usd.value,
        )


class StartRequest(BaseModel):
    """A validated ``claude_start`` call.

    Validation failures surface as Pydantic errors naming the field; the tool
    maps them to ``invalid_argument``. ``working_directory`` must be absolute
    and exist; limits must be positive; there are no upper caps.

    ``resume_task_id`` continues that task's Claude conversation in a new
    task. Nothing is inherited but the conversation: tools and limits are
    this request's own (defaults: no tools, the configured limits), and
    ``working_directory`` must be the parent's (it is filled in when
    omitted, which is why it is optional here).

    ``request_id`` makes the start idempotent: the same ID with the same
    request returns the task it started; with a different one, it fails.

    ``allow_questions`` lets the agent ask its parent (``ask_parent``); a
    caller that will not watch for ``input_required`` leaves it off, so an
    unattended task can never stall on a question nobody answers.

    ``tools`` names the built-in tools to expose; any well-formed name is
    accepted, because the runtime, not this server, knows what exists (a
    task whose session lacks one ends ``tools_unavailable``).
    ``allowed_tools`` is the permission rules that auto-approve calls, each
    for a tool in ``tools``: ``["Bash(git status)"]`` approves that command
    and no other ``Bash`` call. Omitted, every exposed tool is approved
    outright, as before the two were separate.
    """

    model_config = ConfigDict(frozen=True)

    brief: str = Field(min_length=1)
    working_directory: Path | None = None
    tools: tuple[ToolName, ...] = ()
    allowed_tools: tuple[PermissionRule, ...] | None = None
    model: str | None = Field(default=None, min_length=1)
    effort: EffortLevel | None = None
    max_turns: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_budget_usd: float | None = Field(default=None, gt=0)
    resume_task_id: str | None = None
    request_id: str | None = None
    allow_questions: bool = False

    @field_validator("brief")
    @classmethod
    def _brief_has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("brief must not be blank")
        return value

    @field_validator("working_directory")
    @classmethod
    def _working_directory_is_an_existing_absolute_dir(
        cls, value: Path | None
    ) -> Path | None:
        if value is None:
            return None
        if not value.is_absolute():
            raise ValueError("working_directory must be an absolute path")
        if not value.is_dir():
            raise ValueError("working_directory must be an existing directory")
        return value

    @field_validator("tools")
    @classmethod
    def _tools_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value))

    @field_validator("allowed_tools")
    @classmethod
    def _rules_name_exposed_tools(
        cls, value: tuple[str, ...] | None, info: ValidationInfo
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        exposed = set(info.data.get("tools") or ())
        for rule in value:
            tool = permission_rule_tool(rule)
            if tool not in exposed:
                raise ValueError(
                    f"{rule!r} approves {tool}, which is not in tools; an "
                    "approval cannot expose a tool, so add it to tools too"
                )
        return tuple(dict.fromkeys(value))

    @field_validator("request_id")
    @classmethod
    def _request_id_shape(cls, value: str | None) -> str | None:
        if value is not None and not CALLER_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "request_id must be 1-128 characters of letters, digits, '.', '_', ':' or '-'"
            )
        return value

    def fingerprint(self) -> str:
        """A digest of the request as the caller sent it, for retry matching.

        Defaults are not resolved first, so a configuration change between a
        call and its retry does not turn the retry into a conflict.
        """
        exclude = {"request_id"}
        if self.allowed_tools is None:
            # Absent from requests made before it existed: keep their digest.
            exclude.add("allowed_tools")
        payload = self.model_dump(mode="json", exclude=exclude)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_spec(
        self,
        task_id: str,
        defaults: RunDefaults,
        *,
        working_directory: Path,
        session_id: str,
        cost_baseline_usd: float | None = None,
        cost_baseline_exact: bool = True,
    ) -> RunSpec:
        return RunSpec(
            task_id=task_id,
            created_at=utc_now(),
            working_directory=working_directory,
            tools=self.tools,
            allowed_tools=self.allowed_tools,
            model=self.model or defaults.model,
            effort=self.effort if self.effort is not None else defaults.effort,
            max_turns=self.max_turns or defaults.max_turns,
            timeout_seconds=self.timeout_seconds or defaults.timeout_seconds,
            max_budget_usd=self.max_budget_usd or defaults.max_budget_usd,
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            resume_task_id=self.resume_task_id,
            request_id=self.request_id,
            request_fingerprint=self.fingerprint() if self.request_id else None,
            cost_baseline_usd=cost_baseline_usd,
            cost_baseline_exact=cost_baseline_exact,
            allow_questions=self.allow_questions,
        )


def start_task(
    runs_dir: Path, request: StartRequest, defaults: RunDefaults
) -> ClaudeStartResult:
    """Create the run directory, spawn the worker, return once it is durable.

    With a ``request_id`` already on record, returns that task
    (``replayed``) or raises :class:`RequestConflictError`, before any
    capacity check: a retry must find its task even when the slots are full.
    Raises :class:`~.rundir.UnknownTaskError` for an unknown
    ``resume_task_id``, :class:`ResumeUnavailableError` when it has no
    session or the working directory differs, :class:`SessionBusyError`
    when it is still running or another live task writes its session,
    :class:`CapacityExceededError` before anything is created when
    :data:`MAX_LIVE_TASKS` tasks already have a live worker, and
    :class:`WorkerSpawnError` (after recording ``failed`` / ``worker_died``
    in the new directory) when the worker cannot start.
    """
    ensure_runs_dir(runs_dir)
    with runs_lock(runs_dir):
        if request.request_id is not None:
            replay = _replay_locked(runs_dir, request)
            if replay is not None:
                return replay
        live = _live_tasks_locked(runs_dir)
        working_directory, session_id, baseline, exact = _session_for_locked(
            runs_dir, request, live
        )
        if len(live) >= MAX_LIVE_TASKS:
            raise CapacityExceededError(len(live), MAX_LIVE_TASKS)

        spec = request.to_spec(
            new_task_id(),
            defaults,
            working_directory=working_directory,
            session_id=session_id,
            cost_baseline_usd=baseline,
            cost_baseline_exact=exact,
        )
        run_dir = create_run_dir(runs_dir, spec, request.brief)
        update_state(
            run_dir,
            lambda state: state.model_copy(
                update={
                    "session_id": spec.session_id,
                    "resumed_from": spec.resume_task_id,
                }
            ),
        )  # status.json exists at once
        if request.request_id is not None:
            _record_request_locked(runs_dir, request, spec.task_id)
        try:
            process = spawn_worker(run_dir)
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            update_state(
                run_dir,
                _finish(
                    "failed",
                    "worker_died",
                    error=f"the worker process could not be started: {reason}",
                ),
            )
            Inbox(run_dir).close("the worker process could not be started")
            raise WorkerSpawnError(f"could not start the worker: {reason}") from exc

        record = record_process(run_dir, process.pid)
        update_state(
            run_dir,
            lambda state: state.model_copy(
                update={"pid": record.pid, "pgid": record.pgid}
            ),
        )
    return ClaudeStartResult(
        task_id=spec.task_id,
        run_dir=run_dir,
        session_id=spec.session_id,
        resumed_from=spec.resume_task_id,
        request_id=spec.request_id,
    )


def _request_path(runs_dir: Path, request_id: str) -> Path:
    digest = hashlib.sha256(request_id.encode()).hexdigest()[:32]
    return runs_dir / REQUESTS_DIR / f"{digest}.json"


def _replay_locked(runs_dir: Path, request: StartRequest) -> ClaudeStartResult | None:
    assert request.request_id is not None
    path = _request_path(runs_dir, request.request_id)
    if not path.exists():
        return None
    record = RequestRecord.model_validate_json(path.read_bytes())
    if record.request_id != request.request_id:  # pragma: no cover - digest clash
        return None
    if record.fingerprint != request.fingerprint():
        raise RequestConflictError(request.request_id, record.task_id)
    run_dir = task_dir(runs_dir, record.task_id)
    state = _reconcile_locked(run_dir)
    return ClaudeStartResult(
        task_id=state.task_id,
        status=state.status,
        run_dir=run_dir,
        session_id=state.session_id,
        resumed_from=state.resumed_from,
        request_id=request.request_id,
        replayed=True,
    )


def _record_request_locked(runs_dir: Path, request: StartRequest, task_id: str) -> None:
    assert request.request_id is not None
    path = _request_path(runs_dir, request.request_id)
    path.parent.mkdir(mode=DIR_MODE, exist_ok=True)
    write_model(
        path,
        RequestRecord(
            request_id=request.request_id,
            fingerprint=request.fingerprint(),
            task_id=task_id,
        ),
    )


def _session_for_locked(
    runs_dir: Path, request: StartRequest, live: list[RunState]
) -> tuple[Path, str, float | None, bool]:
    """The working directory, Claude session and cost baseline for a start.

    A fresh task gets a new session ID. A resume takes its parent's, after
    checking the parent ended, the session is not being written by another
    live task, the parent is the session's latest writer (resuming an older
    task would silently continue from a later task's state), and the
    working directory matches (Claude keys its sessions
    by directory, and a resume elsewhere would not see the same files).
    """
    if request.resume_task_id is None:
        if request.working_directory is None:
            raise ArgumentError(
                "working_directory", "required unless resume_task_id is given"
            )
        return request.working_directory, str(uuid.uuid4()), None, True
    parent_dir = task_dir(runs_dir, request.resume_task_id)
    parent = _reconcile_locked(parent_dir)
    if not parent.is_terminal:
        raise SessionBusyError(parent.session_id or "?", parent.task_id)
    parent_spec = read_spec(parent_dir)
    session_id = (
        parent.session_id or parent_spec.session_id or _legacy_session(parent_dir)
    )
    if session_id is None:
        raise ResumeUnavailableError(
            parent.task_id, "it recorded no Claude session (it never reached Claude)"
        )
    if parent.termination_reason == "resume_failed":
        raise ResumeUnavailableError(
            parent.task_id, "it was itself a resume that failed; resume its parent"
        )
    working_directory = request.working_directory or parent_spec.working_directory
    if working_directory.resolve() != parent_spec.working_directory.resolve():
        raise ResumeUnavailableError(
            parent.task_id,
            f"working_directory must be the parent's ({parent_spec.working_directory})",
        )
    for other in live:
        if other.session_id == session_id:
            raise SessionBusyError(session_id, other.task_id)
    head = _session_head_locked(runs_dir, session_id, parent)
    if head != parent.task_id:
        raise ResumeUnavailableError(
            parent.task_id,
            f"task {head} has continued this Claude session since; resume that "
            "task (the latest) instead",
            latest_task_id=head,
        )
    return working_directory, session_id, *_cost_baseline(parent_dir, parent)


def _session_head_locked(runs_dir: Path, session_id: str, parent: RunState) -> str:
    """The latest task that wrote ``session_id``: the only valid resume handle.

    "Latest" is by recorded ``created_at``: task IDs sort by creation only
    to the second, and two tasks in one second order by their random part.
    A failed resume never reached the session, so it does not count.
    """
    head, head_created = parent.task_id, parent.created_at
    for run_dir in list_task_dirs(runs_dir):
        if run_dir.name == parent.task_id:
            continue
        try:
            spec = read_spec(run_dir)
        except OSError, ValueError:
            continue
        if spec.session_id != session_id or spec.created_at <= head_created:
            continue
        if read_state(run_dir).termination_reason == "resume_failed":
            continue
        head, head_created = run_dir.name, spec.created_at
    return head


def _cost_baseline(parent_dir: Path, parent: RunState) -> tuple[float | None, bool]:
    """The session's running cost where the parent left it, and whether exact.

    A parent recorded by this version carries ``session_cost_usd``; an older
    one recorded the last result's own cost, which was the session total
    then. Only a parent that ended at a result (``completed``) is exact: a
    limit, error or cancel mid-turn may have spent more than any result
    reported, so the resumed invocation's cost is then an overestimate.
    """
    total = parent.session_cost_usd
    if total is None:
        result = read_result(parent_dir)
        usage = result.usage if result is not None else parent.usage
        total = usage.estimated_cost_usd if usage is not None else None
    return total, total is not None and parent.status == "completed"


def _legacy_session(run_dir: Path, *, max_lines: int = 200) -> str | None:
    """The session ID a pre-``schema_version`` 2 run's transcript recorded.

    Those runs did not store it; their first SDK messages carry it. Reads
    at most ``max_lines`` lines.
    """
    path = run_dir / TRANSCRIPT_FILE
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= max_lines:
                break
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            session = message.get("session_id") or (message.get("data") or {}).get(
                "session_id"
            )
            if isinstance(session, str) and session:
                return session
    return None


def task_status(runs_dir: Path, task_id: str) -> RunState:
    """``status.json`` for ``task_id``, reconciled against the process table.

    Raises :class:`~.rundir.UnknownTaskError` for a malformed or unknown ID.
    """
    return reconcile(task_dir(runs_dir, task_id))


def reconcile(run_dir: Path) -> RunState:
    """Settle a ``working`` task whose worker is gone as ``worker_died``.

    A terminal state is returned as is. A live worker is left alone. A dead
    worker with no terminal state never got to write one, so the server
    records the failure on its behalf, after killing whatever the task's
    tree left behind (``groups.json``). The write re-reads under the status
    lock, so a worker that finished between the two reads still wins.
    """
    with runs_lock(run_dir.parent):
        return _reconcile_locked(run_dir)


def live_tasks(runs_dir: Path) -> list[RunState]:
    """Every task that is ``working`` with a live worker; dead ones get settled."""
    if not runs_dir.is_dir():
        return []
    with runs_lock(runs_dir):
        return _live_tasks_locked(runs_dir)


def _live_tasks_locked(runs_dir: Path) -> list[RunState]:
    table = process_table()
    live: list[RunState] = []
    for run_dir in list_task_dirs(runs_dir):
        state = _reconcile_locked(run_dir, table)
        if not state.is_terminal:
            live.append(state)
    return live


def _reconcile_locked(run_dir: Path, table: dict | None = None) -> RunState:
    state = read_state(run_dir)
    if state.is_terminal:
        return state
    record = read_process_record(run_dir)
    if record is None:
        # Created but not yet spawned. Under the runs lock that is only ever
        # a server that died mid-start; give it the grace, then settle it.
        if utc_now() - state.created_at < STARTUP_GRACE:
            return state
    elif worker_alive(run_dir, record, table if table is not None else process_table()):
        return state
    settle_leftovers(run_dir)
    Inbox(run_dir).close("the worker process exited without recording a result")
    settled = update_state(
        run_dir,
        _finish(
            "failed",
            "worker_died",
            error="the worker process exited without recording a result",
        ),
    )
    if settled.termination_reason == "worker_died":
        _note_finished(run_dir, settled)
    return settled


async def cancel_task(runs_dir: Path, task_id: str) -> RunState:
    """Stop a task's processes and record the outcome.

    A terminal task is returned unchanged. Otherwise the worker's group and
    every group recorded in ``groups.json`` get SIGTERM, up to ten seconds,
    SIGKILL, up to five seconds; then ``cancelled`` is recorded, or
    ``failed`` if something still could not be confirmed dead. A terminal
    state the worker wrote itself in the meantime wins.
    """
    run_dir = task_dir(runs_dir, task_id)
    state = reconcile(run_dir)
    if state.is_terminal:
        return state
    update_state(
        run_dir,
        lambda current: (
            current
            if current.is_terminal or current.cancel_requested_at
            else current.model_copy(update={"cancel_requested_at": utc_now()})
        ),
    )
    if read_spec(run_dir).schema_version >= SCHEMA_VERSION:
        EventLog(run_dir).append("cancel_requested")
    dead = await terminate_task(run_dir)
    Inbox(run_dir).close("the task was cancelled")
    if dead:
        transition = _finish("cancelled", "cancelled")
    else:
        transition = _finish(
            "failed",
            "cancelled",
            error="the task's processes are still alive after SIGKILL",
        )
    before = read_state(run_dir)
    settled = update_state(run_dir, transition)
    if not before.is_terminal:
        _note_finished(run_dir, settled)
    return settled


def _finish(
    status: RunStatus, reason: TerminationReason, *, error: str | None = None
) -> Callable[[RunState], RunState]:
    """A transition to a terminal state that never overwrites one."""

    def transition(state: RunState) -> RunState:
        if state.is_terminal:
            return state
        message = (
            "Cancelled" if status == "cancelled" else f"Failed ({reason}): {error}"
        )
        return state.model_copy(
            update={
                "status": status,
                "termination_reason": reason,
                "finished_at": utc_now(),
                "error": error,
                "status_message": message,
            }
        )

    return transition


def _note_finished(run_dir: Path, state: RunState) -> None:
    """The ``finished`` event for an end the server recorded itself.

    Runs written before events existed have no log to append to.
    """
    if read_spec(run_dir).schema_version < SCHEMA_VERSION:
        return
    EventLog(run_dir).append(
        "finished",
        text=state.status_message,
        detail={"status": state.status, "termination_reason": state.termination_reason},
    )
