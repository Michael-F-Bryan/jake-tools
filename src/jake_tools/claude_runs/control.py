"""Start, status, cancel and capacity for delegated tasks.

These are the operations the ``claude_*`` MCP tools expose, minus the
transport. Each one works from disk plus the process table, never from
server memory, so a freshly started server answers for tasks an earlier
one began.

:func:`start_task` is deliberately synchronous: the capacity check, the
directory creation and the spawn run without a suspension point, so two
``claude_start`` calls on one server cannot both pass the check. It takes a
few milliseconds. :func:`cancel_task` is async because it waits on a process
group, and every wait is an anyio sleep.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..claude import EffortLevel
from ..config import Config
from .models import (
    BuiltinTool,
    ClaudeStartResult,
    RunSpec,
    RunState,
    RunStatus,
    TerminationReason,
)
from .rundir import (
    create_run_dir,
    list_task_dirs,
    new_task_id,
    read_state,
    task_dir,
    update_state,
    utc_now,
)
from .spawn import group_alive, spawn_worker, terminate_group

MAX_LIVE_TASKS = 2
"""At most this many run directories may have a live process group."""


class CapacityExceededError(RuntimeError):
    def __init__(self, live: int, limit: int) -> None:
        super().__init__(
            f"{live} delegated task(s) already running; the limit is {limit}"
        )
        self.live = live
        self.limit = limit


class WorkerSpawnError(RuntimeError):
    """The run directory exists but the worker process could not be started."""


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
    """

    model_config = ConfigDict(frozen=True)

    brief: str = Field(min_length=1)
    working_directory: Path
    tools: tuple[BuiltinTool, ...] = ()
    model: str | None = Field(default=None, min_length=1)
    effort: EffortLevel | None = None
    max_turns: int | None = Field(default=None, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_budget_usd: float | None = Field(default=None, gt=0)

    @field_validator("brief")
    @classmethod
    def _brief_has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("brief must not be blank")
        return value

    @field_validator("working_directory")
    @classmethod
    def _working_directory_is_an_existing_absolute_dir(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("working_directory must be an absolute path")
        if not value.is_dir():
            raise ValueError("working_directory must be an existing directory")
        return value

    @field_validator("tools")
    @classmethod
    def _tools_are_unique(
        cls, value: tuple[BuiltinTool, ...]
    ) -> tuple[BuiltinTool, ...]:
        seen: dict[BuiltinTool, None] = dict.fromkeys(value)
        return tuple(seen)

    def to_spec(self, task_id: str, defaults: RunDefaults) -> RunSpec:
        return RunSpec(
            task_id=task_id,
            created_at=utc_now(),
            working_directory=self.working_directory,
            tools=self.tools,
            model=self.model or defaults.model,
            effort=self.effort if self.effort is not None else defaults.effort,
            max_turns=self.max_turns or defaults.max_turns,
            timeout_seconds=self.timeout_seconds or defaults.timeout_seconds,
            max_budget_usd=self.max_budget_usd or defaults.max_budget_usd,
        )


def start_task(
    runs_dir: Path, request: StartRequest, defaults: RunDefaults
) -> ClaudeStartResult:
    """Create the run directory, spawn the worker, return immediately.

    Raises :class:`CapacityExceededError` before anything is created when
    :data:`MAX_LIVE_TASKS` run directories already have a live process
    group, and :class:`WorkerSpawnError` (after recording ``failed`` /
    ``worker_died`` in the new directory) when the worker cannot start.
    """
    live = live_tasks(runs_dir)
    if len(live) >= MAX_LIVE_TASKS:
        raise CapacityExceededError(len(live), MAX_LIVE_TASKS)

    spec = request.to_spec(new_task_id(), defaults)
    run_dir = create_run_dir(runs_dir, spec, request.brief)
    update_state(run_dir, lambda state: state)  # status.json exists from the start
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
        raise WorkerSpawnError(f"could not start the worker: {reason}") from exc

    pid = process.pid
    update_state(
        run_dir, lambda state: state.model_copy(update={"pid": pid, "pgid": pid})
    )
    return ClaudeStartResult(task_id=spec.task_id, run_dir=run_dir)


def task_status(runs_dir: Path, task_id: str) -> RunState:
    """``status.json`` for ``task_id``, reconciled against the process table.

    Raises :class:`~.rundir.UnknownTaskError` for a malformed or unknown ID.
    """
    return reconcile(task_dir(runs_dir, task_id))


def reconcile(run_dir: Path) -> RunState:
    """Settle a ``working`` task whose process group is gone as ``worker_died``.

    A terminal state is returned as is. A live group is left alone. A dead
    group with no terminal state means the worker never got to write one,
    so the server records the failure on its behalf; the write re-reads under
    the lock, so a worker that finished between the two reads still wins.
    """
    state = read_state(run_dir)
    if state.is_terminal or group_alive(state.pgid):
        return state
    return update_state(
        run_dir,
        _finish(
            "failed",
            "worker_died",
            error="the worker process group exited without recording a result",
        ),
    )


def live_tasks(runs_dir: Path) -> list[RunState]:
    """Every task that is ``working`` with a live group; dead ones get settled."""
    live: list[RunState] = []
    for run_dir in list_task_dirs(runs_dir):
        state = reconcile(run_dir)
        if not state.is_terminal:
            live.append(state)
    return live


async def cancel_task(runs_dir: Path, task_id: str) -> RunState:
    """Stop a task's whole process group and record the outcome.

    A terminal task is returned unchanged. Otherwise the group gets SIGTERM,
    up to ten seconds, SIGKILL, up to five seconds; then ``cancelled`` is
    recorded, or ``failed`` if the group still could not be confirmed dead.
    A terminal state the worker wrote itself in the meantime wins.
    """
    run_dir = task_dir(runs_dir, task_id)
    state = reconcile(run_dir)
    if state.is_terminal or state.pgid is None:
        return state
    pgid = state.pgid
    dead = await terminate_group(pgid)
    if dead:
        transition = _finish("cancelled", "cancelled")
    else:
        transition = _finish(
            "failed",
            "cancelled",
            error=f"process group {pgid} is still alive after SIGKILL",
        )
    return update_state(run_dir, transition)


def _finish(
    status: RunStatus, reason: TerminationReason, *, error: str | None = None
) -> Callable[[RunState], RunState]:
    """A transition to a terminal state that never overwrites one."""

    def transition(state: RunState) -> RunState:
        if state.is_terminal:
            return state
        return state.model_copy(
            update={
                "status": status,
                "termination_reason": reason,
                "finished_at": utc_now(),
                "error": error,
            }
        )

    return transition
