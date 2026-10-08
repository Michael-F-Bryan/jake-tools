"""Delegated Claude tasks: run directories, spawn, status, cancel.

A task lives entirely in ``<runs_dir>/<task_id>/``; the server holds no
state that matters, so it can be recycled by Hermes at any time. The worker
(``python -m jake_tools.mcp worker``) is a separate process in its own
session/process group: it survives server death and can be killed as a
group on cancel. See :mod:`jake_tools.claude_runs.models` for the files,
:mod:`.rundir` for how they are written, :mod:`.spawn` for the process
primitives, :mod:`.control` for start/status/cancel, and :mod:`.worker` for
what runs inside the task.
"""

from __future__ import annotations

from .control import (
    MAX_LIVE_TASKS,
    CapacityExceededError,
    RunDefaults,
    StartRequest,
    WorkerSpawnError,
    cancel_task,
    start_task,
    task_status,
)
from .models import (
    BUILTIN_TOOLS,
    BuiltinTool,
    ClaudeStartResult,
    RunResult,
    RunSpec,
    RunState,
    RunStatus,
    RunTelemetry,
    TerminationReason,
)
from .rundir import UnknownTaskError

__all__ = [
    "BUILTIN_TOOLS",
    "MAX_LIVE_TASKS",
    "BuiltinTool",
    "CapacityExceededError",
    "ClaudeStartResult",
    "RunDefaults",
    "RunResult",
    "RunSpec",
    "RunState",
    "RunStatus",
    "RunTelemetry",
    "StartRequest",
    "TerminationReason",
    "UnknownTaskError",
    "WorkerSpawnError",
    "cancel_task",
    "start_task",
    "task_status",
]
