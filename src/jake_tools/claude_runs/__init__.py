"""Delegated Claude tasks: run directories, spawn, status, cancel.

A task lives entirely in ``<runs_dir>/<task_id>/``; the server holds no
state that matters, so it can be recycled by Hermes at any time. The worker
(``python -m jake_tools.mcp worker``) is a separate process in its own
session/process group: it survives server death and can be killed as a
group on cancel. See :mod:`jake_tools.claude_runs.models` for the files.
"""

from __future__ import annotations

from .models import (
    BUILTIN_TOOLS,
    BuiltinTool,
    ClaudeStartResult,
    RunResult,
    RunSpec,
    RunState,
    RunStatus,
    TerminationReason,
)

__all__ = [
    "BUILTIN_TOOLS",
    "BuiltinTool",
    "ClaudeStartResult",
    "RunResult",
    "RunSpec",
    "RunState",
    "RunStatus",
    "TerminationReason",
]
