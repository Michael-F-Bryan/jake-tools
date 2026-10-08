"""The on-disk and on-the-wire contracts for one delegated task.

Run directory layout (``<runs_dir>/<task_id>/``, directory mode ``0700``,
files ``0600``, every in-place update written to a sibling temp file and
renamed):

``brief.md``
    The task text, verbatim.
``spec.json``
    :class:`RunSpec`: everything the worker needs that is not the brief.
``status.json``
    :class:`RunState`: rewritten atomically at every transition. The
    ``claude_status`` and ``claude_cancel`` tools return exactly this model.
``result.json``
    :class:`RunResult`: written once, when the worker exits.
``transcript.jsonl``
    One :func:`jake_tools.claude.message_to_json` object per SDK message.
``telemetry.json``
    :class:`RunTelemetry`: the ``ClaudeAgent`` telemetry sink's output.
``worker.stdout`` / ``worker.stderr``
    The worker process's own streams.
``process.json``
    :class:`ProcessRecord`: the worker's PID, process group and start-time
    token, written by the server right after the spawn (and by the worker
    if it finds the file missing). Nothing signals that group unless its
    leader's start time still matches, so a reused PID is never hit.
``groups.json``
    :class:`GroupRecords`: every other process group the worker has seen in
    its descendant tree, with the group leader's start-time token. The CLI
    starts each tool shell in its own session, so these are what a cancel
    or a sweep has to reach beyond the worker's own group.
``.status.lock``
    The lock every ``status.json`` read-modify-write takes.
``.worker.lock``
    Held exclusively by the worker for its whole life; "is the worker
    alive" is "is this lock held", which is true for a running worker and
    false for an exited one even when it lingers as a zombie.

``<runs_dir>/.runs.lock`` serialises capacity check, creation and spawn
across every server sharing the directory.

Timestamps are RFC 3339 in UTC. Task IDs are ``<UTC basic timestamp>-<8 hex>``
(``20261008T031500Z-1a2b3c4d``) so directory listings sort by creation.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from ..ai_usage import AICallTelemetry, Usage
from ..claude import EffortLevel

RunStatus = Literal["working", "completed", "failed", "cancelled"]

TerminationReason = Literal[
    "finished",
    "max_turns",
    "timeout",
    "budget",
    "cancelled",
    "worker_died",
    "sdk_error",
]

BuiltinTool = Literal[
    "Read", "Glob", "Grep", "WebSearch", "WebFetch", "Edit", "Write", "Bash"
]
BUILTIN_TOOLS: tuple[str, ...] = get_args(BuiltinTool)

TASK_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")


class RunSpec(BaseModel):
    """``spec.json``: the validated ``claude_start`` request plus identity."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    created_at: datetime
    working_directory: Path
    tools: tuple[BuiltinTool, ...] = ()
    model: str
    effort: EffortLevel | None = None
    max_turns: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    max_budget_usd: float = Field(gt=0)


class RunState(BaseModel):
    """``status.json``, and the result of ``claude_status`` / ``claude_cancel``.

    ``completed`` means the SDK invocation ended normally; it is not
    verification of the worker's claims. ``pgid`` is the worker's process
    group, recorded so a later server can check liveness and cancel by
    group. ``final_text`` is the SDK result message's final text, not the
    concatenated per-turn narration.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    status: RunStatus
    run_dir: Path
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    pid: int | None = None
    pgid: int | None = None
    termination_reason: TerminationReason | None = None
    final_text: str | None = None
    usage: Usage | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status != "working"


class RunResult(BaseModel):
    """``result.json``: the outcome alone, written once when the worker exits."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    status: RunStatus
    termination_reason: TerminationReason
    finished_at: datetime
    final_text: str | None = None
    usage: Usage | None = None
    error: str | None = None


class ClaudeStartResult(BaseModel):
    """What ``claude_start`` returns immediately after spawning the worker."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    status: Literal["working"] = "working"
    run_dir: Path


class ProcessRecord(BaseModel):
    """``process.json``: which process is the worker, pinned by start time.

    ``started`` is an opaque token from the process table (``/proc`` start
    ticks on Linux, ``ps lstart`` elsewhere); ``None`` means the process had
    already gone when it was looked up, which reads as dead.
    """

    model_config = ConfigDict(frozen=True)

    pid: int
    pgid: int
    started: str | None


class GroupRecord(BaseModel):
    """One process group a task's tree has used, pinned by its leader's start."""

    model_config = ConfigDict(frozen=True)

    pgid: int
    leader_started: str | None


class GroupRecords(BaseModel):
    """``groups.json``: append-only set of :class:`GroupRecord`."""

    groups: list[GroupRecord] = Field(default_factory=list)


class RunTelemetry(BaseModel):
    """``telemetry.json``: every call the worker's agent recorded, plus totals."""

    calls: list[AICallTelemetry] = Field(default_factory=list)

    @property
    def totals(self) -> Usage:
        usage = Usage()
        for call in self.calls:
            usage = usage + call.as_usage()
        return usage
