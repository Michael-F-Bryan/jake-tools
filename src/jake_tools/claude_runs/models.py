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
    The ``ClaudeAgent`` telemetry sink's output.
``worker.stdout`` / ``worker.stderr``
    The worker process's own streams.

Timestamps are RFC 3339 in UTC. Task IDs are ``<UTC basic timestamp>-<8 hex>``
(``20261008T031500Z-1a2b3c4d``) so directory listings sort by creation.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from ..ai_usage import Usage
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
