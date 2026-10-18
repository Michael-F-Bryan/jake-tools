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
``final.md``
    The final answer (the last result message's text), in full, when there
    was one. ``result.json`` and ``status.json`` point at it.
``transcript.jsonl``
    One :func:`jake_tools.claude.message_to_json` object per SDK message:
    the raw evidence, a diagnostic escape hatch, not a contract.
``events.jsonl``
    One :class:`~.events.Event` per line, append-only, numbered from 1: the
    normalised public record (narration, tool activity, steering,
    clarification, results) that ``claude_poll`` and ``claude_transcript``
    page through by cursor. Never holds thinking. ``.events.lock``
    serialises appends from the worker and the server.
``messages/<message_id>.json``
    :class:`SteerMessage`: one steering message and its delivery state.
``inputs/<request_id>.json``
    :class:`InputRequest`: one clarification the worker asked for.
``.inbox.lock`` / ``inbox.closed``
    The lock every ``messages/`` and ``inputs/`` change takes, and the
    marker that the task accepts no more messages (the worker ended, or is
    ending, its conversation).
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
across every server sharing the directory. ``<runs_dir>/.requests/`` maps a
caller's ``request_id`` (by digest) to the task it started.

Lock order, outermost first: ``.runs.lock``, ``.inbox.lock``,
``.status.lock``, ``.events.lock``. Nothing takes an outer lock while
holding an inner one.

Timestamps are RFC 3339 in UTC. Task IDs are ``<UTC basic timestamp>-<32
hex>`` (128 random bits, so an ID cannot be guessed) and sort by creation;
runs created before that change have 8 hex digits and stay valid.

Compatibility: every field added since the first release is optional, so a
run directory written by an older server or worker still loads.
``spec.schema_version`` is 1 for those runs; they have no ``events.jsonl``,
and the event tools say so (``legacy_run``) instead of failing.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from ..ai_usage import AICallTelemetry, Usage
from ..claude import EffortLevel

RunStatus = Literal["working", "input_required", "completed", "failed", "cancelled"]
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

TerminationReason = Literal[
    "finished",
    "max_turns",
    "timeout",
    "budget",
    "cancelled",
    "worker_died",
    "sdk_error",
    "resume_failed",
]

SCHEMA_VERSION = 2
"""The run-directory schema new tasks are written with (see the module doc)."""

BuiltinTool = Literal[
    "Read", "Glob", "Grep", "WebSearch", "WebFetch", "Edit", "Write", "Bash"
]
BUILTIN_TOOLS: tuple[str, ...] = get_args(BuiltinTool)

TASK_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-(?:[0-9a-f]{32}|[0-9a-f]{8})$")

CALLER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
"""What a caller-chosen ``request_id`` or ``message_id`` may look like."""


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
    schema_version: int = 1
    session_id: str | None = None
    resume_task_id: str | None = None
    request_id: str | None = None
    request_fingerprint: str | None = None
    cost_baseline_usd: float | None = None
    """``session_id`` is the Claude session this invocation writes: a fresh
    UUID for a new conversation, the parent's for a resume (the CLI keeps it).
    ``cost_baseline_usd`` is the session's cumulative cost when this
    invocation began, because the CLI carries it across a resume."""


class RunState(BaseModel):
    """``status.json``, and the result of ``claude_status`` / ``claude_cancel``.

    ``completed`` means the SDK invocation ended normally; it is not
    verification of the worker's claims. ``pgid`` is the worker's process
    group, recorded so a later server can check liveness and cancel by
    group. ``final_text`` is the SDK result message's final text, not the
    concatenated per-turn narration; ``final_text_path`` holds it in full.

    ``status_message`` is the latest public progress line and
    ``last_activity_at`` when the agent last did something observable (a
    message, a tool call or result); neither is a heartbeat.
    ``cancel_requested_at`` is set when a cancel began, before anything was
    signalled; ``status`` says whether it finished. ``usage`` is known after
    each completed turn; ``invocation_cost_usd`` is this invocation's share
    of a resumed session's cumulative cost.
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
    session_id: str | None = None
    resumed_from: str | None = None
    status_message: str | None = None
    last_activity_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    final_text_path: Path | None = None
    invocation_cost_usd: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


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
    session_id: str | None = None
    final_text_path: Path | None = None
    invocation_cost_usd: float | None = None


class ClaudeStartResult(BaseModel):
    """What ``claude_start`` returns once the task is durably created.

    ``session_id`` is the Claude session the task writes (known up front: a
    new conversation is given its ID, a resume keeps its parent's).
    ``replayed`` is true when ``request_id`` matched an earlier start with
    the same request, and this is that task, not a new one; its ``status``
    is then whatever that task's status is now.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    status: RunStatus = "working"
    run_dir: Path
    session_id: str | None = None
    resumed_from: str | None = None
    request_id: str | None = None
    replayed: bool = False


DeliveryState = Literal["queued", "delivered", "in_conversation", "not_delivered"]


class SteerMessage(BaseModel):
    """``messages/<message_id>.json``: one steering message and where it got to.

    ``queued``: stored, not yet written to Claude. ``delivered``: written to
    the CLI's input stream. ``in_conversation``: the CLI echoed it back into
    the conversation (after the running tool call, or as a new turn).
    ``not_delivered``: the task ended first; ``reason`` says why. None of
    these is evidence the model acted on it. ``attempted_at`` is set just
    before the write, so a worker killed mid-write is reported as unknown
    rather than as never sent.
    """

    message_id: str
    uuid: str
    text: str
    fingerprint: str
    state: DeliveryState = "queued"
    submitted_at: datetime
    attempted_at: datetime | None = None
    delivered_at: datetime | None = None
    in_conversation_at: datetime | None = None
    reason: str | None = None


InputState = Literal["pending", "answered", "declined", "expired"]


class InputRequest(BaseModel):
    """``inputs/<request_id>.json``: a question the worker asked its parent.

    Created by the worker's ``ask_parent`` tool, which blocks until the
    request is ``answered`` or ``declined`` (by ``claude_update``) or the
    task ends (``expired``). ``request_id`` is unique for the task's life.
    ``choices`` are suggestions shown to the parent, not a constraint.
    """

    request_id: str
    question: str
    choices: list[str] = Field(default_factory=list)
    state: InputState = "pending"
    asked_at: datetime
    answer: str | None = None
    answered_at: datetime | None = None
    reason: str | None = None


class RequestRecord(BaseModel):
    """``<runs_dir>/.requests/<digest>.json``: which task a request_id started."""

    request_id: str
    fingerprint: str
    task_id: str


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
