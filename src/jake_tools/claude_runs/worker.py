"""The worker: the only place a delegated task actually runs.

``python -m jake_tools.mcp worker RUN_DIR`` calls :func:`main`. The worker
reads ``spec.json`` and ``brief.md``, runs one :class:`ClaudeAgent` call,
streams every SDK message to ``transcript.jsonl``, records telemetry to
``telemetry.json``, and writes ``result.json`` plus the final
``status.json`` on the way out. Every transition goes through
:func:`~.rundir.update_state`.

Isolation: the agent gets exactly the granted built-ins as ``tools`` (an
empty grant is genuinely tool-less), the same list as ``allowed_tools`` so
they run without a prompt, ``permission_mode="dontAsk"`` so anything not
pre-approved is denied rather than waited on, ``setting_sources=()`` so no
settings file is read, and (via ``AgentSpec.to_options``)
``strict_mcp_config`` so no user MCP server is inherited.

Termination: the timeout is an anyio ``move_on_after`` around the call, and
SIGTERM cancels the same scope. Both are anyio cancellations, which is what
the SDK's transport needs to run its stdin-EOF / SIGTERM / SIGKILL
escalation on the CLI child; a raw asyncio cancellation would orphan it
(see the SDK's ``subprocess_cli.py`` ``close()`` docstring).

Processes: the CLI runs each tool shell in its own session, outside the
worker's process group, so the worker tracks every group in its descendant
tree into ``groups.json`` while it runs (and, on Linux, is a child subreaper
so orphans of a killed CLI stay in that tree). Before recording a terminal
state it kills its live descendants and every recorded group, so "the task
ended" also means "its processes are gone". If the worker itself is killed
first, the server does the same from ``groups.json`` when it settles the
task (see :mod:`.spawn`).

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
from dataclasses import dataclass
from pathlib import Path

import anyio

from ..ai_usage import AICallTelemetry, Usage
from ..claude import AgentSpec, ClaudeAgent, ClaudeAgentError, Message, message_to_json
from .models import (
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
            update={"started_at": utc_now(), "pid": pid, "pgid": pgid}
        ),
    )
    if read_process_record(run_dir) is None:
        record_process(run_dir, pid)  # the server died before it could
    log.info("task %s started (pid %d, pgid %d)", spec.task_id, pid, pgid)

    transcript = TranscriptWriter(run_dir / TRANSCRIPT_FILE)
    telemetry = TelemetryFile(run_dir / TELEMETRY_FILE)
    agent = (
        ClaudeAgent(defaults=agent_spec(spec))
        .with_telemetry(telemetry)
        .with_message_observer(transcript.write)
        .for_stage(STAGE)
    )

    outcome: Outcome | None = None
    try:
        with anyio.CancelScope() as run_scope:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_cancel_on_signal, run_scope)
                tg.start_soon(track_descendant_groups, run_dir)
                outcome = await _run_agent(agent, brief, spec.timeout_seconds)
                tg.cancel_scope.cancel()
        if run_scope.cancel_called:
            outcome = Outcome(status="cancelled", reason="cancelled")
        if outcome is None:  # pragma: no cover - defensive; the scope says why
            outcome = Outcome(
                status="failed",
                reason="sdk_error",
                error="worker ended without an outcome",
            )
    except BaseException as exc:
        outcome = Outcome(status="failed", reason="sdk_error", error=_describe(exc))
        _finish(run_dir, spec, outcome, transcript)
        raise
    _finish(run_dir, spec, outcome, transcript)
    return outcome


def _finish(
    run_dir: Path, spec: RunSpec, outcome: Outcome, transcript: TranscriptWriter
) -> None:
    """Leave nothing behind: sweep the task's processes, then record the outcome.

    The sweep comes first so that a terminal ``status.json`` also means the
    task's processes are gone (bar this one, which exits next).
    """
    transcript.close()
    swept = sweep_own_group(run_dir)
    if swept:
        log.warning("task %s: killed %d leftover process(es)", spec.task_id, swept)
    _record(run_dir, spec, outcome)
    log.info("task %s %s/%s", spec.task_id, outcome.status, outcome.reason)


def agent_spec(spec: RunSpec) -> AgentSpec:
    return AgentSpec(
        model=spec.model,
        tools=spec.tools,
        effort=spec.effort,
        max_turns=spec.max_turns,
        max_budget_usd=spec.max_budget_usd,
        cwd=spec.working_directory,
        setting_sources=(),
        permission_mode="dontAsk",
    )


async def _run_agent(agent: ClaudeAgent, brief: str, timeout_seconds: float) -> Outcome:
    with anyio.move_on_after(timeout_seconds) as timeout_scope:
        try:
            reply = await agent.run(brief)
        except ClaudeAgentError as exc:
            reason = _SUBTYPE_REASONS.get(exc.result_subtype or "", "sdk_error")
            return Outcome(
                status="failed",
                reason=reason,
                final_text=exc.final_text,
                usage=exc.usage if exc.usage.has_activity else None,
                error=_describe(exc),
            )
        except Exception as exc:
            return Outcome(status="failed", reason="sdk_error", error=_describe(exc))
        return Outcome(
            status="completed",
            reason="finished",
            final_text=reply.final_text,
            usage=reply.usage,
        )
    # Only reached when the deadline cancelled the scope: ``cancelled_caught``
    # distinguishes our timeout from any TimeoutError the SDK raises itself.
    assert timeout_scope.cancelled_caught
    return Outcome(
        status="failed",
        reason="timeout",
        error=f"the task exceeded its {timeout_seconds:g}s timeout",
    )


async def _cancel_on_signal(scope: anyio.CancelScope) -> None:
    with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
        async for signum in signals:
            log.info("received %s; cancelling the run", signal.Signals(signum).name)
            scope.cancel()
            return


def _record(run_dir: Path, spec: RunSpec, outcome: Outcome) -> None:
    finished_at = utc_now()
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
        ),
    )

    def transition(state: RunState) -> RunState:
        return state.model_copy(
            update={
                "status": outcome.status,
                "termination_reason": outcome.reason,
                "finished_at": finished_at,
                "final_text": outcome.final_text,
                "usage": outcome.usage,
                "error": outcome.error,
            }
        )

    update_state(run_dir, transition)


def _describe(exc: BaseException) -> str:
    """An error string for ``status.json``: type and message, bounded.

    The SDK's messages describe the CLI's exit or result subtype; they do
    not carry credentials. The bound keeps a runaway stderr dump out of the
    status file, whose readers want a sentence.
    """
    text = str(exc).strip()
    described = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    if len(described) > MAX_ERROR_CHARS:
        described = described[: MAX_ERROR_CHARS - 1] + "…"
    return described


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
