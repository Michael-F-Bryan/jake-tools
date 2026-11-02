"""The interactive delegation workflow against the real CLI and a real model.

Through both surfaces: the ordinary ``claude_*`` tools (a legacy
``initialize`` session, as Hermes connects) and native MCP Tasks (a
2026-07-28 client declaring ``io.modelcontextprotocol/tasks``, from
``tasks_client.py``).

``slow`` (``uv run pytest --slow -k claude_tasks_live``). Nothing is
stubbed: the real stdio server, worker, SDK and ``claude`` CLI, with the
cheapest model and tight limits, in a scratch directory under ``tmp_path``.
Assertions are semantic invariants a correct run must satisfy (counts that
only appear if the model received the steering message or the answer, a
file only the resumed conversation could write correctly), never exact
prose. Set ``JAKE_TOOLS_EVIDENCE_DIR`` to keep every tool result as JSON.

Environment: the hermetic XDG layout from ``conftest.py`` with the
developer's real ``HOME``, where the CLI keeps its credentials.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import ClientSession
from tasks_client import (
    TERMINAL,
    CreateTaskResult,
    cancel_task,
    create_task,
    get_task,
    start_modern_client,
    update_task,
)

pytestmark = pytest.mark.slow

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]

CHEAP: dict[str, Any] = {
    "model": "claude-haiku-4-5",
    "effort": "low",
    "max_budget_usd": 0.5,
    "max_turns": 20,
    "timeout_seconds": 300,
}

INVENTORY = (
    "widget: blue, 3 units\n"
    "gadget: red, 1 unit\n"
    "widget: green, 5 units\n"
    "sprocket: grey, 9 units\n"
    "gadget: black, 2 units\n"
    "widget: white, 1 unit\n"
)
WIDGETS, GADGETS = 3, 2


class Evidence:
    """Every tool call and result, appended to a JSONL file when asked for."""

    def __init__(self, name: str) -> None:
        root = os.environ.get("JAKE_TOOLS_EVIDENCE_DIR")
        self.path = Path(root) / f"{name}.jsonl" if root else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def note(self, kind: str, payload: Any) -> None:
        if self.path is None:
            return
        with self.path.open("a") as handle:
            handle.write(
                json.dumps({"at": time.time(), "kind": kind, "payload": payload}) + "\n"
            )


@pytest.fixture
def live_env(
    mcp_env_factory: Callable[..., dict[str, str]], mcp_home: Path
) -> dict[str, str]:
    passthrough = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))
    }
    return mcp_env_factory(mcp_home, HOME=os.environ["HOME"], **passthrough)


async def call(
    session: ClientSession, evidence: Evidence, tool: str, **arguments: Any
) -> dict[str, Any]:
    result = await session.call_tool(tool, arguments)
    evidence.note(tool, {"arguments": arguments, "result": result.structured_content})
    assert result.is_error is False, result.structured_content
    assert result.structured_content is not None
    return result.structured_content


async def poll_until(
    session: ClientSession,
    evidence: Evidence,
    task_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    token: str | None = None,
    until: str = "attention",
    timeout: float = 360,
) -> dict[str, Any]:
    """Supervise as a parent would: poll with the latest token until
    ``predicate(snapshot)``. The poll wakes on what needs the caller."""
    deadline = time.monotonic() + timeout
    while True:
        args: dict[str, Any] = {"task_id": task_id, "wait_seconds": 60, "until": until}
        if token:
            args["token"] = token
        result = await call(session, evidence, "claude_poll", **args)
        token = result["observation_token"]
        if predicate(result):
            return result
        assert result["status"] in ("working", "input_required"), result
        assert time.monotonic() < deadline, result


async def history(
    session: ClientSession, evidence: Evidence, task_id: str
) -> list[dict[str, Any]]:
    """The task's whole event log, read with ``claude_transcript``."""
    events: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        args: dict[str, Any] = {"task_id": task_id, "limit": 200}
        if cursor:
            args["cursor"] = cursor
        page = await call(session, evidence, "claude_transcript", **args)
        events.extend(page["events"])
        cursor = page["next_cursor"]
        if not page["has_more"]:
            return events


def ended(result: dict[str, Any]) -> bool:
    return result["status"] in ("completed", "failed", "cancelled")


def kinds(events: list[dict[str, Any]]) -> list[str]:
    return [e["kind"] for e in events]


async def test_live_observe_steer_clarify_collect_then_resume_to_edit(
    mcp_server: ServerFactory, live_env: dict[str, str], tmp_path: Path
) -> None:
    evidence = Evidence("workflow")
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    (workdir / "inventory.txt").write_text(INVENTORY)
    brief = (
        "You are reviewing a scratch file. Do these steps in order:\n"
        "1. Run exactly `sleep 20 && wc -l inventory.txt` with the Bash tool, in "
        "the foreground, and wait for it.\n"
        "2. Read inventory.txt.\n"
        "3. Call the ask_parent tool exactly once to ask how the widget count "
        "should be formatted in your answer.\n"
        "4. Give a short final review of the file that follows the parent's "
        "answer and any later instructions from the parent.\n"
        "Do not modify any file."
    )
    async with mcp_server(live_env) as session:
        started = await call(
            session,
            evidence,
            "claude_start",
            brief=brief,
            working_directory=str(workdir),
            tools=["Read", "Bash"],
            allow_questions=True,
            request_id="live-review-1",
            **CHEAP,
        )
        task_id = started["task_id"]
        assert started["session_id"]

        # Observe: wait for checkpoints (until=progress) only to time the
        # steer: the Bash step shows up as the running tool, with progress.
        observed = await poll_until(
            session,
            evidence,
            task_id,
            lambda r: (r.get("running_tool") or {}).get("tool") == "Bash" or ended(r),
            token=started["observation_token"],
            until="progress",
        )
        assert observed["running_tool"]["tool"] == "Bash", observed
        assert observed["progress"]
        assert observed["last_activity_at"]

        # Steer while the 20 s command runs: a requirement only this message has.
        steer = await call(
            session,
            evidence,
            "claude_steer",
            task_id=task_id,
            message_id="correction-1",
            message=(
                "Correction from your parent: in your final answer also count the "
                "lines that mention 'gadget' and include them exactly as GADGETS=<n>."
            ),
        )
        assert steer["state"] in ("queued", "delivered")

        # Clarify: a default poll sleeps through routine work and wakes for
        # the question; the answer carries the widget format.
        waiting = await poll_until(
            session,
            evidence,
            task_id,
            lambda r: bool(r.get("pending_questions")) or ended(r),
            token=observed["observation_token"],
        )
        assert waiting["status"] == "input_required", waiting
        assert waiting["wake_reason"] == "question"
        request = waiting["pending_questions"][0]
        assert request["question"].strip()
        update = await call(
            session,
            evidence,
            "claude_update",
            task_id=task_id,
            responses={
                request["request_id"]: {
                    "answer": "Report the number of lines that mention 'widget' exactly as WIDGETS=<n>."
                }
            },
        )
        assert update["results"][0]["outcome"] == "accepted"

        final = await poll_until(
            session, evidence, task_id, ended, token=waiting["observation_token"]
        )
        info = await call(session, evidence, "claude_status", task_id=task_id)
        events = await history(session, evidence, task_id)

        # Resume the finished conversation as a new task with new grants.
        resumed = await call(
            session,
            evidence,
            "claude_start",
            brief=(
                "Using the counts from your review above (do not recount or read "
                "the file again), create summary.txt in the working directory "
                "containing exactly one line: WIDGETS=<n> GADGETS=<m>. Then reply DONE."
            ),
            resume_task_id=task_id,
            tools=["Write"],
            **CHEAP,
        )
        resumed_final = await poll_until(
            session,
            evidence,
            resumed["task_id"],
            ended,
            token=resumed["observation_token"],
        )
        resumed_info = await call(
            session, evidence, "claude_status", task_id=resumed["task_id"]
        )
        resumed_events = await history(session, evidence, resumed["task_id"])

    # The first task: steering reached the conversation and the model used it.
    assert final["status"] == "completed", final
    message = info["messages"][0]
    assert message["message_id"] == "correction-1"
    assert message["state"] == "in_conversation"
    text = final["result"]["final_text"]
    assert f"WIDGETS={WIDGETS}" in text.replace(" ", ""), text
    assert f"GADGETS={GADGETS}" in text.replace(" ", ""), text
    assert Path(info["result"]["final_text_path"]).read_text() == text
    assert final["result"]["session_id"] == started["session_id"]
    assert {"tool_use", "tool_result", "input_requested", "input_answered"} <= set(
        kinds(events)
    )
    assert {"message_queued", "message_delivered", "message_in_conversation"} <= set(
        kinds(events)
    )
    assert (workdir / "inventory.txt").read_text() == INVENTORY  # review only

    # The resumed task: same session, new task, only the new grant, real edit.
    assert resumed["task_id"] != task_id
    assert resumed["session_id"] == started["session_id"]
    assert resumed_final["status"] == "completed", resumed_final
    assert resumed_info["resumed_from"] == task_id
    used = {e["tool"] for e in resumed_events if e["kind"] == "tool_use"}
    assert used <= {"Write"}, used  # no Bash, no Read, no ask_parent inherited
    summary = (workdir / "summary.txt").read_text().strip()
    assert summary.replace(" ", "") == f"WIDGETS={WIDGETS}GADGETS={GADGETS}", summary
    assert resumed_info["limits"]["tools"] == ["Write"]
    assert resumed_info["limits"]["exposed_tools"] == ["Write"]
    assert resumed_info["limits"]["allow_questions"] is False
    assert resumed_info["invocation_cost_usd"] is not None
    assert (
        resumed_info["invocation_cost_usd"]
        <= resumed_info["usage"]["estimated_cost_usd"]
    )
    assert resumed_final["result"].get("diagnostics", []) == []
    evidence.note("final", {"first": final, "resumed": resumed_final})


async def test_live_cancel_after_a_server_restart_stops_the_real_tree(
    mcp_server: ServerFactory, live_env: dict[str, str], tmp_path: Path
) -> None:
    evidence = Evidence("cancel-reconnect")
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    brief = (
        "Using the Bash tool, run exactly this command in the foreground and wait "
        "for it to finish (do not use run_in_background, do not modify it):\n"
        "until [ -e ./never-created-live ]; do sleep 1; done\n"
        "When it finishes, reply DONE."
    )
    server_pid: int | None = None
    task_id: str | None = None
    token: str | None = None
    try:
        async with mcp_server(live_env) as session:
            server_pid = int((await call(session, evidence, "ping"))["pid"])
            started = await call(
                session,
                evidence,
                "claude_start",
                brief=brief,
                working_directory=str(workdir),
                tools=["Bash"],
                **CHEAP,
            )
            task_id = str(started["task_id"])
            running = await poll_until(
                session,
                evidence,
                task_id,
                lambda r: r.get("running_tool") is not None or ended(r),
                token=started["observation_token"],
                until="progress",
            )
            token = running["observation_token"]
            await anyio.sleep(2)
            os.kill(server_pid, signal.SIGKILL)
            await anyio.sleep(1)
    except Exception:
        if server_pid is None or task_id is None:
            raise
    assert task_id is not None and token is not None
    assert _loop_shells(), "the tool shell should outlive the server"

    async with mcp_server(live_env) as session:
        still = await call(
            session,
            evidence,
            "claude_poll",
            task_id=task_id,
            token=token,
            wait_seconds=1,
        )
        cancelled = await call(session, evidence, "claude_cancel", task_id=task_id)
        after = await call(
            session, evidence, "claude_poll", task_id=task_id, wait_seconds=0
        )

    assert still["status"] == "working" and still["token_status"] == "current"
    assert still["running_tool"]["tool"] == "Bash"  # still visible from a new server
    assert cancelled["status"] == "cancelled"
    assert cancelled["termination"] == "confirmed" and cancelled["rollback"] == "none"
    assert cancelled["cancel_requested_at"] is not None
    assert after["result"]["outcome"] == "cancelled"
    assert after["result"]["session_id"]
    deadline = time.monotonic() + 30
    while _loop_shells() and time.monotonic() < deadline:
        await anyio.sleep(0.25)
    assert _loop_shells() == ""


# --- native MCP Tasks -----------------------------------------------------------


async def native_poll(
    session: ClientSession,
    evidence: Evidence,
    task_id: str,
    until: set[str] | frozenset[str],
    *,
    timeout: float = 360,
) -> dict[str, Any]:
    """``tasks/get`` at the server's suggested interval until ``until``."""
    with anyio.fail_after(timeout):
        while True:
            task = await get_task(session, task_id)
            evidence.note("tasks/get", task)
            assert task["status"] != "failed", task  # reserved for JSON-RPC errors
            if task["status"] in until:
                return task
            await anyio.sleep(task.get("pollIntervalMs", 5000) / 1000)


async def native_start(
    session: ClientSession, evidence: Evidence, arguments: dict[str, Any]
) -> CreateTaskResult:
    created = await create_task(session, "claude_start", arguments)
    evidence.note(
        "claude_start (native)", created.model_dump(mode="json", by_alias=True)
    )
    assert isinstance(created, CreateTaskResult), created
    return created


async def test_live_native_task_observe_steer_clarify_and_complete(
    live_env: dict[str, str], tmp_path: Path
) -> None:
    evidence = Evidence("native-workflow")
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    (workdir / "inventory.txt").write_text(INVENTORY)
    brief = (
        "You are reviewing a scratch file. Do these steps in order:\n"
        "1. Run exactly `sleep 20 && wc -l inventory.txt` with the Bash tool, in "
        "the foreground, and wait for it.\n"
        "2. Read inventory.txt.\n"
        "3. Call the ask_parent tool exactly once to ask how the widget count "
        "should be formatted in your answer.\n"
        "4. Give a short final review of the file that follows the parent's "
        "answer and any later instructions from the parent.\n"
        "Do not modify any file."
    )
    async with start_modern_client(live_env) as client:
        session = client.session
        created = await native_start(
            session,
            evidence,
            {
                "brief": brief,
                "working_directory": str(workdir),
                "tools": ["Read", "Bash"],
                "allow_questions": True,
                **CHEAP,
            },
        )
        task_id = created.task_id
        assert created.status == "working"

        # Observe with the domain tool (tasks/get has no activity detail),
        # then steer while the 20 s command runs: steering stays a domain tool.
        await poll_until(
            session,
            evidence,
            task_id,
            lambda r: (r.get("running_tool") or {}).get("tool") == "Bash" or ended(r),
            until="progress",
        )
        steer = await call(
            session,
            evidence,
            "claude_steer",
            task_id=task_id,
            message_id="correction-1",
            message=(
                "Correction from your parent: in your final answer also count the "
                "lines that mention 'gadget' and include them exactly as GADGETS=<n>."
            ),
        )
        assert steer["state"] in ("queued", "delivered")

        # Clarify natively: a keyed elicitation, answered with tasks/update.
        waiting = await native_poll(
            session, evidence, task_id, {"input_required", *TERMINAL}
        )
        assert waiting["status"] == "input_required", waiting
        ((key, request),) = waiting["inputRequests"].items()
        assert request["method"] == "elicitation/create"
        assert request["params"]["message"].strip()
        ack = await update_task(
            session,
            task_id,
            {
                key: {
                    "action": "accept",
                    "content": {
                        "answer": "Report the number of lines that mention 'widget' exactly as WIDGETS=<n>."
                    },
                }
            },
        )
        evidence.note("tasks/update", ack)
        assert ack["resultType"] == "complete"

        final = await native_poll(session, evidence, task_id, TERMINAL)
        # The same task through the ordinary tool: one task system.
        status = await call(session, evidence, "claude_status", task_id=task_id)

    assert final["status"] == "completed", final
    result = final["result"]
    assert result["isError"] is False, result
    text = result["content"][0]["text"]
    # Counts that only appear if the steering message and the answer arrived.
    assert f"WIDGETS={WIDGETS}" in text.replace(" ", ""), text
    assert f"GADGETS={GADGETS}" in text.replace(" ", ""), text
    assert result["structuredContent"]["termination_reason"] == "finished"
    assert result["structuredContent"]["final_text"] == status["result"]["final_text"]
    assert status["status"] == "completed"
    assert "error" not in final and "inputRequests" not in final
    assert (workdir / "inventory.txt").read_text() == INVENTORY
    evidence.note("final", final)


async def test_live_native_turn_limit_is_a_completed_task_with_an_error_result(
    live_env: dict[str, str], tmp_path: Path
) -> None:
    evidence = Evidence("native-turn-limit")
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    (workdir / "inventory.txt").write_text(INVENTORY)
    async with start_modern_client(live_env) as client:
        created = await native_start(
            client.session,
            evidence,
            {
                "brief": (
                    "Read inventory.txt with the Read tool, then run `wc -l "
                    "inventory.txt` with the Bash tool, then report both results."
                ),
                "working_directory": str(workdir),
                "tools": ["Read", "Bash"],
                **{**CHEAP, "max_turns": 1},
            },
        )
        final = await native_poll(client.session, evidence, created.task_id, TERMINAL)

    assert final["status"] == "completed", final  # never protocol "failed"
    assert final["result"]["isError"] is True
    assert final["result"]["structuredContent"]["termination_reason"] == "max_turns"
    assert "error" not in final


async def test_live_native_cancel_from_a_new_server_stops_the_real_tree(
    live_env: dict[str, str], tmp_path: Path
) -> None:
    evidence = Evidence("native-cancel-reconnect")
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    brief = (
        "Using the Bash tool, run exactly this command in the foreground and wait "
        "for it to finish (do not use run_in_background, do not modify it):\n"
        "until [ -e ./never-created-live ]; do sleep 1; done\n"
        "When it finishes, reply DONE."
    )
    async with start_modern_client(live_env) as client:
        first_pid = (await call(client.session, evidence, "ping"))["pid"]
        created = await native_start(
            client.session,
            evidence,
            {
                "brief": brief,
                "working_directory": str(workdir),
                "tools": ["Bash"],
                **CHEAP,
            },
        )
        with anyio.fail_after(180):
            while not _loop_shells():
                await anyio.sleep(1)
        await anyio.sleep(2)  # let the worker record the shell's group
    # The first server process is gone; the task is not.

    async with start_modern_client(live_env) as client:
        assert (await call(client.session, evidence, "ping"))["pid"] != first_pid
        still = await get_task(client.session, created.task_id)
        ack = await cancel_task(client.session, created.task_id)
        evidence.note("tasks/cancel", ack)
        final = await native_poll(
            client.session, evidence, created.task_id, TERMINAL, timeout=60
        )

    assert still["status"] == "working", still
    assert ack["resultType"] == "complete"
    assert final["status"] == "cancelled", final
    deadline = time.monotonic() + 30
    while _loop_shells() and time.monotonic() < deadline:
        await anyio.sleep(0.25)
    assert _loop_shells() == ""


def _loop_shells() -> str:
    completed = subprocess.run(
        ["pgrep", "-fl", "never-created-live"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()
