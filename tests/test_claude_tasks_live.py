"""The interactive delegation workflow against the real CLI and a real model.

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
    predicate: Callable[[dict[str, Any], list[dict[str, Any]]], bool],
    *,
    cursor: str | None = None,
    timeout: float = 360,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    deadline = time.monotonic() + timeout
    events: list[dict[str, Any]] = []
    while True:
        args: dict[str, Any] = {"task_id": task_id, "wait_seconds": 20}
        if cursor:
            args["cursor"] = cursor
        result = await call(session, evidence, "claude_poll", **args)
        events.extend(result["events"])
        cursor = result["next_cursor"]
        if predicate(result, events):
            assert cursor is not None
            return result, events, cursor
        assert result["status"] in ("working", "input_required"), result
        assert time.monotonic() < deadline, [e["kind"] for e in events]


def ended(result: dict[str, Any], _: list[dict[str, Any]]) -> bool:
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

        # Observe: the Bash step shows up as tool activity, with progress.
        observed, events, cursor = await poll_until(
            session,
            evidence,
            task_id,
            lambda _, evs: any(
                e["kind"] == "tool_use" and e["tool"] == "Bash" for e in evs
            ),
        )
        assert observed["status_message"]
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

        # Clarify: the model asks; the answer carries the widget format.
        waiting, more, cursor = await poll_until(
            session,
            evidence,
            task_id,
            lambda r, _: r["status"] == "input_required" or ended(r, _),
            cursor=cursor,
        )
        events += more
        assert waiting["status"] == "input_required", waiting
        request = waiting["pending_inputs"][0]
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

        final, more, _ = await poll_until(
            session, evidence, task_id, ended, cursor=cursor
        )
        events += more

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
        resumed_final, resumed_events, _ = await poll_until(
            session, evidence, resumed["task_id"], ended
        )

    # The first task: steering reached the conversation and the model used it.
    assert final["status"] == "completed", final
    message = final["messages"][0]
    assert message["message_id"] == "correction-1"
    assert message["state"] == "in_conversation"
    text = final["result"]["final_text"]
    assert f"WIDGETS={WIDGETS}" in text.replace(" ", ""), text
    assert f"GADGETS={GADGETS}" in text.replace(" ", ""), text
    assert Path(final["result"]["final_text_path"]).read_text() == text
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
    assert resumed_final["resumed_from"] == task_id
    used = {e["tool"] for e in resumed_events if e["kind"] == "tool_use"}
    assert used <= {"Write"}, used  # no Bash, no Read, no ask_parent inherited
    summary = (workdir / "summary.txt").read_text().strip()
    assert summary.replace(" ", "") == f"WIDGETS={WIDGETS}GADGETS={GADGETS}", summary
    assert resumed_final["limits"]["tools"] == ["Write"]
    assert resumed_final["limits"]["allow_questions"] is False
    assert resumed_final["invocation_cost_usd"] is not None
    assert (
        resumed_final["invocation_cost_usd"]
        <= resumed_final["usage"]["estimated_cost_usd"]
    )
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
    cursor: str | None = None
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
            _, _, cursor = await poll_until(
                session,
                evidence,
                task_id,
                lambda _, evs: any(e["kind"] == "tool_use" for e in evs),
            )
            await anyio.sleep(2)
            os.kill(server_pid, signal.SIGKILL)
            await anyio.sleep(1)
    except Exception:
        if server_pid is None or task_id is None:
            raise
    assert task_id is not None and cursor is not None
    assert _loop_shells(), "the tool shell should outlive the server"

    async with mcp_server(live_env) as session:
        still = await call(
            session,
            evidence,
            "claude_poll",
            task_id=task_id,
            cursor=cursor,
            wait_seconds=1,
        )
        cancelled = await call(session, evidence, "claude_cancel", task_id=task_id)
        after = await call(
            session, evidence, "claude_poll", task_id=task_id, wait_seconds=0
        )

    assert still["status"] == "working"
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested_at"] is not None
    assert after["result"]["outcome"] == "cancelled"
    assert after["result"]["session_id"]
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
