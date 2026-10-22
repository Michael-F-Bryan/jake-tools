"""Delegated tasks through the real stdio server, the real SDK and the real CLI.

Every test here is ``slow`` (``uv run pytest --slow -k claude_runs_live``).
They are the lifecycle acceptance tests: nothing is stubbed, and each one
that ends a task checks the process table afterwards, because the point of
the worker is that "the task ended" also means "its processes are gone".

Environment: the server runs under the hermetic ``XDG_*`` layout from
``conftest.py`` (its own config and runs directory under ``tmp_path``) but
with the developer's real ``HOME``, which is where the CLI keeps its
credentials; nothing else is shared. The cheapest model, lowest effort and
tight limits keep each run at well under a cent.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import ClientSession

from jake_tools.claude_runs import RunState
from jake_tools.claude_runs.procs import process_table
from jake_tools.claude_runs.rundir import (
    STATUS_FILE,
    TRANSCRIPT_FILE,
    read_groups,
    read_state,
)
from jake_tools.claude_runs.spawn import group_alive, group_pids

pytestmark = pytest.mark.slow

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]

MODEL = "claude-haiku-4-5"
CHEAP: dict[str, Any] = {
    "model": MODEL,
    "effort": "low",
    "max_budget_usd": 0.25,
    # Every task is bounded below the poll window, so a run that the API
    # keeps retrying (``api_retry`` system messages) surfaces as
    # ``failed/timeout`` rather than as a worker outliving its test.
    "timeout_seconds": 240,
}
POLL_WINDOW = 300

WAIT_FOREVER_BRIEF = (
    "Using the Bash tool, run exactly this command in the foreground and wait "
    "for it to finish (do not use run_in_background, do not modify it):\n"
    "until [ -e ./never-created ]; do sleep 1; done\n"
    "When it finishes, reply DONE."
)


@pytest.fixture
def live_env(
    mcp_env_factory: Callable[..., dict[str, str]], mcp_home: Path
) -> dict[str, str]:
    """The hermetic layout plus the real HOME so the CLI can authenticate."""
    passthrough = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))
    }
    return mcp_env_factory(mcp_home, HOME=os.environ["HOME"], **passthrough)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


async def _start(session: ClientSession, **arguments: Any) -> RunState:
    result = await session.call_tool("claude_start", {**CHEAP, **arguments})
    assert result.is_error is False, result.structured_content
    assert result.structured_content is not None
    status = await session.call_tool(
        "claude_status", {"task_id": result.structured_content["task_id"]}
    )
    assert status.is_error is False, status.structured_content
    state = RunState.model_validate(status.structured_content)
    assert state.status == "working"
    assert state.pgid is not None and state.pgid > 1
    return state


async def _status(session: ClientSession, task_id: str) -> RunState:
    result = await session.call_tool("claude_status", {"task_id": task_id})
    assert result.is_error is False, result.structured_content
    return RunState.model_validate(result.structured_content)


async def _wait_terminal(
    session: ClientSession, task_id: str, *, timeout: float = POLL_WINDOW
) -> RunState:
    deadline = time.monotonic() + timeout
    while True:
        state = await _status(session, task_id)
        if state.is_terminal:
            return state
        assert time.monotonic() < deadline, f"task {task_id} still working"
        await anyio.sleep(1)


async def _wait_for_bash_tool(state: RunState, *, timeout: float = POLL_WINDOW) -> None:
    """Block until the transcript shows the Bash tool running the loop.

    The CLI runs tool shells outside the worker's process group (observed:
    the group holds only the worker and the CLI while the loop runs), so the
    transcript, not ``pgrep -g``, is where "the tool is running" shows up.
    Gives up as soon as the task is terminal, naming how many times the API
    was retried, so a rate-limited run reads as what it is.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        transcript = (
            _transcript(state) if (state.run_dir / TRANSCRIPT_FILE).exists() else []
        )
        if any(use["name"] == "Bash" for use in _tool_uses(transcript)):
            await anyio.sleep(2)  # let the shell actually start
            return
        current = read_state(state.run_dir)
        if current.is_terminal:
            break
        await anyio.sleep(0.5)
    retries = sum(
        1
        for m in _transcript(state)
        if m["type"] == "SystemMessage" and m.get("subtype") == "api_retry"
    )
    pytest.fail(
        f"task {state.task_id} never invoked Bash (now {read_state(state.run_dir).status}; "
        f"{retries} api_retry event(s) in the transcript)"
    )


def _pgrep_listing(pgid: int) -> str:
    completed = subprocess.run(
        ["pgrep", "-g", str(pgid), "-l"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


async def _assert_group_gone(pgid: int, *, timeout: float = 30) -> None:
    """After a bounded wait: killpg(pgid, 0) fails, pgrep -g is empty, and no
    shell running the wait loop survives anywhere in the process table."""
    deadline = time.monotonic() + timeout
    while (group_alive(pgid) or _loop_shells()) and time.monotonic() < deadline:
        await anyio.sleep(0.25)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
    assert _pgrep_listing(pgid) == ""
    assert group_pids(pgid) == set()
    assert _loop_shells() == ""


async def _wait_for_recorded_group(state: RunState, *, timeout: float = 60) -> int:
    """Block until the worker has recorded the tool shell's group; returns it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        groups = read_groups(state.run_dir).groups
        if groups:
            return groups[-1].pgid
        await anyio.sleep(0.25)
    pytest.fail(f"task {state.task_id} recorded no descendant group")


def _cli_pids(worker_pid: int) -> set[int]:
    """The CLI processes: direct children of the worker."""
    return {info.pid for info in process_table().values() if info.ppid == worker_pid}


def _loop_shells() -> str:
    """Any process whose command line carries the wait loop's marker."""
    completed = subprocess.run(
        ["pgrep", "-fl", "never-created"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


def _transcript(state: RunState) -> list[dict[str, Any]]:
    lines = (state.run_dir / TRANSCRIPT_FILE).read_text().splitlines()
    return [json.loads(line) for line in lines if line]


def _tool_uses(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uses: list[dict[str, Any]] = []
    for message in transcript:
        if message["type"] != "AssistantMessage":
            continue
        uses.extend(block for block in message["content"] if "name" in block)
    return uses


def _result_message(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    results = [m for m in transcript if m["type"] == "ResultMessage"]
    assert len(results) == 1
    return results[0]


# 1. a real task completes with the result message's text and usage ----------


async def test_live_delegated_task_completes_with_final_text_and_usage(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief="Reply with exactly the single word OK and nothing else.",
            working_directory=str(workdir),
            max_turns=2,
        )
        state = await _wait_terminal(session, started.task_id)

    assert state.status == "completed"
    assert state.termination_reason == "finished"
    assert state.final_text is not None and "OK" in state.final_text
    assert state.started_at is not None and state.finished_at is not None
    assert state.usage is not None
    assert state.usage.api_calls >= 1
    assert (
        state.usage.estimated_cost_usd is not None
        and state.usage.estimated_cost_usd > 0
    )
    assert state.error is None

    transcript = _transcript(state)
    assert _result_message(transcript)["result"] == state.final_text
    assert (
        json.loads((state.run_dir / STATUS_FILE).read_text())["status"] == "completed"
    )
    telemetry = json.loads((state.run_dir / "telemetry.json").read_text())
    assert len(telemetry["calls"]) == 1 and telemetry["calls"][0]["status"] == "success"
    result = json.loads((state.run_dir / "result.json").read_text())
    assert result["termination_reason"] == "finished"
    await _assert_group_gone(started.pgid or 0)


# 2. a real edit-and-test task actually edits and actually runs the test -----


async def test_live_edit_and_test_task_changes_a_scratch_checkout(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    subprocess.run(["git", "init", "-q", str(workdir)], check=True)
    (workdir / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (workdir / "test_calc.py").write_text(
        "from calc import add\n"
        "assert add(2, 3) == 5, add(2, 3)\n"
        "assert add(-1, 1) == 0\n"
        "print('PASS')\n"
    )
    subprocess.run(["git", "-C", str(workdir), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workdir),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "broken add",
        ],
        check=True,
    )
    broken = subprocess.run(
        [sys.executable, "test_calc.py"], cwd=workdir, capture_output=True, text=True
    )
    assert broken.returncode != 0

    brief = (
        "This directory is a small Python project. `calc.add` has a bug: it "
        "subtracts instead of adding, and `test_calc.py` fails. Fix `calc.py` "
        "with the Edit tool so `add` returns the sum, then run the test with "
        "the Bash tool: `python3 test_calc.py`. It prints PASS when fixed. "
        "Reply with PASS once you have seen the test print it, or FAIL."
    )
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=brief,
            working_directory=str(workdir),
            tools=["Edit", "Write", "Bash"],
            max_turns=12,
        )
        state = await _wait_terminal(session, started.task_id)

    assert state.status == "completed", state
    assert "a + b" in (workdir / "calc.py").read_text().replace(" ", " ")
    fixed = subprocess.run(
        [sys.executable, "test_calc.py"], cwd=workdir, capture_output=True, text=True
    )
    assert fixed.returncode == 0 and "PASS" in fixed.stdout
    changed = subprocess.run(
        ["git", "-C", str(workdir), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "calc.py" in changed

    transcript = _transcript(state)
    names = {use["name"] for use in _tool_uses(transcript)}
    assert names & {"Edit", "Write"}, names
    assert "Bash" in names
    assert _result_message(transcript)["permission_denials"] in (None, [])
    assert state.final_text is not None and "PASS" in state.final_text
    await _assert_group_gone(started.pgid or 0)


# 3. no grant, no tools ----------------------------------------------------------


async def test_live_task_without_a_grant_cannot_use_any_tool(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    brief = (
        "Create a file named proof.txt containing the word done in the current "
        "working directory, using the Write tool or a shell command. If you "
        "have no way to do that, reply with exactly: CANNOT"
    )
    async with mcp_server(live_env) as session:
        started = await _start(
            session, brief=brief, working_directory=str(workdir), max_turns=3
        )
        state = await _wait_terminal(session, started.task_id)

    assert not (workdir / "proof.txt").exists()
    assert sorted(p.name for p in workdir.iterdir()) == []
    transcript = _transcript(state)
    assert _tool_uses(transcript) == []
    assert state.status == "completed", state
    await _assert_group_gone(started.pgid or 0)


# 4. timeout --------------------------------------------------------------------


async def test_live_timeout_fails_the_task_and_empties_its_group(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=WAIT_FOREVER_BRIEF,
            working_directory=str(workdir),
            tools=["Bash"],
            max_turns=4,
            # Long enough to reach the tool call through a few API retries,
            # short enough that the loop (which never ends) is what gets cut.
            timeout_seconds=90,
        )
        pgid = started.pgid or 0
        await _wait_for_bash_tool(started)
        state = await _wait_terminal(session, started.task_id, timeout=150)

    assert state.status == "failed"
    assert state.termination_reason == "timeout"
    assert state.error and "timeout" in state.error
    assert state.finished_at is not None
    on_disk = RunState.model_validate_json((state.run_dir / STATUS_FILE).read_bytes())
    assert on_disk.termination_reason == "timeout"
    await _assert_group_gone(pgid)


# 5. cancel ---------------------------------------------------------------------


async def test_live_cancel_stops_the_task_and_empties_its_group(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=WAIT_FOREVER_BRIEF,
            working_directory=str(workdir),
            tools=["Bash"],
            max_turns=4,
            timeout_seconds=POLL_WINDOW,
        )
        pgid = started.pgid or 0
        await _wait_for_bash_tool(started)
        before = time.monotonic()
        result = await session.call_tool("claude_cancel", {"task_id": started.task_id})
        elapsed = time.monotonic() - before
        assert result.is_error is False, result.structured_content
        state = RunState.model_validate(result.structured_content)
        again = await session.call_tool("claude_cancel", {"task_id": started.task_id})

    assert state.status == "cancelled"
    assert state.termination_reason == "cancelled"
    assert state.finished_at is not None
    assert elapsed < 20
    assert RunState.model_validate(again.structured_content) == state  # idempotent
    await _assert_group_gone(pgid)


# 6. budget and turns --------------------------------------------------------------


async def test_live_tiny_budget_fails_with_budget(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    brief = (
        "Use the Bash tool to run `ls -la`, then `pwd`, then `date`, one tool "
        "call each, then summarise what you saw."
    )
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=brief,
            working_directory=str(workdir),
            tools=["Bash"],
            max_turns=6,
            max_budget_usd=0.001,
        )
        state = await _wait_terminal(session, started.task_id)

    assert state.status == "failed"
    assert state.termination_reason == "budget", state
    assert state.error and "error_max_budget_usd" in state.error
    assert state.usage is not None and state.usage.estimated_cost_usd
    await _assert_group_gone(started.pgid or 0)


async def test_live_max_turns_fails_with_max_turns(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    brief = (
        "Use the Bash tool to run `echo one`, then `echo two`, then `echo "
        "three`, each as a separate tool call, then report what was printed."
    )
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=brief,
            working_directory=str(workdir),
            tools=["Bash"],
            max_turns=1,
        )
        state = await _wait_terminal(session, started.task_id)

    assert state.status == "failed"
    assert state.termination_reason == "max_turns", state
    assert state.error and "error_max_turns" in state.error
    await _assert_group_gone(started.pgid or 0)


async def test_live_max_turns_handoff_remembers_and_cannot_use_tools(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    """After a turn-limit stop the same session writes a tool-free handoff.

    The codeword appears only in the brief, so finding it in the handoff
    means the handoff call had the earlier conversation. The brief also asks
    for a file to be created during any handoff; with no tools exposed it
    must not exist, and the handoff's own ``init`` must list no tools."""
    codeword = f"PELICAN-{os.urandom(3).hex().upper()}"
    brief = (
        f"The project codeword is {codeword}. Use the Bash tool to run `echo "
        "one`, then `echo two`, then `echo three`, each as a separate tool "
        "call, then report what was printed. If you are ever asked for a "
        "handoff, include the codeword in it, and first run `touch "
        "handoff-tool-ran` with Bash."
    )
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=brief,
            working_directory=str(workdir),
            tools=["Bash"],
            max_turns=1,
        )
        state = await _wait_terminal(session, started.task_id)

    assert state.status == "failed"
    assert state.termination_reason == "max_turns", state
    handoff = state.handoff
    assert handoff is not None
    assert handoff.status == "written", handoff
    assert handoff.source == "worker_reported"
    assert handoff.text and codeword in handoff.text, handoff.text
    assert not (workdir / "handoff-tool-ran").exists()
    transcript = _transcript(state)
    inits = [
        m for m in transcript if m["type"] == "SystemMessage" and m["subtype"] == "init"
    ]
    last_init = inits[-1]
    tools = last_init.get("tools", last_init.get("data", {}).get("tools"))
    assert tools == [], tools
    assert last_init["data"]["session_id"] == started.session_id
    handoff_at = transcript.index(last_init)
    assert not _tool_uses(transcript[handoff_at:])
    assert handoff.cost_usd is not None and handoff.cost_usd < 0.05
    await _assert_group_gone(started.pgid or 0)


# 7. the server dies; the worker does not ----------------------------------------


async def test_live_worker_survives_server_sigkill_and_a_new_server_cancels_it(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    server_pid: int | None = None
    started: RunState | None = None
    try:
        async with mcp_server(live_env) as session:
            ping = await session.call_tool("ping", {})
            assert ping.structured_content is not None
            server_pid = int(ping.structured_content["pid"])
            started = await _start(
                session,
                brief=WAIT_FOREVER_BRIEF,
                working_directory=str(workdir),
                tools=["Bash"],
                max_turns=4,
                timeout_seconds=POLL_WINDOW,
            )
            await _wait_for_bash_tool(started)
            os.kill(server_pid, signal.SIGKILL)
            await anyio.sleep(1)
    except Exception:  # the client's teardown of a SIGKILLed server may complain
        if server_pid is None or started is None:
            raise
    assert server_pid is not None and started is not None
    pgid = started.pgid or 0

    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)
    assert group_alive(pgid)
    assert len(group_pids(pgid)) >= 2, _pgrep_listing(pgid)  # worker + CLI
    assert _loop_shells()  # and the tool's shell is still running somewhere

    async with mcp_server(live_env) as session:
        state = await _status(session, started.task_id)
        assert state.status == "working"
        assert state.pgid == pgid
        result = await session.call_tool("claude_cancel", {"task_id": started.task_id})
        assert result.is_error is False, result.structured_content
        cancelled = RunState.model_validate(result.structured_content)

    assert cancelled.status == "cancelled"
    assert cancelled.termination_reason == "cancelled"
    await _assert_group_gone(pgid)


# 8. the CLI is SIGKILLed mid-tool; the worker still clears the tool shell ------


async def test_live_sigkill_of_the_cli_still_ends_with_the_tool_shell_gone(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    """The CLI runs tool shells in their own session (setsid). Killing the CLI
    outright orphans the shell; the worker must reach it through groups.json."""
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=WAIT_FOREVER_BRIEF,
            working_directory=str(workdir),
            tools=["Bash"],
            max_turns=4,
            timeout_seconds=POLL_WINDOW,
        )
        pgid = started.pgid or 0
        await _wait_for_bash_tool(started)
        shell_pgid = await _wait_for_recorded_group(started)
        assert shell_pgid != pgid
        assert _loop_shells()
        cli = _cli_pids(started.pid or 0)
        assert cli, "no CLI child under the worker"
        for pid in cli:
            os.kill(pid, signal.SIGKILL)
        state = await _wait_terminal(session, started.task_id, timeout=60)

    assert state.status == "failed"
    assert state.termination_reason == "sdk_error", state
    on_disk = RunState.model_validate_json((state.run_dir / STATUS_FILE).read_bytes())
    assert on_disk == state
    await _assert_group_gone(pgid)
    assert not group_alive(shell_pgid)


# 9. the worker group is SIGKILLed with no SIGTERM; cancel still clears the shell


async def test_live_sigkill_of_the_worker_group_then_cancel_clears_the_tool_shell(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    """What claude_cancel's SIGKILL backstop (or a server's group kill) leaves:
    the shell in its own session survives the group kill, and the server must
    settle the task and clear it from groups.json."""
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=WAIT_FOREVER_BRIEF,
            working_directory=str(workdir),
            tools=["Bash"],
            max_turns=4,
            timeout_seconds=POLL_WINDOW,
        )
        pgid = started.pgid or 0
        await _wait_for_bash_tool(started)
        shell_pgid = await _wait_for_recorded_group(started)
        os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while group_alive(pgid) and time.monotonic() < deadline:
            await anyio.sleep(0.1)
        assert not group_alive(pgid)
        assert _loop_shells(), "the shell should have survived the group kill"
        result = await session.call_tool("claude_cancel", {"task_id": started.task_id})
        assert result.is_error is False, result.structured_content
        state = RunState.model_validate(result.structured_content)

    assert state.status == "failed"
    assert state.termination_reason == "worker_died", state
    on_disk = RunState.model_validate_json((state.run_dir / STATUS_FILE).read_bytes())
    assert on_disk == state
    await _assert_group_gone(pgid)
    assert not group_alive(shell_pgid)


# 10. exposure is not approval: a narrow Bash rule approves only its command ----


async def test_live_narrow_bash_rule_approves_only_matching_commands(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    """``Bash`` exposed, only ``Bash(touch approved-*)`` approved, ``dontAsk``.

    ``touch`` changes the filesystem, so it is never one of the CLI's
    intrinsically approved read-only commands: the approved one must run
    and the unapproved one must be denied. ``ls`` is read-only, and runs
    with no rule at all: "not approved by our rules" is not "forbidden".
    """
    brief = (
        "Use the Bash tool to run each of these three commands exactly as "
        "written, as three separate Bash tool calls, in this order, even if "
        "one is denied or fails:\n1. touch approved-1\n2. touch denied-1\n"
        "3. ls\nDo not try any other command. Then reply with one line per "
        "command saying whether it ran or was denied."
    )
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief=brief,
            working_directory=str(workdir),
            tools=["Bash"],
            allowed_tools=["Bash(touch approved-*)"],
            max_turns=6,
        )
        state = await _wait_terminal(session, started.task_id)

    assert state.status == "completed", state
    assert (workdir / "approved-1").exists()
    assert not (workdir / "denied-1").exists()
    transcript = _transcript(state)
    denials = _result_message(transcript)["permission_denials"] or []
    denied = [d["tool_input"].get("command") for d in denials]
    assert "touch denied-1" in denied, denials
    assert "touch approved-1" not in denied
    results = {
        block["tool_use_id"]: block
        for message in transcript
        if message["type"] == "UserMessage" and isinstance(message["content"], list)
        for block in message["content"]
        if "tool_use_id" in block
    }
    ls_calls = [u for u in _tool_uses(transcript) if u["input"].get("command") == "ls"]
    assert ls_calls, "the model did not run ls"
    assert not results[ls_calls[0]["id"]].get("is_error"), results[ls_calls[0]["id"]]
    await _assert_group_gone(started.pgid or 0)


# 11. tools outside the old hard-coded list reach the session -------------------


async def test_live_tools_outside_the_old_list_are_exposed(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(live_env) as session:
        started = await _start(
            session,
            brief="Reply with exactly: OK. Do not use any tools.",
            working_directory=str(workdir),
            tools=["Read", "NotebookEdit", "Monitor"],
            max_turns=2,
        )
        state = await _wait_terminal(session, started.task_id)

    assert state.status == "completed", state
    (init,) = [
        m
        for m in _transcript(state)
        if m["type"] == "SystemMessage" and m["subtype"] == "init"
    ]
    assert {"Read", "NotebookEdit", "Monitor"} <= set(init["data"]["tools"])
    await _assert_group_gone(started.pgid or 0)


# 12. a tool the session does not provide fails the task, by name ---------------


async def test_live_unavailable_tool_fails_the_task_by_name(
    mcp_server: ServerFactory, live_env: dict[str, str], workdir: Path
) -> None:
    """Also the Artifact feasibility gate (``docs/claude-tools.md``).

    Agent SDK sessions do not provide ``Artifact`` (documented, and observed
    with a claude.ai Max login on CLI 2.1.293): the CLI drops it from
    ``--tools`` silently. If this starts failing because the session lists
    ``Artifact``, re-run the publishing feasibility checks.
    """
    async with mcp_server(live_env) as session:
        result = await session.call_tool(
            "claude_start",
            {
                **CHEAP,
                "brief": "Reply with exactly: OK. Do not use any tools.",
                "working_directory": str(workdir),
                "tools": ["Read", "Artifact"],
                "max_turns": 2,
            },
        )
        assert result.is_error is False, result.structured_content
        assert result.structured_content is not None
        task_id = result.structured_content["task_id"]
        state = await _wait_terminal(session, task_id)

    assert state.status == "failed", state
    assert state.termination_reason == "tools_unavailable", state
    assert state.error is not None and "Artifact" in state.error
    (init,) = [
        m
        for m in _transcript(state)
        if m["type"] == "SystemMessage" and m["subtype"] == "init"
    ]
    assert "Read" in init["data"]["tools"]
    assert "Artifact" not in init["data"]["tools"]
    await _assert_group_gone(state.pgid or 0)
