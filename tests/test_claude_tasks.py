"""Delegated-task observation, steering, clarification, idempotency and resume.

Every test drives the real stdio server, which spawns the real worker,
which runs the real Claude Agent SDK. Only the ``claude`` executable is a
stand-in (``tests/fake_claude.py``, first on PATH): it speaks the CLI's
stream-json and control protocol and answers mechanically, so these tests
prove mechanics, never judgement. ``test_claude_tasks_live.py`` repeats the
workflow against the real CLI and model.
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

from jake_tools.claude_runs.procs import process_table

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]

FAST: dict[str, Any] = {"model": "fake-model", "max_turns": 5, "max_budget_usd": 1.0}


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def runs_dir(mcp_home: Path) -> Path:
    return mcp_home / ".local" / "state" / "jake-tools" / "runs"


async def call(session: ClientSession, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await session.call_tool(tool, arguments)
    assert result.is_error is False, result.structured_content
    assert result.structured_content is not None
    return result.structured_content


async def call_error(
    session: ClientSession, tool: str, **arguments: Any
) -> dict[str, Any]:
    result = await session.call_tool(tool, arguments)
    assert result.is_error is True, result.structured_content
    assert result.structured_content is not None
    return result.structured_content


async def start(
    session: ClientSession, workdir: Path, brief: str, **extra: Any
) -> dict[str, Any]:
    return await call(
        session,
        "claude_start",
        brief=brief,
        working_directory=str(workdir),
        **{**FAST, "timeout_seconds": 60, **extra},
    )


async def poll_until(
    session: ClientSession,
    task_id: str,
    predicate: Callable[[dict[str, Any], list[dict[str, Any]]], bool],
    *,
    timeout: float = 60,
    cursor: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Poll with the cursor until ``predicate(result, all_events)`` holds."""
    deadline = time.monotonic() + timeout
    events: list[dict[str, Any]] = []
    while True:
        args: dict[str, Any] = {"task_id": task_id, "wait_seconds": 5}
        if cursor:
            args["cursor"] = cursor
        result = await call(session, "claude_poll", **args)
        events.extend(result["events"])
        cursor = result["next_cursor"]
        if predicate(result, events):
            assert cursor is not None
            return result, events, cursor
        assert time.monotonic() < deadline, (
            result["status"],
            [e["kind"] for e in events],
        )


def terminal(result: dict[str, Any], _: list[dict[str, Any]]) -> bool:
    return result["status"] in ("completed", "failed", "cancelled")


def kinds(events: list[dict[str, Any]]) -> list[str]:
    return [e["kind"] for e in events]


def argv_files(state: Path) -> list[list[str]]:
    files = sorted(state.glob("argv-*.json"), key=lambda p: int(p.stem.split("-")[1]))
    return [json.loads(p.read_text()) for p in files]


def flag(argv: list[str], name: str) -> str | None:
    for index, token in enumerate(argv):
        if token == name:
            return argv[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def sleepers(marker: str) -> list[int]:
    completed = subprocess.run(
        ["pgrep", "-x", "-f", f"sleep {marker}"], capture_output=True, text=True
    )
    return [int(pid) for pid in completed.stdout.split()]


# --- discovery -----------------------------------------------------------------


async def test_stdio_lists_the_delegation_tools_with_their_arguments(
    mcp_server: ServerFactory, mcp_env: dict[str, str]
) -> None:
    async with mcp_server(mcp_env) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}

    expected = {
        "claude_start": {"brief"},
        "claude_status": {"task_id"},
        "claude_poll": set(),  # task_id or tasks; checked at the tool boundary
        "claude_transcript": {"task_id"},
        "claude_steer": {"task_id", "message", "message_id"},
        "claude_update": {"task_id", "responses"},
        "claude_cancel": {"task_id"},
        "claude_list": set(),
    }
    for name, required in expected.items():
        assert set(tools[name].input_schema.get("required", [])) == required, name
        assert tools[name].output_schema is None
    assert {"resume_task_id", "request_id", "allow_questions", "allowed_tools"} <= set(
        tools["claude_start"].input_schema["properties"]
    )
    # No catalogue of tool names is advertised: the runtime is the authority.
    start_schema = tools["claude_start"].input_schema["properties"]["tools"]
    assert "enum" not in json.dumps(start_schema)
    assert {"cursor", "wait_seconds", "limit"} <= set(
        tools["claude_poll"].input_schema["properties"]
    )


# --- observing: poll, cursor, events, results -----------------------------------


async def test_poll_returns_new_events_once_and_a_complete_result(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:echo]] hello there")
        task_id = started["task_id"]
        assert started["session_id"]
        assert started["status"] == "working"
        final, events, cursor = await poll_until(session, task_id, terminal)
        again = await call(
            session, "claude_poll", task_id=task_id, cursor=cursor, wait_seconds=0
        )
        from_start = await call(session, "claude_poll", task_id=task_id, wait_seconds=0)

    assert kinds(events) == ["started", "assistant_text", "turn_result", "finished"]
    assert [e["seq"] for e in events] == [1, 2, 3, 4]
    assert again["events"] == [] and again["has_more"] is False
    assert again["wake_reason"] == "terminal"
    assert kinds(from_start["events"]) == kinds(events)  # stable replay from the start

    assert final["status"] == "completed"
    assert final["termination_reason"] == "finished"
    result = final["result"]
    assert result["final_text"] == "ECHO: [[scenario:echo]] hello there"
    assert Path(result["final_text_path"]).read_text() == result["final_text"]
    assert result["session_id"] == started["session_id"]
    assert result["resumable"] is True
    assert result["usage"]["estimated_cost_usd"] == 0.01
    assert Path(result["transcript_path"]).exists()
    assert final["last_activity_at"] is not None
    assert final["limits"]["max_turns"] == FAST["max_turns"]
    assert final["limits"]["deadline_at"] is not None

    argv = argv_files(fake_claude_state)[0]
    assert flag(argv, "--session-id") == started["session_id"]
    assert "--replay-user-messages" in argv
    assert "--strict-mcp-config" in argv
    assert flag(argv, "--permission-mode") == "dontAsk"
    assert flag(argv, "--tools") == ""  # no built-ins without a grant
    assert flag(argv, "--allowedTools") is None  # and no ask_parent by default
    assert flag(argv, "--mcp-config") is None
    assert final["limits"]["allow_questions"] is False


async def test_multi_poll_returns_all_ready_tasks_with_independent_cursors(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        tasks = []
        for text in ("first", "second"):
            started = await start(session, workdir, f"[[scenario:echo]] {text}")
            await poll_until(session, started["task_id"], terminal)
            tasks.append({"task_id": started["task_id"]})
        page = await call(session, "claude_poll", tasks=tasks, wait_seconds=20, limit=1)
        assert page["wake_reason"] == "ready"
        assert [r["task_id"] for r in page["ready"]] == [t["task_id"] for t in tasks]
        assert all(len(r["events"]) == 1 and r["has_more"] for r in page["ready"])
        next_page = await call(
            session,
            "claude_poll",
            wait_seconds=0,
            limit=1,
            tasks=[
                {"task_id": r["task_id"], "cursor": r["next_cursor"]}
                for r in page["ready"]
            ],
        )
        assert all(r["events"][0]["seq"] == 2 for r in next_page["ready"])


async def test_multi_poll_terminal_wait_ignores_progress_and_returns_first_finished(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        slow = await start(session, workdir, "[[scenario:slow_tool:8]] go")
        fast = await start(session, workdir, "[[scenario:slow_tool:2]] go")
        began = time.monotonic()
        page = await call(
            session,
            "claude_poll",
            tasks=[{"task_id": slow["task_id"]}, {"task_id": fast["task_id"]}],
            until="terminal",
            wait_seconds=20,
        )
        assert time.monotonic() - began < 7
        assert [r["task_id"] for r in page["ready"]] == [fast["task_id"]]
        assert page["ready"][0]["wake_reason"] == "terminal"
        assert page["ready"][0]["result"]["final_text"]
        assert (await call(session, "claude_status", task_id=slow["task_id"]))[
            "status"
        ] == "working"
        await call(session, "claude_cancel", task_id=slow["task_id"])


async def test_poll_waits_for_change_and_stops_at_its_own_deadline(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:slow_tool:6]] go")
        task_id = started["task_id"]
        _, _, cursor = await poll_until(
            session, task_id, lambda _, evs: "tool_use" in kinds(evs)
        )
        began = time.monotonic()
        quiet = await call(
            session, "claude_poll", task_id=task_id, cursor=cursor, wait_seconds=1.5
        )
        elapsed = time.monotonic() - began
        final, events, _ = await poll_until(session, task_id, terminal, cursor=cursor)

    assert quiet["wake_reason"] == "deadline"
    assert quiet["events"] == [] and quiet["status"] == "working"
    assert 1.2 <= elapsed < 5
    assert quiet["status_message"] == "Bash: sleep 6.0"
    assert final["status"] == "completed"  # the poll deadline is not the task's


async def test_poll_rejects_bad_bounds_and_cursors_without_waiting(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:echo]] x")
        task_id = started["task_id"]
        await poll_until(session, task_id, terminal)
        cases: list[tuple[dict[str, Any], str]] = [
            ({"wait_seconds": 26}, "wait_seconds"),
            ({"wait_seconds": -1}, "wait_seconds"),
            ({"limit": 0}, "limit"),
            ({"limit": 101}, "limit"),
            ({"cursor": "garbage"}, "cursor"),
            ({"cursor": "20261008T031500Z-1a2b3c4d:1:0"}, "cursor"),  # another task
            ({"cursor": f"{task_id}:99:999999"}, "cursor"),  # past the end
            ({"cursor": f"{task_id}:2:3"}, "cursor"),  # not at a line start
            ({"cursor": f"{task_id}:3:0"}, "cursor"),  # wrong seq for the offset
        ]
        for arguments, argument in cases:
            began = time.monotonic()
            # A long wait proves validation happens before any waiting.
            error = await call_error(
                session,
                "claude_poll",
                task_id=task_id,
                **{"wait_seconds": 20, **arguments},
            )
            assert error["code"] == "invalid_argument", arguments
            assert error["detail"] == {"argument": argument}, arguments
            assert time.monotonic() - began < 2, arguments
        unknown = await call_error(session, "claude_poll", task_id="nope")
        assert unknown["code"] == "unknown_task"


async def test_cursor_pages_are_bounded_and_resume_exactly(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    fake_env = {**fake_env, "FAKE_CLAUDE_BIG_EVENTS": "7"}
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:big:3000]] go", max_turns=20
        )
        task_id = started["task_id"]
        await poll_until(session, task_id, terminal)
        seen: list[int] = []
        cursor = None
        pages = 0
        while True:
            args: dict[str, Any] = {"task_id": task_id, "limit": 3}
            if cursor:
                args["cursor"] = cursor
            page = await call(session, "claude_transcript", **args)
            pages += 1
            assert len(page["events"]) <= 3
            seen.extend(e["seq"] for e in page["events"])
            cursor = page["next_cursor"]
            if not page["has_more"]:
                break

    assert seen == list(range(1, len(seen) + 1))  # no gap, no repeat
    assert pages >= 4


async def test_long_text_is_clipped_with_coverage_and_full_text_stays_reachable(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:big:30000]] go")
        task_id = started["task_id"]
        final, events, _ = await poll_until(session, task_id, terminal)
        wide = await call(
            session,
            "claude_transcript",
            task_id=task_id,
            max_text_chars=50000,
            limit=200,
        )

    narration = next(e for e in events if e["kind"] == "assistant_text")
    assert narration["text_truncated"] is True
    assert len(narration["text"]) == 1000
    assert narration["text_chars"] == 30002
    assert final["events_truncated"] or any(e["text_truncated"] for e in events)
    wide_narration = next(e for e in wide["events"] if e["kind"] == "assistant_text")
    assert len(wide_narration["text"]) == 30002 and not wide_narration["text_truncated"]
    tool_result = next(e for e in wide["events"] if e["kind"] == "tool_result")
    assert tool_result["text_chars"] == 20000 and tool_result["text_truncated"] is True
    assert len(tool_result["text"]) == 8000  # stored bound; raw transcript has it all
    result = final["result"]
    assert result["final_text_truncated"] is True
    assert result["final_text_chars"] == 30000
    assert len(result["final_text"]) == 20000
    assert len(Path(result["final_text_path"]).read_text()) == 30000
    assert "y" * 20000 in Path(wide["raw_transcript_path"]).read_text()


async def test_thinking_never_reaches_events_or_transcript_pages(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:thinking]] go")
        final, events, _ = await poll_until(session, started["task_id"], terminal)
        page = await call(session, "claude_transcript", task_id=started["task_id"])

    assert "SECRET-THOUGHT" not in json.dumps(final)
    assert "SECRET-THOUGHT" not in json.dumps(events)
    assert "SECRET-THOUGHT" not in json.dumps(page)
    assert any(e.get("text") == "Public narration." for e in events)


async def test_a_turn_limit_keeps_partial_output_and_says_why(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:max_turns]] go")
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    assert final["status"] == "failed"
    assert final["termination_reason"] == "max_turns"
    result = final["result"]
    assert result["final_text"] == "partial answer"
    assert result["last_narration"] == "Partial progress before the limit."
    assert "error_max_turns" in result["error"]


async def test_a_turn_limit_hands_off_in_the_same_session_without_tools(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,
) -> None:
    """Mechanics only: the fake is not a model. The real-model acceptance is
    ``test_live_max_turns_handoff_remembers_and_cannot_use_tools``."""
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:max_turns]] go", tools=["Bash"]
        )
        final, events, _ = await poll_until(session, started["task_id"], terminal)

    # The outcome is still the turn limit; the handoff is extra evidence.
    assert final["status"] == "failed"
    assert final["termination_reason"] == "max_turns"
    result = final["result"]
    assert result["final_text"] == "partial answer"
    assert result["last_narration"] == "Partial progress before the limit."
    handoff = result["handoff"]
    assert handoff["status"] == "written", handoff
    assert handoff["source"] == "worker_reported"
    assert handoff["text"] == "REMEMBERED: [[scenario:max_turns]] go"
    assert Path(handoff["text_path"]).read_text() == handoff["text"]
    assert handoff["cost_usd"] == pytest.approx(0.01)
    assert result["invocation_cost_usd"] == pytest.approx(0.02)  # both calls
    assert final["session_cost_usd"] == pytest.approx(0.02)  # next baseline
    order = kinds(events)
    assert order[-3:] == ["handoff_started", "handoff_finished", "finished"]
    assert events[-2]["detail"]["status"] == "written"

    # The handoff resumed the same session, tool-free, for one bounded turn.
    first, second = argv_files(fake_claude_state)
    assert flag(second, "--resume") == started["session_id"]
    assert flag(second, "--tools") == ""
    assert flag(second, "--max-turns") == "1"
    assert flag(second, "--allowedTools") in (None, "")
    assert "--mcp-config" not in second or "jake" not in (
        flag(second, "--mcp-config") or ""
    )
    assert flag(first, "--tools") == "Bash"
    env = json.loads((fake_claude_state / "env-2.json").read_text())
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"


async def test_a_handoff_is_skipped_when_too_little_time_is_left(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:max_turns]] go", timeout_seconds=5
        )
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    assert final["termination_reason"] == "max_turns"
    handoff = final["result"]["handoff"]
    assert handoff["status"] == "skipped"
    assert "timeout" in handoff["error"]
    assert final["result"]["final_text"] == "partial answer"
    assert len(argv_files(fake_claude_state)) == 1  # no model call


async def test_cancel_during_a_handoff_keeps_max_turns_and_fails_the_handoff(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    env = {**fake_env, "FAKE_CLAUDE_RESUME": "stall"}
    async with mcp_server(env) as session:
        started = await start(session, workdir, "[[scenario:max_turns]] go")
        task_id = started["task_id"]
        await poll_until(
            session, task_id, lambda _, evs: "handoff_started" in kinds(evs)
        )
        await call(session, "claude_cancel", task_id=task_id)
        final, events, _ = await poll_until(session, task_id, terminal)

    assert final["status"] == "failed"
    assert final["termination_reason"] == "max_turns"
    handoff = final["result"]["handoff"]
    assert handoff["status"] == "failed"
    assert "cancelled" in handoff["error"]
    assert final["result"]["final_text"] == "partial answer"
    assert final["result"]["resumable"] is True


@pytest.mark.parametrize(
    ("bend", "said"),
    [
        ("omit", "matching init"),
        ("no_tools", "session and tools"),
        ("no_session", "session and tools"),
        ("wrong_result", "reported session"),
    ],
)
async def test_a_handoff_without_a_verified_session_is_not_written(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    workdir: Path,
    bend: str,
    said: str,
) -> None:
    env = {**fake_env, "FAKE_CLAUDE_RESUME_INIT": bend}
    async with mcp_server(env) as session:
        started = await start(session, workdir, "[[scenario:max_turns]] go")
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    assert final["termination_reason"] == "max_turns"
    handoff = final["result"]["handoff"]
    assert handoff["status"] == "failed", handoff
    assert said in handoff["error"]
    assert final["result"]["final_text"] == "partial answer"


async def test_a_tool_call_during_a_handoff_fails_the_handoff(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    """A guard only: the handoff session exposes no tools, so a real model
    has nothing to call. A fake that calls one anyway must not get a pass."""
    env = {**fake_env, "FAKE_CLAUDE_RESUME": "tool"}
    async with mcp_server(env) as session:
        started = await start(session, workdir, "[[scenario:max_turns]] go")
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    assert final["termination_reason"] == "max_turns"
    handoff = final["result"]["handoff"]
    assert handoff["status"] == "failed"
    assert "tool call" in handoff["error"]


async def test_permission_denials_are_structured_and_resume_uses_its_own_rules(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,
) -> None:
    """Plumbing only: the fake reports denials the way the CLI does; whether
    the real runtime refuses these forms is
    ``test_live_scoped_write_and_compound_bash_denials_are_reported``."""
    rules = ["Write(./allowed/**)", "Bash(touch approved-*)"]
    async with mcp_server(fake_env) as session:
        started = await start(
            session,
            workdir,
            "[[scenario:deny]] go",
            tools=["Bash", "Write"],
            allowed_tools=rules,
        )
        final, events, _ = await poll_until(session, started["task_id"], terminal)
        resumed = await start(
            session,
            workdir,
            "carry on",
            resume_task_id=started["task_id"],
            tools=["Bash"],
            allowed_tools=["Bash(touch approved-*)"],
        )
        await poll_until(session, resumed["task_id"], terminal)

    assert final["status"] == "completed"
    assert final["permission_denials_total"] == 3
    denials = final["permission_denials"]
    assert [(d["tool"], d["requested"]) for d in denials] == [
        ("Write", "outside/b.txt"),
        ("Bash", "cd sub && touch approved-2"),
        ("Bash", "python3 -c \"open('py.txt', 'w')\""),
    ]
    full = len("Permission to use Bash has been denied." + " pad" * 200)
    for d in denials:
        assert d["runtime_message"].startswith(f"Permission to use {d['tool']} ")
        assert len(d["runtime_message"]) == 500
        assert d["runtime_message_truncated"] is True
        assert d["runtime_message_chars"] in (full, full + 1)  # Bash/Write
    by_seq = {e["seq"]: e for e in events}
    assert all(by_seq[d["event_seq"]]["kind"] == "permission_denied" for d in denials)
    assert final["limits"]["allowed_tools"] == rules
    assert final["limits"]["permission_mode"] == "dontAsk"
    denied = [e for e in events if e["kind"] == "permission_denied"]
    assert [e["tool_use_id"] for e in denied] == [d["tool_use_id"] for d in denials]

    # A resume runs under the rules its own request gave, not the parent's.
    first, second = argv_files(fake_claude_state)
    assert flag(first, "--allowedTools") == ",".join(rules)
    assert flag(second, "--resume") == started["session_id"]
    assert flag(second, "--allowedTools") == "Bash(touch approved-*)"


async def test_claude_list_pages_newest_first_and_reports_bad_records(
    mcp_server: ServerFactory, fake_env: dict[str, str], mcp_home: Path, workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        ids = []
        for brief in ("first task", "\n  second task\nmore detail", "x" * 300):
            started = await start(session, workdir, brief)
            await poll_until(session, started["task_id"], terminal)
            ids.append(started["task_id"])
        runs = runs_dir(mcp_home)
        bad = "20000101T000000Z-" + "0" * 32  # sorts oldest
        (runs / bad).mkdir()
        (runs / bad / "spec.json").write_text("{not json")
        spec_path = runs / ids[0] / "spec.json"
        legacy = json.loads(spec_path.read_text())
        legacy["schema_version"] = 1
        spec_path.write_text(json.dumps(legacy))

        page1 = await call(session, "claude_list", limit=2)
        page2 = await call(session, "claude_list", limit=2, before=page1["next_before"])
        failed = await call(session, "claude_list", status="failed")
        bad_cursor = await call_error(session, "claude_list", before="../etc")
        bad_limit = await call_error(session, "claude_list", limit=0)

    # Newest first to the second; IDs from the same second sort by ID.
    order = sorted(ids, reverse=True)
    listed = page1["tasks"] + page2["tasks"]
    assert [t["task_id"] for t in listed] == order
    assert page1["next_before"] == order[1]
    labels = {t["task_id"]: t["label"] for t in listed}
    assert labels == {
        ids[0]: "first task",
        ids[1]: "second task",
        ids[2]: "x" * 119 + "…",
    }
    first = next(t for t in listed if t["task_id"] == ids[0])
    assert first["status"] == "completed" and first["resumable"] is True
    assert first["legacy_run"] is True
    assert [u["task_id"] for u in page2["unreadable"]] == [bad]
    assert page2["next_before"] is None
    assert failed["tasks"] == [] and failed["next_before"] is None
    assert bad_cursor["code"] == "invalid_argument"
    assert bad_limit["code"] == "invalid_argument"


# --- steering ---------------------------------------------------------------------


async def test_steering_mid_turn_is_delivered_into_the_same_conversation(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:slow_tool:4]] go")
        task_id = started["task_id"]
        _, _, cursor = await poll_until(
            session, task_id, lambda _, evs: "tool_use" in kinds(evs)
        )
        sent = await call(
            session,
            "claude_steer",
            task_id=task_id,
            message="also say BANANA",
            message_id="m-1",
        )
        retried = await call(
            session,
            "claude_steer",
            task_id=task_id,
            message="also say BANANA",
            message_id="m-1",
        )
        conflict = await call_error(
            session,
            "claude_steer",
            task_id=task_id,
            message="something else",
            message_id="m-1",
        )
        final, events, _ = await poll_until(session, task_id, terminal, cursor=cursor)
        late_retry = await call(
            session,
            "claude_steer",
            task_id=task_id,
            message="also say BANANA",
            message_id="m-1",
        )
        late_new = await call_error(
            session,
            "claude_steer",
            task_id=task_id,
            message="too late",
            message_id="m-2",
        )

    assert sent["state"] == "queued" and sent["replayed"] is False
    assert retried["replayed"] is True and retried["message_id"] == "m-1"
    assert conflict["code"] == "request_conflict"
    assert conflict["detail"] == {"key": "m-1"}
    assert [
        k for k in kinds(events) if k.startswith("message_") and k != "message_queued"
    ] == ["message_delivered", "message_in_conversation"]
    assert final["result"]["final_text"] == "Done. Heard: also say BANANA"
    assert len([e for e in events if e["kind"] == "turn_result"]) == 1  # same turn
    assert final["messages"][0]["state"] == "in_conversation"
    assert late_retry["state"] == "in_conversation" and late_retry["replayed"] is True
    assert late_new["code"] == "task_not_running"
    assert late_new["detail"]["status"] == "completed"


async def test_a_message_that_arrives_as_a_turn_ends_gets_its_own_turn(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    """Sent right after start: whether it lands mid-turn or after the first
    result, the worker must not finish until the CLI has taken it in."""
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:echo]] first")
        await call(
            session,
            "claude_steer",
            task_id=started["task_id"],
            message="second",
            message_id="s",
        )
        final, events, _ = await poll_until(session, started["task_id"], terminal)

    assert final["status"] == "completed"
    assert final["messages"][0]["state"] == "in_conversation"
    assert final["result"]["final_text"] == "ECHO: second"
    assert len([e for e in events if e["kind"] == "turn_result"]) == 2
    # Two turns: tokens add up, cost is the session total's growth, and the
    # telemetry file agrees with the status (the CLI reports cost cumulatively).
    assert final["usage"]["input_tokens"] == 20
    assert final["invocation_cost_usd"] == pytest.approx(0.02)
    assert final["session_cost_usd"] == pytest.approx(0.02)
    telemetry = json.loads((Path(final["run_dir"]) / "telemetry.json").read_text())
    costs = [call["usage"]["estimated_cost_usd"] for call in telemetry["calls"]]
    assert costs == [pytest.approx(0.01), pytest.approx(0.01)]


async def test_steer_rejects_bad_arguments(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    cases: list[tuple[dict[str, Any], str]] = [
        ({"message": "", "message_id": "a"}, "message"),
        ({"message": "   ", "message_id": "a"}, "message"),
        ({"message": "x" * 50_001, "message_id": "a"}, "message"),
        ({"message": "x", "message_id": "has space"}, "message_id"),
        ({"message": "x", "message_id": "a/b"}, "message_id"),
        ({"message": "x", "message_id": "x" * 129}, "message_id"),
    ]
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:slow_tool:2]] go")
        for arguments, argument in cases:
            error = await call_error(
                session, "claude_steer", task_id=started["task_id"], **arguments
            )
            assert error["code"] == "invalid_argument", arguments
            assert error["detail"] == {"argument": argument}, arguments
        unknown = await call_error(
            session, "claude_steer", task_id="nope", message="x", message_id="a"
        )
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    assert unknown["code"] == "unknown_task"
    assert final["messages_total"] == 0  # nothing invalid was stored


# --- clarification ----------------------------------------------------------------


async def test_clarification_round_trip_resumes_the_same_task(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:ask:Which colour?]] go", allow_questions=True
        )
        task_id = started["task_id"]
        waiting, events, cursor = await poll_until(
            session, task_id, lambda r, _: r["status"] == "input_required"
        )
        status = await call(session, "claude_status", task_id=task_id)
        request = waiting["pending_inputs"][0]
        request_id = request["request_id"]
        unknown = await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={"q9-nope": {"answer": "x"}},
        )
        answered = await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={request_id: {"answer": "green"}},
        )
        duplicate = await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={request_id: {"answer": "green"}},
        )
        conflict = await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={request_id: {"answer": "red"}},
        )
        final, more, _ = await poll_until(session, task_id, terminal, cursor=cursor)

    assert waiting["wake_reason"] in ("events", "input_required")
    assert status["status"] == "input_required"
    assert request["question"] == "Which colour?"
    assert request["choices"] == ["red", "green"]
    assert waiting["status_message"] == "Waiting for an answer: Which colour?"
    assert "input_requested" in kinds(events)
    assert unknown["results"] == [
        {"request_id": "q9-nope", "outcome": "unknown", "state": None, "reason": None}
    ]
    assert answered["results"][0]["outcome"] == "accepted"
    # The snapshot is coherent at once: nothing pending, no longer waiting.
    assert answered["pending_inputs"] == []
    assert answered["task_status"] in ("working", "completed")
    assert duplicate["results"][0]["outcome"] == "duplicate"
    assert conflict["results"][0]["outcome"] == "conflict"
    assert final["status"] == "completed"
    assert final["result"]["final_text"] == "Answer was: Your parent answered: green"
    assert final["pending_inputs"] == []
    assert "input_answered" in kinds(more)
    assert "Waiting for an answer" not in (final["status_message"] or "")
    assert started["task_id"] == final["task_id"]  # same task, same conversation


async def test_without_allow_questions_the_worker_has_no_way_to_ask(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:ask]] go")
        final, events, _ = await poll_until(session, started["task_id"], terminal)

    assert "input_requested" not in kinds(events)
    assert final["status"] == "completed"
    assert "Server 'jake' not found" in final["result"]["final_text"]


async def test_two_pending_questions_need_both_answers(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:ask_twice]] go", allow_questions=True
        )
        task_id = started["task_id"]
        waiting, _, _ = await poll_until(
            session, task_id, lambda r, _: len(r["pending_inputs"]) == 2
        )
        by_question = {
            r["question"]: r["request_id"] for r in waiting["pending_inputs"]
        }
        partial = await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={by_question["first?"]: {"answer": "one"}},
        )
        await anyio.sleep(1)
        still = await call(session, "claude_poll", task_id=task_id, wait_seconds=0)
        await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={by_question["second?"]: {"action": "decline"}},
        )
        final, _, _ = await poll_until(session, task_id, terminal)

    assert len(partial["pending_inputs"]) == 1
    assert still["status"] == "input_required"
    assert [r["question"] for r in still["pending_inputs"]] == ["second?"]
    assert final["result"]["final_text"] == (
        "first=Your parent answered: one second=Your parent declined to answer. "
        "Continue with your best judgement and say in your final message what you assumed."
    )


async def test_an_abandoned_question_is_withdrawn_and_the_other_stays_pending(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:ask_one_abandoned]] go", allow_questions=True
        )
        task_id = started["task_id"]
        waiting, events, cursor = await poll_until(
            session, task_id, lambda _, evs: "input_expired" in kinds(evs)
        )
        expired = next(e for e in events if e["kind"] == "input_expired")
        status = await call(session, "claude_status", task_id=task_id)
        kept = waiting["pending_inputs"]
        await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={kept[0]["request_id"]: {"answer": "yes"}},
        )
        final, _, _ = await poll_until(session, task_id, terminal, cursor=cursor)

    assert "withdrawn" in expired["detail"]["reason"]
    assert [r["question"] for r in kept] == ["kept?"]
    assert waiting["status"] == "input_required"  # the other question is open
    assert status["status"] == "input_required"
    assert final["status"] == "completed"
    assert final["result"]["final_text"] == (
        "abandoned=ERROR: the tool call was cancelled kept=Your parent answered: yes"
    )


async def test_a_malformed_question_is_a_tool_error_not_a_pending_request(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:bad_ask]] go", allow_questions=True
        )
        final, events, _ = await poll_until(session, started["task_id"], terminal)

    assert "input_requested" not in kinds(events)
    assert (
        final["result"]["final_text"]
        == "Tool said: `question` must be a non-empty string."
    )


async def test_update_rejects_malformed_responses(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    cases: list[dict[str, Any]] = [
        {},
        {"q1-x": {"action": "accept"}},
        {"q1-x": {"answer": "   "}},
        {"q1-x": {"action": "decline", "answer": "no"}},
        {"q1-x": {"action": "maybe"}},
        {"q1-x": {"answer": "x" * 20_001}},
    ]
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:echo]] x")
        for responses in cases:
            error = await call_error(
                session,
                "claude_update",
                task_id=started["task_id"],
                responses=responses,
            )
            assert error["code"] == "invalid_argument", responses
        await poll_until(session, started["task_id"], terminal)


async def test_timeout_while_waiting_for_an_answer_expires_the_request(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session,
            workdir,
            "[[scenario:ask:Proceed?]] go",
            timeout_seconds=4,
            allow_questions=True,
        )
        task_id = started["task_id"]
        waiting, _, _ = await poll_until(
            session, task_id, lambda r, _: r["status"] == "input_required"
        )
        request_id = waiting["pending_inputs"][0]["request_id"]
        final, events, _ = await poll_until(session, task_id, terminal)
        stale = await call(
            session,
            "claude_update",
            task_id=task_id,
            responses={request_id: {"answer": "yes"}},
        )

    assert final["status"] == "failed"
    assert final["termination_reason"] == "timeout"
    assert final["pending_inputs"] == []
    assert "input_expired" in kinds(events)
    assert stale["results"][0]["outcome"] == "not_pending"
    assert stale["results"][0]["state"] == "expired"


async def test_max_turns_bounds_the_whole_task_across_steered_prompts(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    """The CLI's own turn count restarts with every prompt, so a message that
    starts a new turn must not buy a fresh max_turns."""
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:slow_echo:2]] first", max_turns=3
        )
        task_id = started["task_id"]
        await poll_until(session, task_id, lambda _, evs: "tool_use" in kinds(evs))
        await call(
            session, "claude_steer", task_id=task_id, message="more", message_id="m"
        )
        final, events, _ = await poll_until(session, task_id, terminal)

    assert final["status"] == "failed"
    assert final["termination_reason"] == "max_turns"
    assert "max_turns=3" in final["result"]["error"]
    assert final["result"]["final_text"] == "ECHO: [[scenario:slow_echo:2]] first"
    assert final["messages"][0]["state"] == "in_conversation"
    assert final["turns_used"] == 4  # the one past the limit stopped it
    near = [e for e in events if e["kind"] == "notice"]
    assert near and near[0]["detail"] == {"turns_used": 2, "max_turns": 3}
    # Stopped mid-prompt: that prompt's cost was never reported, so the
    # budget left is unknown and no handoff call is made.
    handoff = final["result"]["handoff"]
    assert handoff["status"] == "skipped"
    assert "budget is unknown" in handoff["error"]


# --- idempotent start --------------------------------------------------------------


async def test_request_id_replays_the_original_task_and_rejects_a_changed_request(
    mcp_server: ServerFactory, fake_env: dict[str, str], mcp_home: Path, workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        first = await start(
            session, workdir, "[[scenario:echo]] once", request_id="req-1"
        )
        again = await start(
            session, workdir, "[[scenario:echo]] once", request_id="req-1"
        )
        conflict = await call_error(
            session,
            "claude_start",
            brief="[[scenario:echo]] different",
            working_directory=str(workdir),
            request_id="req-1",
            **FAST,
            timeout_seconds=60,
        )
        await poll_until(session, first["task_id"], terminal)
        after = await start(
            session, workdir, "[[scenario:echo]] once", request_id="req-1"
        )
        bad = await call_error(
            session,
            "claude_start",
            brief="x",
            working_directory=str(workdir),
            request_id="a b",
        )

    assert first["replayed"] is False and first["request_id"] == "req-1"
    assert again["replayed"] is True and again["task_id"] == first["task_id"]
    assert again["session_id"] == first["session_id"]
    assert conflict["code"] == "request_conflict"
    assert conflict["detail"] == {"request_id": "req-1", "task_id": first["task_id"]}
    assert after["task_id"] == first["task_id"] and after["status"] == "completed"
    assert bad["code"] == "invalid_argument" and bad["detail"] == {
        "argument": "request_id"
    }
    task_dirs = [p for p in runs_dir(mcp_home).iterdir() if not p.name.startswith(".")]
    assert len(task_dirs) == 1


async def test_a_retry_finds_its_task_even_at_capacity(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        one = await start(
            session, workdir, "[[scenario:hang:31001]] a", request_id="r-a"
        )
        two = await start(session, workdir, "[[scenario:hang:31002]] b")
        full = await call_error(
            session, "claude_start", brief="c", working_directory=str(workdir), **FAST
        )
        retry = await start(
            session, workdir, "[[scenario:hang:31001]] a", request_id="r-a"
        )
        for task in (one, two):
            await call(session, "claude_cancel", task_id=task["task_id"])

    assert full["code"] == "capacity_exceeded"
    assert retry["task_id"] == one["task_id"] and retry["replayed"] is True
    assert sleepers("31001") == [] and sleepers("31002") == []


# --- resume --------------------------------------------------------------------------


async def test_resume_continues_the_conversation_as_a_new_task_with_its_own_grants(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,
) -> None:
    async with mcp_server(fake_env) as session:
        parent = await start(
            session,
            workdir,
            "[[scenario:echo]] remember me",
            tools=["Bash", "Edit"],
            allowed_tools=["Edit", "Bash(git status)"],
        )
        await poll_until(session, parent["task_id"], terminal)
        child = await call(
            session,
            "claude_start",
            brief="what did I say first?",
            resume_task_id=parent["task_id"],
            tools=["Read"],
            **FAST,
            timeout_seconds=60,
        )
        final, _, _ = await poll_until(session, child["task_id"], terminal)
        parent_after = await call(session, "claude_status", task_id=parent["task_id"])

    assert child["task_id"] != parent["task_id"]
    assert child["session_id"] == parent["session_id"]
    assert child["resumed_from"] == parent["task_id"]
    assert final["resumed_from"] == parent["task_id"]
    assert final["result"]["final_text"] == "REMEMBERED: [[scenario:echo]] remember me"
    assert final["invocation_cost_usd"] == pytest.approx(0.01)  # 0.02 total - 0.01
    assert final["usage"]["estimated_cost_usd"] == pytest.approx(0.01)
    assert final["session_cost_usd"] == pytest.approx(0.02)
    assert final["invocation_cost_exact"] is True
    assert parent_after["status"] == "completed"  # untouched
    first_argv, resume_argv = argv_files(fake_claude_state)
    assert flag(first_argv, "--tools") == "Bash,Edit"
    assert flag(first_argv, "--allowedTools") == "Edit,Bash(git status)"
    assert flag(resume_argv, "--resume") == parent["session_id"]
    assert flag(resume_argv, "--session-id") is None
    assert flag(resume_argv, "--tools") == "Read"  # nothing inherited
    assert flag(resume_argv, "--allowedTools") == "Read"  # nor are its rules
    assert final["limits"]["allowed_tools"] == ["Read"]


async def test_resume_refuses_busy_sessions_and_mismatched_directories(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    async with mcp_server(fake_env) as session:
        running = await start(session, workdir, "[[scenario:hang:31003]] a")
        busy_parent = await call_error(
            session,
            "claude_start",
            brief="x",
            resume_task_id=running["task_id"],
            **FAST,
        )
        await call(session, "claude_cancel", task_id=running["task_id"])

        done = await start(session, workdir, "[[scenario:echo]] b")
        await poll_until(session, done["task_id"], terminal)
        wrong_dir = await call_error(
            session,
            "claude_start",
            brief="x",
            resume_task_id=done["task_id"],
            working_directory=str(elsewhere),
            **FAST,
        )
        held = await call(
            session,
            "claude_start",
            brief="[[hold]] keep going",
            resume_task_id=done["task_id"],
            **FAST,
            timeout_seconds=60,
        )
        await poll_until(
            session, held["task_id"], lambda _, evs: "assistant_text" in kinds(evs)
        )
        busy_session = await call_error(
            session, "claude_start", brief="x", resume_task_id=done["task_id"], **FAST
        )
        await call(session, "claude_cancel", task_id=held["task_id"])
        unknown = await call_error(
            session,
            "claude_start",
            brief="x",
            resume_task_id="20261008T031500Z-1a2b3c4d",
            **FAST,
        )
        no_dir = await call_error(session, "claude_start", brief="x", **FAST)

    assert busy_parent["code"] == "session_busy"
    assert busy_parent["detail"]["task_id"] == running["task_id"]
    assert wrong_dir["code"] == "resume_unavailable"
    assert busy_session["code"] == "session_busy"
    assert busy_session["detail"] == {
        "session_id": done["session_id"],
        "task_id": held["task_id"],
    }
    assert unknown["code"] == "unknown_task"
    assert no_dir["code"] == "invalid_argument"
    assert no_dir["detail"] == {"argument": "working_directory"}


async def test_only_the_latest_task_in_a_session_can_be_resumed(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        first = await start(session, workdir, "[[scenario:echo]] a")
        await poll_until(session, first["task_id"], terminal)
        second = await call(
            session, "claude_start", brief="b", resume_task_id=first["task_id"], **FAST
        )
        await poll_until(session, second["task_id"], terminal)
        stale = await call_error(
            session, "claude_start", brief="c", resume_task_id=first["task_id"], **FAST
        )
        third = await call(
            session, "claude_start", brief="c", resume_task_id=second["task_id"], **FAST
        )
        final, _, _ = await poll_until(session, third["task_id"], terminal)

    assert stale["code"] == "resume_unavailable"
    assert stale["detail"] == {
        "task_id": first["task_id"],
        "latest_task_id": second["task_id"],
    }
    # The baseline is the head's session total, so B's spend is not C's.
    assert final["session_cost_usd"] == pytest.approx(0.03)
    assert final["invocation_cost_usd"] == pytest.approx(0.01)


async def test_a_resume_of_a_vanished_session_fails_instead_of_starting_afresh(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,
) -> None:
    async with mcp_server(fake_env) as session:
        parent = await start(session, workdir, "[[scenario:echo]] a")
        await poll_until(session, parent["task_id"], terminal)
        (fake_claude_state / f"{parent['session_id']}.json").unlink()
        child = await call(
            session,
            "claude_start",
            brief="go on",
            resume_task_id=parent["task_id"],
            **FAST,
        )
        final, events, _ = await poll_until(session, child["task_id"], terminal)
        again = await call_error(
            session, "claude_start", brief="x", resume_task_id=child["task_id"], **FAST
        )

    assert final["status"] == "failed"
    assert final["termination_reason"] == "resume_failed"
    assert parent["session_id"] in final["result"]["error"]
    assert final["result"]["resumable"] is False
    assert "assistant_text" not in kinds(events)  # no fresh conversation ran
    assert again["code"] == "resume_unavailable"


async def test_a_session_id_mismatch_stops_the_conversation(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    env = {**fake_env, "FAKE_CLAUDE_WRONG_SESSION": "1"}
    async with mcp_server(env) as session:
        started = await start(session, workdir, "[[scenario:echo]] a")
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    assert final["status"] == "failed"
    assert final["termination_reason"] == "sdk_error"
    assert "not the expected" in final["result"]["error"]


# --- tool exposure and permission rules -------------------------------------------


async def test_a_narrow_bash_rule_reaches_the_cli_without_a_blanket_approval(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,
) -> None:
    rules = ["Read", "Bash(git status)", "Bash(uv run pytest *)"]
    async with mcp_server(fake_env) as session:
        started = await start(
            session,
            workdir,
            "[[scenario:echo]] narrow",
            tools=["Read", "Bash"],
            allowed_tools=rules,
        )
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    assert final["status"] == "completed"
    assert final["limits"]["tools"] == ["Read", "Bash"]
    assert final["limits"]["allowed_tools"] == rules
    (argv,) = argv_files(fake_claude_state)
    assert flag(argv, "--tools") == "Read,Bash"
    approved = flag(argv, "--allowedTools")
    assert approved == "Read,Bash(git status),Bash(uv run pytest *)"
    assert approved is not None and "Bash" not in approved.split(",")
    assert flag(argv, "--permission-mode") == "dontAsk"


async def test_tools_outside_the_old_list_are_requested_from_the_runtime(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    workdir: Path,
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session,
            workdir,
            "[[scenario:echo]] new tools",
            tools=["NotebookEdit", "Monitor", "Agent"],
        )
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    # The session reported them (Agent as Task, as the real CLI does).
    assert final["status"] == "completed", final["result"]
    (argv,) = argv_files(fake_claude_state)
    assert flag(argv, "--tools") == "NotebookEdit,Monitor,Agent"
    assert flag(argv, "--allowedTools") == "NotebookEdit,Monitor,Agent"


async def test_a_tool_the_session_does_not_provide_fails_the_task_by_name(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    env = {**fake_env, "FAKE_CLAUDE_UNAVAILABLE_TOOLS": "Artifact"}
    async with mcp_server(env) as session:
        started = await start(
            session,
            workdir,
            "[[scenario:slow_tool:30]] never runs",
            tools=["Read", "Artifact"],
        )
        final, events, _ = await poll_until(
            session, started["task_id"], terminal, timeout=20
        )

    assert final["status"] == "failed"
    assert final["termination_reason"] == "tools_unavailable"
    error = final["result"]["error"]
    assert "Artifact" in error and "exposed only Read" in error
    assert "tool_use" not in kinds(events)  # stopped at the session's start


# --- old run directories -----------------------------------------------------------


def _legacy_run(runs: Path, workdir: Path, *, session: str) -> str:
    """A run directory exactly as the previous release wrote it."""
    task_id = "20261008T031500Z-1a2b3c4d"
    run_dir = runs / task_id
    run_dir.mkdir(parents=True)
    (run_dir / "brief.md").write_text("old brief")
    (run_dir / "spec.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "created_at": "2026-10-08T03:15:00Z",
                "working_directory": str(workdir),
                "tools": ["Read"],
                "model": "claude-haiku-4-5",
                "effort": None,
                "max_turns": 3,
                "timeout_seconds": 30.0,
                "max_budget_usd": 0.1,
            }
        )
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "status": "completed",
                "run_dir": str(run_dir),
                "created_at": "2026-10-08T03:15:00Z",
                "started_at": "2026-10-08T03:15:01Z",
                "finished_at": "2026-10-08T03:15:09Z",
                "pid": 1234,
                "pgid": 1234,
                "termination_reason": "finished",
                "final_text": "old answer",
                "usage": None,
                "error": None,
            }
        )
    )
    (run_dir / "transcript.jsonl").write_text(
        json.dumps(
            {
                "type": "SystemMessage",
                "subtype": "init",
                "data": {"session_id": session},
            }
        )
        + "\n"
    )
    return task_id


async def test_runs_from_the_previous_release_stay_readable_and_resumable(
    mcp_server: ServerFactory,
    fake_env: dict[str, str],
    fake_claude_state: Path,
    mcp_home: Path,
    workdir: Path,
) -> None:
    session_id = "33333333-3333-4333-8333-333333333333"
    (fake_claude_state / f"{session_id}.json").write_text(
        json.dumps({"user_messages": ["the old brief"], "cost": 0.05})
    )
    task_id = _legacy_run(runs_dir(mcp_home), workdir, session=session_id)
    async with mcp_server(fake_env) as session:
        status = await call(session, "claude_status", task_id=task_id)
        polled = await call(session, "claude_poll", task_id=task_id, wait_seconds=0)
        page = await call(session, "claude_transcript", task_id=task_id)
        steer = await call_error(
            session, "claude_steer", task_id=task_id, message="hi", message_id="m"
        )
        child = await call(
            session, "claude_start", brief="continue", resume_task_id=task_id, **FAST
        )
        final, _, _ = await poll_until(session, child["task_id"], terminal)

    assert status["status"] == "completed" and status["final_text"] == "old answer"
    assert polled["legacy_run"] is True and polled["events"] == []
    assert polled["result"]["final_text"] == "old answer"
    assert polled["result"]["session_id"] is None
    assert page["legacy_run"] is True and page["events"] == []
    assert steer["code"] == "task_not_running"
    assert child["session_id"] == session_id  # recovered from the old transcript
    assert final["result"]["final_text"] == "REMEMBERED: the old brief"
    assert final["invocation_cost_usd"] is None or final["invocation_cost_usd"] >= 0


# --- lifecycle: real processes, checked in the process table ------------------------


async def test_cancel_reports_request_and_stop_and_settles_queued_messages(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:hang:31004]] go")
        task_id = started["task_id"]
        await poll_until(session, task_id, lambda _, evs: "tool_use" in kinds(evs))
        deadline = time.monotonic() + 10
        while not sleepers("31004") and time.monotonic() < deadline:
            await anyio.sleep(0.1)
        assert sleepers("31004"), "the fake tool shell never started"
        await anyio.sleep(1)  # let the worker record the shell's group
        cancelled = await call(session, "claude_cancel", task_id=task_id)
        final, events, _ = await poll_until(session, task_id, terminal)

    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested_at"] is not None
    assert "cancel_requested" in kinds(events) and kinds(events)[-1] == "finished"
    assert sleepers("31004") == []  # the tool shell in its own session is gone
    assert final["result"]["outcome"] == "cancelled"
    assert final["result"]["last_narration"] == "Starting a long tool."


async def test_a_message_queued_to_a_wedged_task_is_reported_not_delivered(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(session, workdir, "[[scenario:hang:31005]] go")
        task_id = started["task_id"]
        await poll_until(session, task_id, lambda _, evs: "tool_use" in kinds(evs))
        # The worker writes it (delivered); the hung CLI never takes it in.
        await call(
            session, "claude_steer", task_id=task_id, message="hello", message_id="w"
        )
        await poll_until(
            session, task_id, lambda _, evs: "message_delivered" in kinds(evs)
        )
        await call(session, "claude_cancel", task_id=task_id)
        final = await call(session, "claude_poll", task_id=task_id, wait_seconds=0)

    message = final["messages"][0]
    assert message["state"] == "delivered"  # honest: written, never confirmed
    assert "never confirmed adding it to the conversation" in message["reason"]
    assert sleepers("31005") == []


async def test_timeout_ends_the_task_and_its_tool_shells(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    async with mcp_server(fake_env) as session:
        started = await start(
            session, workdir, "[[scenario:hang:31006]] go", timeout_seconds=4
        )
        final, _, _ = await poll_until(session, started["task_id"], terminal)

    assert final["status"] == "failed"
    assert final["termination_reason"] == "timeout"
    deadline = time.monotonic() + 5
    while sleepers("31006") and time.monotonic() < deadline:
        await anyio.sleep(0.1)
    assert sleepers("31006") == []


async def test_the_worker_outlives_a_killed_server_and_a_new_one_carries_on(
    mcp_server: ServerFactory, fake_env: dict[str, str], workdir: Path
) -> None:
    server_pid: int | None = None
    started: dict[str, Any] | None = None
    cursor: str | None = None
    try:
        async with mcp_server(fake_env) as session:
            ping = await call(session, "ping")
            server_pid = int(ping["pid"])
            started = await start(session, workdir, "[[scenario:slow_tool:5]] go")
            _, _, cursor = await poll_until(
                session, started["task_id"], lambda _, evs: "tool_use" in kinds(evs)
            )
            await call(
                session,
                "claude_steer",
                task_id=started["task_id"],
                message="after restart",
                message_id="r",
            )
            os.kill(server_pid, signal.SIGKILL)
            await anyio.sleep(0.5)
    except Exception:  # the client's teardown of a SIGKILLed server may complain
        if server_pid is None or started is None:
            raise
    assert started is not None and cursor is not None
    record = json.loads((Path(started["run_dir"]) / "process.json").read_text())
    worker = process_table().get(record["pid"])
    assert worker is not None and not worker.zombie  # the worker did not die with it

    async with mcp_server(fake_env) as session:
        final, events, _ = await poll_until(
            session, started["task_id"], terminal, cursor=cursor
        )

    assert final["status"] == "completed"
    assert final["result"]["final_text"] == "Done. Heard: after restart"
    assert [e["seq"] for e in events] == sorted({e["seq"] for e in events})
    assert events[0]["seq"] > 3  # the cursor from the dead server still works


async def test_cancelling_a_poll_leaves_the_task_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-process service: cancelling a waiting poll ends only the wait."""
    from conftest import install_fake_claude

    from jake_tools.claude_runs.control import RunDefaults, StartRequest
    from jake_tools.claude_runs.service import TaskService

    bin_dir, state = install_fake_claude(tmp_path / "fake")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLAUDE_STATE", str(state))
    workdir = tmp_path / "work"
    workdir.mkdir()
    service = TaskService(
        tmp_path / "runs",
        RunDefaults(model="fake", max_turns=3, timeout_seconds=60, max_budget_usd=1),
    )
    started = service.start(
        StartRequest(brief="[[scenario:slow_tool:4]] go", working_directory=workdir)
    )
    cursor = None
    deadline = time.monotonic() + 30
    while True:
        page = await service.poll(started.task_id, cursor=cursor, wait_seconds=5)
        cursor = page.next_cursor
        if any(e.kind == "tool_use" for e in page.events):
            break
        assert time.monotonic() < deadline

    with anyio.move_on_after(0.5) as scope:
        await service.poll(started.task_id, cursor=cursor, wait_seconds=20)
    assert scope.cancelled_caught  # it was waiting, and only the wait ended

    while not service.status(started.task_id).is_terminal:
        assert time.monotonic() < deadline + 30
        await anyio.sleep(0.2)
    assert service.status(started.task_id).status == "completed"
