"""The event log and the inbox, hermetically: cursors, concurrency, races."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jake_tools.claude_runs.events import Cursor, EventLog, InvalidCursorError
from jake_tools.claude_runs.inbox import (
    ConflictError,
    Inbox,
    InboxClosedError,
    InputResponse,
    SteerSubmission,
)
from jake_tools.claude_runs.models import RunSpec
from jake_tools.claude_runs.rundir import create_run_dir, new_task_id, read_state


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    spec = RunSpec(
        task_id=new_task_id(),
        created_at=datetime.now(UTC),
        working_directory=tmp_path,
        model="m",
        max_turns=1,
        timeout_seconds=1,
        max_budget_usd=1,
    )
    return create_run_dir(tmp_path / "runs", spec, "brief")


def page(log: EventLog, cursor: Cursor, limit: int = 10):
    return log.page(cursor, limit=limit, max_text_chars=1000, max_bytes=100_000)


def test_a_partly_written_line_is_invisible_until_complete(run_dir: Path) -> None:
    log = EventLog(run_dir)
    log.append("assistant_text", text="one")
    with log.path.open("ab") as handle:
        handle.write(b'{"seq": 2, "at": "2026-10-18T00:00:00Z", "kind": "assis')

    first = page(log, Cursor.start(run_dir.name))
    assert [e.text for e in first.events] == ["one"]
    assert first.has_more is False
    cursor = Cursor.parse(first.next_cursor, run_dir.name)
    assert page(log, cursor).events == []  # still valid at the torn line


def test_concurrent_appenders_get_unique_contiguous_sequence_numbers(
    run_dir: Path,
) -> None:
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from jake_tools.claude_runs.events import EventLog\n"
        "log = EventLog(Path(sys.argv[1]))\n"
        "for i in range(40):\n"
        "    log.append('notice', text=sys.argv[2] * 300)\n"
    )
    writers = [
        subprocess.Popen([sys.executable, "-c", script, str(run_dir), tag])
        for tag in ("a", "b", "c")
    ]
    for writer in writers:
        assert writer.wait() == 0

    events = page(EventLog(run_dir), Cursor.start(run_dir.name), limit=500).events
    assert [e.seq for e in events] == list(range(1, 121))


def test_cursors_are_bound_to_their_task_and_position(run_dir: Path) -> None:
    log = EventLog(run_dir)
    for index in range(3):
        log.append("notice", text=str(index))
    first = page(log, Cursor.start(run_dir.name), limit=1)
    cursor = Cursor.parse(first.next_cursor, run_dir.name)

    assert [e.text for e in page(log, cursor).events] == ["1", "2"]
    with pytest.raises(InvalidCursorError, match="different task"):
        Cursor.parse(first.next_cursor, new_task_id())
    for bad in ("x", f"{run_dir.name}:0:0", f"{run_dir.name}:a:1", f"{run_dir.name}:1"):
        with pytest.raises(InvalidCursorError):
            page(log, Cursor.parse(bad, run_dir.name))
    with pytest.raises(InvalidCursorError):
        page(log, Cursor(run_dir.name, cursor.seq + 1, cursor.offset))
    with pytest.raises(InvalidCursorError):
        page(log, Cursor(run_dir.name, 9, log.size() + 10))


def test_an_accepted_message_is_never_lost_to_a_close(run_dir: Path) -> None:
    inbox = Inbox(run_dir)
    queued, _ = inbox.submit(SteerSubmission(message_id="a", message="hello"))
    written, _ = inbox.submit(SteerSubmission(message_id="b", message="world"))
    assert inbox.mark_attempted(written) is not None  # the worker began writing b

    inbox.close("the task was cancelled")

    states = {m.message_id: (m.state, m.reason) for m in inbox.messages()}
    assert states["a"] == ("not_delivered", "the task was cancelled")
    assert states["b"][0] == "not_delivered"
    assert "may or may not have reached Claude" in (states["b"][1] or "")
    assert inbox.mark_attempted(queued) is None  # a late worker must not send it
    with pytest.raises(InboxClosedError):
        inbox.submit(SteerSubmission(message_id="c", message="late"))
    replay, replayed = inbox.submit(SteerSubmission(message_id="a", message="hello"))
    assert replayed and replay.state == "not_delivered"  # retries still answered
    with pytest.raises(ConflictError):
        inbox.submit(SteerSubmission(message_id="a", message="changed"))


def test_close_if_idle_waits_for_messages_still_on_their_way(run_dir: Path) -> None:
    inbox = Inbox(run_dir)
    message, _ = inbox.submit(SteerSubmission(message_id="a", message="hi"))
    assert inbox.close_if_idle() is False  # queued
    attempted = inbox.mark_attempted(message)
    assert attempted is not None
    inbox.mark_delivered(attempted)
    assert inbox.close_if_idle() is False  # written, not yet in the conversation
    inbox.mark_in_conversation(message.uuid)
    assert inbox.close_if_idle() is True


def test_input_requests_are_unique_and_settle_once(run_dir: Path) -> None:
    inbox = Inbox(run_dir)
    first = inbox.ask("one?", [])
    second = inbox.ask("two?", ["a", "b"])
    assert first.request_id != second.request_id

    results = inbox.respond({first.request_id: InputResponse(answer="yes")})
    assert results[0].outcome == "accepted"
    assert [r.request_id for r in inbox.pending_inputs()] == [second.request_id]
    inbox.close("timeout")
    late = inbox.respond({second.request_id: InputResponse(action="decline")})
    assert (late[0].outcome, late[0].state) == ("not_pending", "expired")
    assert inbox.respond({"../x": InputResponse(answer="x")})[0].outcome == "unknown"


def test_the_session_head_is_the_latest_by_creation_not_by_id(tmp_path: Path) -> None:
    """Two tasks created in the same second can sort either way by ID."""
    from datetime import timedelta

    from jake_tools.claude_runs.control import _session_head_locked

    runs = tmp_path / "runs"
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    def make(task_id: str, created: datetime) -> Path:
        spec = RunSpec(
            task_id=task_id,
            created_at=created,
            working_directory=tmp_path,
            model="m",
            max_turns=1,
            timeout_seconds=1,
            max_budget_usd=1,
            schema_version=2,
            session_id="s-1",
        )
        return create_run_dir(runs, spec, "b")

    parent = make(f"{stamp}-{'f' * 32}", now)
    later = make(f"{stamp}-{'0' * 32}", now + timedelta(milliseconds=300))

    head = _session_head_locked(runs, "s-1", read_state(parent))

    assert later.name < parent.name  # sorts first by ID...
    assert head == later.name  # ...but was created after


# --- poll progress coherent with the events it returns ----------------------------


def _service_run(tmp_path: Path):
    from jake_tools.claude_runs.control import RunDefaults
    from jake_tools.claude_runs.service import TaskService

    runs = tmp_path / "runs"
    spec = RunSpec(
        task_id=new_task_id(),
        created_at=datetime.now(UTC),
        working_directory=tmp_path,
        model="m",
        max_turns=5,
        timeout_seconds=60,
        max_budget_usd=1,
        schema_version=2,
    )
    run_dir = create_run_dir(runs, spec, "brief")  # no process yet: stays working
    service = TaskService(
        runs, RunDefaults(model="m", max_turns=5, timeout_seconds=60, max_budget_usd=1)
    )
    return service, run_dir


async def test_poll_reports_progress_from_events_newer_than_the_snapshot(
    tmp_path: Path,
) -> None:
    """The worker appends an event before its throttled status write."""
    service, run_dir = _service_run(tmp_path)
    tool = EventLog(run_dir).append(
        "tool_use", tool="Bash", tool_use_id="t1", text="sleep 20 && wc -l x"
    )
    assert read_state(run_dir).status_message is None  # the snapshot lags

    result = await service.poll(run_dir.name, wait_seconds=0)

    assert result.events[-1].kind == "tool_use"
    assert result.status_message == "Bash: sleep 20 && wc -l x"
    assert result.last_activity_at == tool.at


async def test_an_old_cursor_page_never_rolls_progress_back(tmp_path: Path) -> None:
    from jake_tools.claude_runs.rundir import update_state, utc_now

    service, run_dir = _service_run(tmp_path)
    EventLog(run_dir).append("assistant_text", text="old narration")
    later = utc_now()
    update_state(
        run_dir,
        lambda s: s.model_copy(
            update={
                "status_message": "Write: summary.txt",
                "status_message_at": later,
                "last_activity_at": later,
            }
        ),
    )

    result = await service.poll(run_dir.name, wait_seconds=0)  # from the start

    assert [e.text for e in result.events] == ["old narration"]
    assert result.status_message == "Write: summary.txt"
    assert result.last_activity_at == later


async def test_a_question_or_an_end_keeps_its_summary(tmp_path: Path) -> None:
    from jake_tools.claude_runs.rundir import update_state

    service, run_dir = _service_run(tmp_path)
    log = EventLog(run_dir)
    Inbox(run_dir, log).ask("Which colour?", [])
    log.append("assistant_text", text="narration racing the question")

    waiting = await service.poll(run_dir.name, wait_seconds=0)

    update_state(
        run_dir,
        lambda s: s.model_copy(
            update={
                "status": "completed",
                "termination_reason": "finished",
                "status_message": "Finished: green",
            }
        ),
    )
    log.append("assistant_text", text="a late line")
    ended = await service.poll(run_dir.name, wait_seconds=0)

    assert waiting.status == "input_required"
    assert waiting.status_message == "Waiting for an answer: Which colour?"
    assert ended.status_message == "Finished: green"


def test_the_worker_never_writes_an_older_line_over_a_newer_one(tmp_path: Path) -> None:
    from jake_tools.claude_runs.rundir import update_state, utc_now
    from jake_tools.claude_runs.worker import Progress

    _, run_dir = _service_run(tmp_path)
    log = EventLog(run_dir)
    old = log.append("assistant_text", text="before the answer")
    update_state(
        run_dir,
        lambda s: s.model_copy(
            update={
                "status_message": "Answer received; continuing",
                "status_message_at": utc_now(),
            }
        ),
    )
    progress = Progress(run_dir)
    progress.note(old)
    progress.flush()

    assert read_state(run_dir).status_message == "Answer received; continuing"
