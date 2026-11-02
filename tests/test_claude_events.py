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


# --- compact observation: tokens, wake rules, grouping, consistency ---------------


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


def _token_at_start(run_dir: Path) -> str:
    from jake_tools.claude_runs.events import Cursor
    from jake_tools.claude_runs.observe import encode_token

    return encode_token(Cursor.start(run_dir.name))


def _deny(log: EventLog, tool_use_id: str, command: str, **detail: str) -> None:
    log.append(
        "permission_denied",
        tool="Bash",
        tool_use_id=tool_use_id,
        is_error=True,
        text=command,
        detail={"runtime_message": "Permission to use Bash has been denied.", **detail},
    )


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

    assert result.progress == "Bash: sleep 20 && wc -l x"
    assert result.last_activity_at == tool.at
    assert result.running_tool is not None
    assert result.running_tool.tool == "Bash" and result.running_tool.since == tool.at


async def test_an_old_event_never_rolls_progress_back(tmp_path: Path) -> None:
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

    result = await service.poll(
        run_dir.name, token=_token_at_start(run_dir), wait_seconds=0
    )

    assert result.progress == "Write: summary.txt"
    assert result.last_activity_at == later
    assert result.changes is not None and result.changes.activity == 1


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
    assert waiting.progress == "Waiting for an answer: Which colour?"
    assert ended.progress == "Finished: green"


async def test_routine_activity_never_wakes_but_progress_mode_does(
    tmp_path: Path,
) -> None:
    service, run_dir = _service_run(tmp_path)
    log = EventLog(run_dir)
    token = _token_at_start(run_dir)
    log.append("assistant_text", text="working")
    log.append("tool_use", tool="Bash", tool_use_id="t1", text="ls")
    log.append("tool_result", tool_use_id="t1", text="a\nb")
    log.append(
        "notice",
        text="API retry (attempt 2): overloaded",
        detail={"category": "api_retry"},
    )

    attention = await service.poll(run_dir.name, token=token, wait_seconds=0.6)
    progress = await service.poll(
        run_dir.name, token=token, wait_seconds=5, until="progress"
    )

    assert attention.wake_reason == "deadline" and attention.waited_seconds >= 0.5
    assert attention.changes is not None
    assert attention.changes.events == 4 and attention.changes.activity == 3
    assert [w.category for w in attention.changes.warnings] == ["api_retry"]
    assert progress.wake_reason == "progress" and progress.waited_seconds < 1


async def test_denials_group_repeats_and_only_a_new_one_wakes(tmp_path: Path) -> None:
    from jake_tools.claude_runs.models import PermissionDenial
    from jake_tools.claude_runs.rundir import update_state

    service, run_dir = _service_run(tmp_path)
    log = EventLog(run_dir)
    token = _token_at_start(run_dir)
    for index in range(3):
        _deny(log, f"t{index}", "rm -rf build", runtime_reason="mode")
    _deny(log, "t2", "rm -rf build")  # the runtime confirming one already seen
    log.append(
        "permission_denied",
        tool="Bash",
        tool_use_id="t9",
        is_error=True,
        text="sleep 40 && echo",
        detail={
            "source": "inferred_from_tool_error",
            "runtime_message": "Blocked: sleep",
        },
    )

    woke = await service.poll(run_dir.name, token=token, wait_seconds=5)
    assert woke.wake_reason == "denial" and woke.changes is not None
    groups = {(g.requested, g.count, g.source, g.new) for g in woke.changes.denials}
    assert groups == {
        ("rm -rf build", 3, "runtime", True),
        ("sleep 40 && echo", 1, "inferred_from_tool_error", True),
    }
    first = next(g for g in woke.changes.denials if g.requested == "rm -rf build")
    assert (
        first.runtime_reason == "mode" and first.first_seq == 1 and first.last_seq == 3
    )

    # The same refusal again, after the caller saw it: reported, not a wake.
    seen = woke.observation_token
    update_state(
        run_dir,
        lambda s: s.model_copy(
            update={
                "permission_denials": [
                    PermissionDenial(
                        at=datetime.now(UTC),
                        tool="Bash",
                        requested="rm -rf build",
                        event_seq=1,
                    )
                ]
            }
        ),
    )
    _deny(log, "t10", "rm -rf build")
    repeat = await service.poll(run_dir.name, token=seen, wait_seconds=0.5)
    assert repeat.wake_reason == "deadline" and repeat.changes is not None
    assert [(g.count, g.new) for g in repeat.changes.denials] == [(1, False)]


async def test_undeliverable_steering_and_expired_questions_wake(
    tmp_path: Path,
) -> None:
    service, run_dir = _service_run(tmp_path)
    log = EventLog(run_dir)
    token = _token_at_start(run_dir)
    log.append("message_delivered", message_id="m1")
    first = await service.poll(run_dir.name, token=token, wait_seconds=0.4)
    log.append("message_not_delivered", message_id="m2", detail={"reason": "it ended"})
    steering = await service.poll(
        run_dir.name, token=first.observation_token, wait_seconds=5
    )
    log.append("input_expired", request_id="q1", detail={"reason": "withdrawn"})
    expired = await service.poll(
        run_dir.name, token=steering.observation_token, wait_seconds=5
    )

    assert first.wake_reason == "deadline"  # delivery is reported, not a wake
    assert first.changes is not None
    assert [(c.message_id, c.state) for c in first.changes.steering] == [
        ("m1", "delivered")
    ]
    assert steering.wake_reason == "steering" and steering.changes is not None
    assert steering.changes.steering[0].reason == "it ended"
    assert expired.wake_reason == "question" and expired.changes is not None
    assert [(q.request_id, q.state) for q in expired.changes.questions] == [
        ("q1", "expired")
    ]


async def test_reading_consumes_nothing_and_advancing_never_dismisses_a_question(
    tmp_path: Path,
) -> None:
    service, run_dir = _service_run(tmp_path)
    log = EventLog(run_dir)
    token = _token_at_start(run_dir)
    _deny(log, "t1", "touch x")
    Inbox(run_dir, log).ask("Proceed?", ["yes", "no"])

    first = await service.poll(run_dir.name, token=token, wait_seconds=0)
    retry = await service.poll(run_dir.name, token=token, wait_seconds=0)
    advanced = await service.poll(
        run_dir.name, token=first.observation_token, until="terminal", wait_seconds=20
    )
    other_reader = await service.poll(run_dir.name, wait_seconds=0)

    assert first.wake_reason == "question"  # a question outranks the denial
    assert retry.changes == first.changes
    assert retry.observation_token == first.observation_token
    # A newer token, and a terminal-only wait: the open question still returns
    # at once and is still listed, because nothing answered it.
    assert advanced.wake_reason == "question" and advanced.waited_seconds < 1
    assert [q.question for q in advanced.pending_questions] == ["Proceed?"]
    assert advanced.pending_questions[0].choices == ["yes", "no"]
    assert advanced.changes is not None and advanced.changes.events == 0
    assert other_reader.pending_questions_total == 1 and other_reader.changes is None


async def test_a_bad_token_resynchronises_to_a_consistent_snapshot(
    tmp_path: Path,
) -> None:
    from jake_tools.claude_runs.events import Cursor
    from jake_tools.claude_runs.observe import decode_token, encode_token

    service, run_dir = _service_run(tmp_path)
    log = EventLog(run_dir)
    log.append("assistant_text", text="one")
    past_end = encode_token(Cursor(task_id=run_dir.name, seq=9, offset=99_999))

    for token in ("nonsense", past_end, f"{run_dir.name}:1:0"):
        result = await service.poll(run_dir.name, token=token, wait_seconds=20)
        assert result.token_status == "resynchronised"
        assert result.wake_reason == "resynchronised" and result.waited_seconds < 1
        assert result.resync_reason
        # The fresh token is the end of the log: nothing before it is lost
        # to the snapshot, nothing after it is skipped.
        position = decode_token(result.observation_token, run_dir.name)
        assert position == log.end()


async def test_long_backlogs_do_not_stale_the_snapshot_and_overflow_names_its_route(
    tmp_path: Path,
) -> None:
    from jake_tools.claude_runs.observe import CHANGES_SHOWN
    from jake_tools.claude_runs.rundir import update_state

    service, run_dir = _service_run(tmp_path)
    log = EventLog(run_dir)
    token = _token_at_start(run_dir)
    for index in range(2000):
        log.append("tool_result", tool_use_id=f"r{index}", text="x" * 2000)
    for index in range(CHANGES_SHOWN + 5):
        _deny(log, f"d{index}", f"cmd {index}")
    update_state(
        run_dir,
        lambda s: s.model_copy(
            update={"status": "completed", "termination_reason": "finished"}
        ),
    )

    result = await service.poll(run_dir.name, token=token, wait_seconds=0)

    assert result.status == "completed" and result.wake_reason == "terminal"
    assert (
        result.changes is not None and result.changes.events == 2000 + CHANGES_SHOWN + 5
    )
    assert len(result.changes.denials) == CHANGES_SHOWN
    assert result.changes.omitted == 5
    assert result.changes.retrieve is not None
    assert result.changes.retrieve.tool == "claude_transcript"
    assert result.changes.retrieve.arguments["cursor"] == f"{run_dir.name}:1:0"
    assert len(result.model_dump_json()) < 8_000


def test_every_claude_result_reads_back_after_nulls_are_omitted() -> None:
    """Results drop ``None`` fields; a reader must still be able to load them."""
    import typing

    from pydantic import BaseModel

    from jake_tools.claude_runs import models, observe, service

    seen: set[type[BaseModel]] = set()

    def walk(model: type[BaseModel]) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, field in model.model_fields.items():
            nullable = type(None) in typing.get_args(field.annotation)
            assert not (field.is_required() and nullable), f"{model.__name__}.{name}"
            stack = [field.annotation]
            while stack:
                current = stack.pop()
                if isinstance(current, type) and issubclass(current, BaseModel):
                    walk(current)
                stack.extend(typing.get_args(current))

    for model in (
        observe.Observation,
        service.MultiObservation,
        service.TaskView,
        service.ResultText,
        service.CancelResult,
        service.TranscriptPage,
        service.SteerResult,
        service.UpdateResult,
        service.TaskList,
        models.ClaudeStartResult,
    ):
        walk(model)


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


async def test_a_cut_error_names_the_route_that_returns_it_whole(
    tmp_path: Path,
) -> None:
    from jake_tools.claude_runs.rundir import update_state

    service, run_dir = _service_run(tmp_path)
    error = "E" * 5_000
    update_state(
        run_dir,
        lambda s: s.model_copy(
            update={
                "status": "failed",
                "termination_reason": "sdk_error",
                "error": error,
            }
        ),
    )

    result = (await service.poll(run_dir.name, wait_seconds=0)).result
    assert result is not None and result.final_text is None
    assert result.error is not None and len(result.error) < len(error)
    assert result.retrieve is not None
    assert result.retrieve.arguments == {
        "task_id": run_dir.name,
        "view": "result",
        "part": "error",
    }
    chunk = service.result_text(run_dir.name, part="error")
    assert chunk.text == error and chunk.next_offset is None
