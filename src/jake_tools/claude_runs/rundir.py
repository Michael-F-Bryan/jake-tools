"""Run directories: creation, atomic writes, reads.

Everything a task leaves behind lives in ``<runs_dir>/<task_id>/`` (the file
list is in :mod:`jake_tools.claude_runs.models`). This module is the only
place that knows the file names and modes: directories are ``0700``, files
``0600``, and every file that is updated in place is written to a sibling
temp file and renamed, so a reader never sees a torn ``status.json``.

Nothing here knows about processes; that is :mod:`.spawn` and :mod:`.control`.
"""

from __future__ import annotations

import fcntl
import os
import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from .models import (
    TASK_ID_PATTERN,
    GroupRecords,
    ProcessRecord,
    RunResult,
    RunSpec,
    RunState,
)

BRIEF_FILE = "brief.md"
SPEC_FILE = "spec.json"
STATUS_FILE = "status.json"
RESULT_FILE = "result.json"
TRANSCRIPT_FILE = "transcript.jsonl"
TELEMETRY_FILE = "telemetry.json"
STDOUT_FILE = "worker.stdout"
STDERR_FILE = "worker.stderr"
PROCESS_FILE = "process.json"
GROUPS_FILE = "groups.json"
LOCK_FILE = ".status.lock"
WORKER_LOCK_FILE = ".worker.lock"
RUNS_LOCK_FILE = ".runs.lock"

DIR_MODE = 0o700
FILE_MODE = 0o600


class UnknownTaskError(LookupError):
    """``task_id`` is malformed or has no run directory."""

    def __init__(self, task_id: str) -> None:
        super().__init__(f"unknown task {task_id!r}")
        self.task_id = task_id


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_task_id(now: datetime | None = None) -> str:
    """``<UTC basic timestamp>-<8 hex>``, so directory listings sort by creation."""
    stamp = (now or utc_now()).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(4)}"


def ensure_runs_dir(runs_dir: Path) -> Path:
    """Create ``runs_dir`` on demand, private to the user."""
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(runs_dir, DIR_MODE)
    return runs_dir


def task_dir(runs_dir: Path, task_id: str) -> Path:
    """The run directory for ``task_id``; :class:`UnknownTaskError` otherwise.

    The ID is matched against :data:`TASK_ID_PATTERN` before it touches the
    filesystem, so a caller can never name a path outside ``runs_dir``.
    """
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise UnknownTaskError(task_id)
    path = runs_dir / task_id
    if not path.is_dir():
        raise UnknownTaskError(task_id)
    return path


def list_task_dirs(runs_dir: Path) -> list[Path]:
    """Every run directory under ``runs_dir``, oldest first."""
    if not runs_dir.is_dir():
        return []
    return sorted(
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and TASK_ID_PATTERN.fullmatch(path.name)
    )


def create_run_dir(runs_dir: Path, spec: RunSpec, brief: str) -> Path:
    """Create ``<runs_dir>/<task_id>/`` with ``brief.md`` and ``spec.json``.

    The directory is created exclusively, so a (vanishingly unlikely) task-ID
    collision raises :class:`FileExistsError` rather than reusing a run.
    """
    ensure_runs_dir(runs_dir)
    path = runs_dir / spec.task_id
    path.mkdir(mode=DIR_MODE)
    write_text(path / BRIEF_FILE, brief)
    write_model(path / SPEC_FILE, spec)
    return path


def read_brief(run_dir: Path) -> str:
    return (run_dir / BRIEF_FILE).read_text(encoding="utf-8")


def read_spec(run_dir: Path) -> RunSpec:
    return RunSpec.model_validate_json((run_dir / SPEC_FILE).read_bytes())


def read_state(run_dir: Path) -> RunState:
    """The current ``status.json``, or the initial state if none exists yet.

    A run directory without a status file is one whose creator died between
    writing ``spec.json`` and spawning the worker; it reads as ``working``
    with no process, which the liveness check then settles as
    ``worker_died``. Nothing ever has to special-case the file's absence.
    """
    path = run_dir / STATUS_FILE
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return initial_state(run_dir)
    return RunState.model_validate_json(raw)


def initial_state(run_dir: Path) -> RunState:
    spec = read_spec(run_dir)
    return RunState(
        task_id=spec.task_id,
        status="working",
        run_dir=run_dir,
        created_at=spec.created_at,
    )


def update_state(run_dir: Path, transition: Callable[[RunState], RunState]) -> RunState:
    """Read-modify-write ``status.json`` under the run directory's lock.

    The server (recording the PID after spawn, reconciling a dead group,
    recording a cancel) and the worker (every transition it makes) all write
    the same file; the lock makes each one see the other's latest write, so
    ``started_at`` set by the worker is never clobbered by the server's PID
    update racing it. Readers do not lock: the write itself is atomic.
    """
    lock_fd = os.open(run_dir / LOCK_FILE, os.O_RDWR | os.O_CREAT, FILE_MODE)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        state = transition(read_state(run_dir))
        write_model(run_dir / STATUS_FILE, state)
        return state
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def read_result(run_dir: Path) -> RunResult | None:
    path = run_dir / RESULT_FILE
    if not path.exists():
        return None
    return RunResult.model_validate_json(path.read_bytes())


def write_result(run_dir: Path, result: RunResult) -> RunResult:
    write_model(run_dir / RESULT_FILE, result)
    return result


def read_process_record(run_dir: Path) -> ProcessRecord | None:
    path = run_dir / PROCESS_FILE
    if not path.exists():
        return None
    return ProcessRecord.model_validate_json(path.read_bytes())


def write_process_record(run_dir: Path, record: ProcessRecord) -> None:
    write_model(run_dir / PROCESS_FILE, record)


def read_groups(run_dir: Path) -> GroupRecords:
    path = run_dir / GROUPS_FILE
    if not path.exists():
        return GroupRecords()
    return GroupRecords.model_validate_json(path.read_bytes())


def write_groups(run_dir: Path, groups: GroupRecords) -> None:
    write_model(run_dir / GROUPS_FILE, groups)


@contextmanager
def runs_lock(runs_dir: Path) -> Iterator[None]:
    """Exclusive ``<runs_dir>/.runs.lock`` for the duration of the block.

    Serialises capacity check, directory creation, spawn and PID recording
    across every server sharing the directory, and keeps a status read from
    overlapping a half-started task. Not re-entrant: a holder must call the
    ``_locked`` variants, never take it again.
    """
    fd = os.open(runs_dir / RUNS_LOCK_FILE, os.O_RDWR | os.O_CREAT, FILE_MODE)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def acquire_worker_lock(run_dir: Path) -> int | None:
    """Take ``.worker.lock`` exclusively for the life of this process.

    Returns the fd (keep it; closing it releases the lock) or ``None`` when
    another process already holds it, which means another worker is running
    this task and this one must not touch it.
    """
    fd = os.open(run_dir / WORKER_LOCK_FILE, os.O_RDWR | os.O_CREAT, FILE_MODE)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def worker_lock_held(run_dir: Path) -> bool:
    """Whether some process currently holds ``.worker.lock``.

    The kernel releases a ``flock`` when its holder exits, zombie or not, so
    this is a liveness test that cannot be fooled by an unreaped worker.
    """
    path = run_dir / WORKER_LOCK_FILE
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def write_model(path: Path, model: BaseModel) -> None:
    write_text(path, model.model_dump_json(indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text``, mode ``0600``.

    The content goes to an exclusively created sibling temp file, is fsynced,
    and is renamed over ``path``; the directory is fsynced afterwards so the
    rename itself is durable. A concurrent reader sees the old file or the
    new one, never a partial write.
    """
    temp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def open_private(path: Path, *, append: bool = True) -> int:
    """An fd on ``path`` opened for writing with mode ``0600``.

    Used for the files that grow in place (the worker's streams, the
    transcript) rather than being replaced.
    """
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    return os.open(path, flags, FILE_MODE)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
