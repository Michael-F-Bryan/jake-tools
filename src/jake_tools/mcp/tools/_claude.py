"""What the ``claude_*`` tools share: the task service and error mapping."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import ValidationError

from ...claude_runs.control import (
    ArgumentError,
    CapacityExceededError,
    RequestConflictError,
    ResumeUnavailableError,
    RunDefaults,
    SessionBusyError,
    WorkerSpawnError,
)
from ...claude_runs.events import InvalidCursorError
from ...claude_runs.inbox import ConflictError
from ...claude_runs.rundir import UnknownTaskError
from ...claude_runs.service import TaskNotRunningError, TaskService
from ...config import Config
from ..errors import ToolError


def task_service(config: Config) -> TaskService:
    return TaskService(config.runs_dir.value, RunDefaults.from_config(config))


@contextmanager
def task_errors() -> Iterator[None]:
    """Translate the task service's expected failures into tool errors."""
    try:
        yield
    except ValidationError as exc:
        raise invalid_argument(exc) from exc
    except UnknownTaskError as exc:
        raise unknown_task_error(exc) from exc
    except InvalidCursorError as exc:
        raise ToolError(
            "invalid_argument", f"cursor: {exc}", detail={"argument": "cursor"}
        ) from exc
    except ArgumentError as exc:
        raise ToolError(
            "invalid_argument", str(exc), detail={"argument": exc.argument}
        ) from exc
    except CapacityExceededError as exc:
        raise ToolError(
            "capacity_exceeded", str(exc), detail={"live": exc.live, "limit": exc.limit}
        ) from exc
    except WorkerSpawnError as exc:
        raise ToolError("worker_failed", str(exc)) from exc
    except RequestConflictError as exc:
        raise ToolError(
            "request_conflict",
            str(exc),
            detail={"request_id": exc.request_id, "task_id": exc.task_id},
        ) from exc
    except ConflictError as exc:
        raise ToolError("request_conflict", str(exc), detail={"key": exc.key}) from exc
    except SessionBusyError as exc:
        raise ToolError(
            "session_busy",
            str(exc),
            detail={"session_id": exc.session_id, "task_id": exc.task_id},
        ) from exc
    except ResumeUnavailableError as exc:
        detail = {"task_id": exc.task_id}
        if exc.latest_task_id is not None:
            detail["latest_task_id"] = exc.latest_task_id
        raise ToolError("resume_unavailable", str(exc), detail=detail) from exc
    except TaskNotRunningError as exc:
        raise ToolError(
            "task_not_running",
            exc.reason,
            detail={"task_id": exc.task_id, "status": exc.status},
        ) from exc


def invalid_argument(exc: ValidationError) -> ToolError:
    first = exc.errors()[0]
    location = first.get("loc") or ()
    argument = str(location[0]) if location else None
    message = first.get("msg", "invalid argument")
    message = message.removeprefix("Value error, ")
    prefix = f"{argument}: " if argument else ""
    detail = {"argument": argument} if argument else None
    return ToolError("invalid_argument", f"{prefix}{message}", detail=detail)


def unknown_task_error(exc: UnknownTaskError) -> ToolError:
    return ToolError(
        "unknown_task", "no such delegated task", detail={"task_id": exc.task_id}
    )
