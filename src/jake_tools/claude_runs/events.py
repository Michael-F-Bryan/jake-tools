"""``events.jsonl``: the normalised, public record of a task, read by cursor.

The worker turns each SDK message into zero or more :class:`Event` s through
:func:`events_from_view`; steering, clarification and cancellation append
their own. Events never carry thinking: :func:`jake_tools.claude.describe`
has already dropped it. Assistant narration is stored in full; a tool
result is stored up to :data:`STORED_TOOL_RESULT_CHARS` with its full length
recorded, because ``transcript.jsonl`` keeps the whole thing.

A :class:`Cursor` is ``<task_id>:<seq>:<offset>``: the sequence number of
the next event and the byte offset of its line. Reading a page seeks to the
offset and reads forward, so a poll never rescans the file, and the line at
the offset must carry the cursor's sequence number, so a cursor from
another task, a hand-edited one, or one past the end is rejected rather
than silently misread.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ..claude import MessageView
from .rundir import EVENTS_FILE, EVENTS_LOCK_FILE, file_lock, open_private, utc_now

EventKind = Literal[
    "started",
    "assistant_text",
    "tool_use",
    "tool_result",
    "turn_result",
    "notice",
    "message_queued",
    "message_delivered",
    "message_in_conversation",
    "message_not_delivered",
    "input_requested",
    "input_answered",
    "input_declined",
    "input_expired",
    "cancel_requested",
    "finished",
]

ACTIVITY_KINDS: frozenset[str] = frozenset(
    {"assistant_text", "tool_use", "tool_result", "turn_result", "input_requested"}
)
"""Kinds that are the agent doing something, as opposed to bookkeeping."""

STORED_TOOL_RESULT_CHARS = 8000
STORED_TOOL_INPUT_CHARS = 2000
STATUS_MESSAGE_CHARS = 240


class Event(BaseModel):
    """One normalised event. ``kind`` says which optional fields are set.

    ``text`` is narration, a tool's input summary or result, a message, a
    question or an answer. ``text_chars`` is the full length of the text
    the event stands for; ``text_truncated`` is true when ``text`` holds
    less (the full text is in ``transcript.jsonl``, or ``final.md`` for a
    final answer). ``detail`` carries small kind-specific values such as a
    result subtype or a termination reason.
    """

    seq: int
    at: datetime
    kind: EventKind
    text: str | None = None
    text_chars: int | None = None
    text_truncated: bool = False
    tool: str | None = None
    tool_use_id: str | None = None
    is_error: bool | None = None
    message_id: str | None = None
    request_id: str | None = None
    choices: list[str] | None = None
    detail: dict[str, str | int | float | bool | None] | None = None

    @property
    def is_activity(self) -> bool:
        return self.kind in ACTIVITY_KINDS

    def clipped(self, max_chars: int) -> Event:
        """This event with ``text`` cut to ``max_chars``, marked if it was."""
        if self.text is None or len(self.text) <= max_chars:
            return self
        return self.model_copy(
            update={
                "text": self.text[:max_chars],
                "text_chars": self.text_chars or len(self.text),
                "text_truncated": True,
            }
        )


class InvalidCursorError(ValueError):
    """A cursor that is malformed, for another task, or not at an event."""


@dataclass(frozen=True)
class Cursor:
    task_id: str
    seq: int
    offset: int

    @classmethod
    def start(cls, task_id: str) -> Cursor:
        return cls(task_id=task_id, seq=1, offset=0)

    @classmethod
    def parse(cls, text: str, task_id: str) -> Cursor:
        parts = text.rsplit(":", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            raise InvalidCursorError("cursor is malformed")
        if parts[0] != task_id:
            raise InvalidCursorError("cursor belongs to a different task")
        seq, offset = int(parts[1]), int(parts[2])
        if seq < 1:
            raise InvalidCursorError("cursor is malformed")
        return cls(task_id=task_id, seq=seq, offset=offset)

    def encode(self) -> str:
        return f"{self.task_id}:{self.seq}:{self.offset}"


class Page(BaseModel):
    """Up to ``limit`` events from a cursor, and where to continue."""

    events: list[Event]
    next_cursor: str
    has_more: bool
    truncated: bool
    """True when any returned event's text was clipped for this page."""


class EventLog:
    """Append and page ``events.jsonl`` for one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / EVENTS_FILE
        self.task_id = run_dir.name

    def exists(self) -> bool:
        return self.path.exists()

    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    def append(self, kind: EventKind, **fields: Any) -> Event:
        """Append one event under the log's lock and return it.

        The line goes out in a single ``write`` on an ``O_APPEND`` fd, and a
        reader treats a line without its newline as not yet written.
        """
        with file_lock(self.run_dir / EVENTS_LOCK_FILE):
            last = _last_event(self.path)
            event = Event(
                seq=(last.seq if last else 0) + 1, at=utc_now(), kind=kind, **fields
            )
            fd = open_private(self.path, append=True)
            try:
                os.write(fd, (event.model_dump_json() + "\n").encode())
            finally:
                os.close(fd)
        return event

    def latest(self) -> Event | None:
        return _last_event(self.path)

    def tail(self, max_bytes: int) -> list[Event]:
        """The complete events in (about) the last ``max_bytes`` of the log."""
        size = self.size()
        start = max(0, size - max_bytes)
        if start:
            start = _last_newline_before(self.path, start) + 1
        return [event for _, event in _read_from(self.path, start)]

    def page(
        self,
        cursor: Cursor,
        *,
        limit: int,
        max_text_chars: int,
        max_bytes: int,
    ) -> Page:
        """Events from ``cursor``: at most ``limit``, about ``max_bytes``.

        At least one event is returned when one exists, however large;
        ``has_more`` says whether more were already written.
        """
        if cursor.task_id != self.task_id:
            raise InvalidCursorError("cursor belongs to a different task")
        size = self.size()
        self._validate(cursor, size)
        events: list[Event] = []
        truncated = False
        offset, seq, used = cursor.offset, cursor.seq, 0
        for line_end, event in _read_from(self.path, cursor.offset):
            if len(events) >= limit:
                break
            shown = event.clipped(max_text_chars)
            cost = len(shown.model_dump_json())
            if events and used + cost > max_bytes:
                break
            truncated = truncated or (shown is not event)
            events.append(shown)
            used += cost
            offset, seq = line_end, event.seq + 1
        next_cursor = Cursor(task_id=self.task_id, seq=seq, offset=offset)
        return Page(
            events=events,
            next_cursor=next_cursor.encode(),
            has_more=self._complete_size() > offset,
            truncated=truncated,
        )

    def _complete_size(self) -> int:
        """The size up to the last complete line."""
        size = self.size()
        if size == 0:
            return 0
        with self.path.open("rb") as handle:
            handle.seek(size - 1)
            if handle.read(1) == b"\n":
                return size
        last = _last_newline_before(self.path, size)
        return last + 1 if last >= 0 else 0

    def _validate(self, cursor: Cursor, size: int) -> None:
        if cursor.offset > size:
            raise InvalidCursorError("cursor is beyond the end of the task's events")
        if cursor.offset == 0:
            if cursor.seq != 1:
                raise InvalidCursorError("cursor does not point at an event")
            return
        with self.path.open("rb") as handle:
            handle.seek(cursor.offset - 1)
            if handle.read(1) != b"\n":
                raise InvalidCursorError("cursor does not point at an event")
            line = handle.readline()
        if line.endswith(b"\n"):
            event = _parse(line)
            if event is None or event.seq != cursor.seq:
                raise InvalidCursorError("cursor does not point at an event")
            return
        previous = _event_before(self.path, cursor.offset)
        if previous is None or previous.seq != cursor.seq - 1:
            raise InvalidCursorError("cursor does not point at an event")


def _parse(line: bytes) -> Event | None:
    try:
        return Event.model_validate_json(line)
    except ValidationError:
        return None


def _read_from(path: Path, offset: int) -> Iterator[tuple[int, Event]]:
    """``(end offset, event)`` for each complete line from ``offset``."""
    if not path.exists():
        return
    with path.open("rb") as handle:
        handle.seek(offset)
        position = offset
        while True:
            line = handle.readline()
            if not line.endswith(b"\n"):
                return  # end of file, or a line still being written
            position += len(line)
            event = _parse(line)
            if event is not None:
                yield position, event


def _last_newline_before(path: Path, end: int) -> int:
    """The offset of the last ``\\n`` strictly before ``end``, or -1."""
    chunk = 65536
    with path.open("rb") as handle:
        position = end
        while position > 0:
            start = max(0, position - chunk)
            handle.seek(start)
            data = handle.read(position - start)
            index = data.rfind(b"\n")
            if index >= 0:
                return start + index
            position = start
    return -1


def _event_before(path: Path, offset: int) -> Event | None:
    """The complete event whose line ends exactly at ``offset``."""
    start = _last_newline_before(path, offset - 1) + 1
    with path.open("rb") as handle:
        handle.seek(start)
        return _parse(handle.read(offset - start))


def _last_event(path: Path) -> Event | None:
    if not path.exists():
        return None
    size = path.stat().st_size
    if size == 0:
        return None
    end = _last_newline_before(path, size + 1) if size else -1
    if end < 0:
        return None
    return _event_before(path, end + 1)


# --- normalising SDK messages ----------------------------------------------------


def events_from_view(view: MessageView) -> list[tuple[EventKind, dict[str, Any]]]:
    """The events one SDK message stands for, as ``(kind, fields)`` pairs.

    ``init`` and ``user_input`` views are handled by the worker (session
    check, delivery confirmation), so they produce nothing here.
    """
    found: list[tuple[EventKind, dict[str, Any]]] = []
    if view.kind == "assistant":
        for text in view.texts:
            if text.strip():
                found.append(
                    ("assistant_text", {"text": text, "text_chars": len(text)})
                )
        for use in view.tool_uses:
            summary = tool_input_summary(use.input)
            found.append(
                (
                    "tool_use",
                    {
                        "tool": use.name,
                        "tool_use_id": use.id,
                        "text": summary[:STORED_TOOL_INPUT_CHARS],
                        "text_chars": len(summary),
                        "text_truncated": len(summary) > STORED_TOOL_INPUT_CHARS,
                    },
                )
            )
    elif view.kind == "tool_results":
        for result in view.tool_results:
            found.append(
                (
                    "tool_result",
                    {
                        "tool_use_id": result.tool_use_id,
                        "is_error": result.is_error,
                        "text": result.text[:STORED_TOOL_RESULT_CHARS],
                        "text_chars": len(result.text),
                        "text_truncated": len(result.text) > STORED_TOOL_RESULT_CHARS,
                    },
                )
            )
    elif view.kind == "result" and view.result is not None:
        result = view.result
        text = result.text or ""
        found.append(
            (
                "turn_result",
                {
                    "text": text or None,
                    "text_chars": len(text),
                    "is_error": result.is_error,
                    "detail": {
                        "subtype": result.subtype,
                        "stop_reason": result.stop_reason,
                        "num_turns": result.num_turns,
                        "cost_usd": result.usage.estimated_cost_usd,
                        "errors": "; ".join(result.errors) or None,
                    },
                },
            )
        )
    elif view.kind == "notice" and view.notice:
        found.append(("notice", {"text": view.notice, "text_chars": len(view.notice)}))
    return found


_SUMMARY_KEYS = ("command", "file_path", "path", "pattern", "url", "query", "question")


def tool_input_summary(tool_input: dict[str, Any]) -> str:
    """A one-line description of a tool call's input.

    The well-known argument (a command, a path, a pattern) when there is
    one, else the input as compact JSON.
    """
    for key in _SUMMARY_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return json.dumps(tool_input, separators=(",", ":"), default=str)


def progress_line(event: Event) -> str | None:
    """The ``status_message`` an event implies, or ``None`` to keep the last."""
    match event.kind:
        case "assistant_text":
            line = _first_line(event.text or "")
        case "tool_use":
            line = f"{event.tool}: {_first_line(event.text or '')}"
        case "notice":
            line = event.text or ""
        case "input_requested":
            line = f"Waiting for an answer: {_first_line(event.text or '')}"
        case _:
            return None
    if len(line) > STATUS_MESSAGE_CHARS:
        line = line[: STATUS_MESSAGE_CHARS - 1] + "…"
    return line


def _first_line(text: str) -> str:
    stripped = text.strip()
    return stripped.splitlines()[0] if stripped else ""
