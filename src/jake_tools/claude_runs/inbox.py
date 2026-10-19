"""Steering messages and clarification requests: durable, keyed, locked.

Everything a parent sends a running task, and everything the worker asks
its parent, is a small file in the run directory, changed only under
``.inbox.lock``:

- ``messages/<message_id>.json`` (:class:`~.models.SteerMessage`): written
  ``queued`` by the server, moved to ``delivered`` and ``in_conversation``
  by the worker, or to ``not_delivered`` when the task ends first.
- ``inputs/<request_id>.json`` (:class:`~.models.InputRequest`): written
  ``pending`` by the worker's ``ask_parent`` tool, answered or declined by
  the server, ``expired`` when the task ends first.
- ``status.json``'s ``input_required``/``working`` is derived from the
  pending requests inside the same lock that changes them
  (:meth:`Inbox._sync_status_locked`), so a status snapshot never shows
  ``input_required`` with nothing pending, or ``working`` with a question
  open. (Lock order: inbox, then status, then events.)
- ``inbox.closed``: once present, no message is accepted. The worker
  creates it only when its conversation is idle with nothing queued, and
  every way a task ends creates it while sweeping what is left; both happen
  under the lock that accepting a message takes, so an accepted message is
  always either delivered or reported ``not_delivered``, never lost.

Retrying a submission with the same key and the same payload returns the
stored record; the same key with a different payload is a conflict.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid as uuid_module
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .events import EventLog
from .models import CALLER_ID_PATTERN, InputRequest, RunState, SteerMessage
from .rundir import (
    DIR_MODE,
    INBOX_CLOSED_FILE,
    INBOX_LOCK_FILE,
    INPUTS_DIR,
    MESSAGES_DIR,
    file_lock,
    update_state,
    utc_now,
    write_model,
    write_text,
)

MAX_MESSAGE_CHARS = 50_000
MAX_QUESTION_CHARS = 4_000
MAX_CHOICES = 10
MAX_ANSWER_CHARS = 20_000


class InboxClosedError(RuntimeError):
    """The task is no longer accepting messages."""


class ConflictError(RuntimeError):
    """A caller key was reused with a different payload."""

    def __init__(self, message: str, *, key: str) -> None:
        super().__init__(message)
        self.key = key


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class InputResponse(BaseModel):
    """A parent's response to one :class:`~.models.InputRequest`.

    ``accept`` carries a non-blank ``answer``; ``decline`` tells the worker
    no answer is coming, and carries none.
    """

    action: Literal["accept", "decline"] = "accept"
    answer: str | None = Field(default=None, max_length=MAX_ANSWER_CHARS)

    @model_validator(mode="after")
    def _answer_matches_action(self) -> InputResponse:
        if self.action == "accept" and not (self.answer and self.answer.strip()):
            raise ValueError("an accepted response needs a non-blank answer")
        if self.action == "decline" and self.answer is not None:
            raise ValueError("a declined response carries no answer")
        return self


AnswerOutcome = Literal["accepted", "duplicate", "conflict", "unknown", "not_pending"]


class AnswerResult(BaseModel):
    request_id: str
    outcome: AnswerOutcome
    state: str | None = None
    reason: str | None = None


class SteerSubmission(BaseModel):
    """A validated ``claude_steer`` call."""

    message_id: str
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)

    @field_validator("message_id")
    @classmethod
    def _message_id_shape(cls, value: str) -> str:
        if not CALLER_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "message_id must be 1-128 characters of letters, digits, '.', '_', ':' or '-'"
            )
        return value

    @field_validator("message")
    @classmethod
    def _message_has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class Inbox:
    """The ``messages/`` and ``inputs/`` of one run directory."""

    def __init__(self, run_dir: Path, events: EventLog | None = None) -> None:
        self.run_dir = run_dir
        self.events = events or EventLog(run_dir)

    @contextmanager
    def locked(self) -> Iterator[None]:
        with file_lock(self.run_dir / INBOX_LOCK_FILE):
            yield

    # --- state --------------------------------------------------------------

    def is_closed(self) -> bool:
        return (self.run_dir / INBOX_CLOSED_FILE).exists()

    def messages(self) -> list[SteerMessage]:
        """Every steering message, in submission order."""
        found = [
            SteerMessage.model_validate_json(path.read_bytes())
            for path in self._dir(MESSAGES_DIR).glob("*.json")
        ]
        return sorted(found, key=lambda m: (m.submitted_at, m.message_id))

    def inputs(self) -> list[InputRequest]:
        """Every clarification request, in the order they were asked."""
        found = [
            InputRequest.model_validate_json(path.read_bytes())
            for path in self._dir(INPUTS_DIR).glob("*.json")
        ]
        return sorted(found, key=lambda r: (r.asked_at, r.request_id))

    def pending_inputs(self) -> list[InputRequest]:
        return [r for r in self.inputs() if r.state == "pending"]

    def read_input(self, request_id: str) -> InputRequest | None:
        path = self._dir(INPUTS_DIR) / f"{request_id}.json"
        if not path.exists():
            return None
        return InputRequest.model_validate_json(path.read_bytes())

    # --- steering messages (server side) -------------------------------------

    def submit(self, submission: SteerSubmission) -> tuple[SteerMessage, bool]:
        """Store a message ``queued``; ``(record, replayed)``.

        Raises :class:`ConflictError` when the ID holds a different message
        and :class:`InboxClosedError` when the task accepts no more.
        A retry of a stored message is answered even after the inbox closed,
        so a caller can always learn what happened to a message it sent.
        """
        digest = fingerprint(submission.message)
        with self.locked():
            path = self._dir(MESSAGES_DIR) / f"{submission.message_id}.json"
            if path.exists():
                existing = SteerMessage.model_validate_json(path.read_bytes())
                if existing.fingerprint != digest:
                    raise ConflictError(
                        "message_id was already used for a different message",
                        key=submission.message_id,
                    )
                return existing, True
            if self.is_closed():
                raise InboxClosedError("the task is no longer accepting messages")
            message = SteerMessage(
                message_id=submission.message_id,
                uuid=str(uuid_module.uuid4()),
                text=submission.message,
                fingerprint=digest,
                submitted_at=utc_now(),
            )
            self._ensure_dir(MESSAGES_DIR)
            write_model(path, message)
            self.events.append(
                "message_queued",
                message_id=message.message_id,
                text=message.text,
                text_chars=len(message.text),
            )
        return message, False

    # --- steering messages (worker side) -------------------------------------

    def queued(self) -> list[SteerMessage]:
        return [m for m in self.messages() if m.state == "queued"]

    def mark_attempted(self, message: SteerMessage) -> SteerMessage | None:
        """Record that a write is about to begin; ``None`` if it must not.

        Re-read under the lock: a close that settled the message as
        ``not_delivered`` in the meantime wins, and nothing is written.
        """
        with self.locked():
            current = self._message(message.message_id)
            if current is None or current.state != "queued" or self.is_closed():
                return None
            return self._save(current.model_copy(update={"attempted_at": utc_now()}))

    def mark_delivered(self, message: SteerMessage) -> SteerMessage:
        with self.locked():
            current = self._message(message.message_id)
            if current is None or current.state != "queued":
                return current or message
            updated = self._save(
                current.model_copy(
                    update={"state": "delivered", "delivered_at": utc_now()}
                )
            )
            self.events.append("message_delivered", message_id=message.message_id)
        return updated

    def mark_in_conversation(self, message_uuid: str) -> SteerMessage | None:
        """Record that the CLI echoed the message with ``message_uuid``."""
        with self.locked():
            for current in self.messages():
                if current.uuid != message_uuid:
                    continue
                if current.state == "in_conversation":
                    return current
                updated = self._save(
                    current.model_copy(
                        update={
                            "state": "in_conversation",
                            "in_conversation_at": utc_now(),
                        }
                    )
                )
                self.events.append(
                    "message_in_conversation", message_id=current.message_id
                )
                return updated
        return None

    def close_if_idle(self) -> bool:
        """Close the inbox when nothing is queued or still on its way.

        "On its way" is a message written to the CLI that it has not yet
        echoed into the conversation: the CLI will act on it in a new turn,
        so the conversation is not over. True when the inbox is now closed.
        """
        with self.locked():
            if self.is_closed():
                return True
            if any(m.state in ("queued", "delivered") for m in self.messages()):
                return False
            write_text(self.run_dir / INBOX_CLOSED_FILE, "")
            return True

    # --- clarification (worker side) -----------------------------------------

    def ask(self, question: str, choices: list[str]) -> InputRequest:
        with self.locked():
            count = len(list(self._dir(INPUTS_DIR).glob("*.json")))
            request = InputRequest(
                request_id=f"q{count + 1}-{secrets.token_hex(4)}",
                question=question,
                choices=choices,
                asked_at=utc_now(),
            )
            self._ensure_dir(INPUTS_DIR)
            write_model(self._dir(INPUTS_DIR) / f"{request.request_id}.json", request)
            self.events.append(
                "input_requested",
                request_id=request.request_id,
                text=question,
                text_chars=len(question),
                choices=choices or None,
            )
            self._sync_status_locked()
        return request

    def withdraw(self, request_id: str, reason: str) -> None:
        """Expire a still-pending request whose tool call was abandoned."""
        with self.locked():
            current = self.read_input(request_id)
            if current is not None and current.state == "pending":
                self._save_input(
                    current.model_copy(update={"state": "expired", "reason": reason})
                )
                self.events.append(
                    "input_expired", request_id=request_id, detail={"reason": reason}
                )
            self._sync_status_locked()

    def sync_status(self) -> None:
        with self.locked():
            self._sync_status_locked()

    def _sync_status_locked(self) -> None:
        """``input_required`` while any request is pending, else ``working``.

        Never touches a terminal state, and only leaves ``input_required``
        (not some other non-terminal state) when nothing is pending.
        """
        pending = self.pending_inputs()

        def transition(state: RunState) -> RunState:
            if state.is_terminal:
                return state
            if pending:
                first = pending[0].question.strip().splitlines()[0]
                return state.model_copy(
                    update={
                        "status": "input_required",
                        "status_message": f"Waiting for an answer: {first}",
                    }
                )
            if state.status != "input_required":
                return state
            return state.model_copy(
                update={
                    "status": "working",
                    "status_message": "Answer received; continuing",
                }
            )

        update_state(self.run_dir, transition)

    # --- clarification (server side) -----------------------------------------

    def respond(self, responses: dict[str, InputResponse]) -> list[AnswerResult]:
        """Apply keyed responses; one :class:`AnswerResult` per key.

        A pending request takes the response. A key that was already
        settled reports ``duplicate`` (same response) or ``conflict``
        (different); one that expired reports ``not_pending``; one never
        issued reports ``unknown``. Nothing is applied partially per key.
        """
        results: list[AnswerResult] = []
        with self.locked():
            for request_id, response in responses.items():
                results.append(self._respond_locked(request_id, response))
            self._sync_status_locked()
        return results

    def _respond_locked(self, request_id: str, response: InputResponse) -> AnswerResult:
        if not CALLER_ID_PATTERN.fullmatch(request_id):
            return AnswerResult(request_id=request_id, outcome="unknown")
        current = self.read_input(request_id)
        if current is None:
            return AnswerResult(request_id=request_id, outcome="unknown")
        state = "answered" if response.action == "accept" else "declined"
        if current.state == "expired":
            return AnswerResult(
                request_id=request_id,
                outcome="not_pending",
                state=current.state,
                reason=current.reason,
            )
        if current.state != "pending":
            same = current.state == state and current.answer == response.answer
            return AnswerResult(
                request_id=request_id,
                outcome="duplicate" if same else "conflict",
                state=current.state,
                reason=None if same else "this request was already settled differently",
            )
        self._save_input(
            current.model_copy(
                update={
                    "state": state,
                    "answer": response.answer,
                    "answered_at": utc_now(),
                }
            )
        )
        if state == "answered":
            self.events.append(
                "input_answered",
                request_id=request_id,
                text=response.answer,
                text_chars=len(response.answer or ""),
            )
        else:
            self.events.append("input_declined", request_id=request_id)
        return AnswerResult(request_id=request_id, outcome="accepted", state=state)

    # --- ending ----------------------------------------------------------------

    def close(self, reason: str) -> None:
        """Close for good: refuse new messages, settle everything left open.

        Queued messages become ``not_delivered`` (or, if a write had begun,
        ``not_delivered`` saying delivery is unknown) and pending requests
        ``expired``, each with ``reason``. A ``delivered`` message the CLI
        never echoed keeps its state (it was written) and gains a reason
        saying it was never confirmed in the conversation. Idempotent.
        """
        with self.locked():
            write_text(self.run_dir / INBOX_CLOSED_FILE, "")
            for message in self.messages():
                if message.state == "delivered" and message.reason is None:
                    self._save(
                        message.model_copy(
                            update={
                                "reason": (
                                    f"{reason}; it was written to Claude's input, but "
                                    "Claude Code never confirmed adding it to the "
                                    "conversation"
                                )
                            }
                        )
                    )
                    continue
                if message.state != "queued":
                    continue
                why = reason
                if message.attempted_at is not None:
                    why = (
                        f"{reason}; the worker stopped while writing this message, "
                        "so it may or may not have reached Claude"
                    )
                self._save(
                    message.model_copy(update={"state": "not_delivered", "reason": why})
                )
                self.events.append(
                    "message_not_delivered",
                    message_id=message.message_id,
                    detail={"reason": why},
                )
            for request in self.pending_inputs():
                self._save_input(
                    request.model_copy(update={"state": "expired", "reason": reason})
                )
                self.events.append(
                    "input_expired",
                    request_id=request.request_id,
                    detail={"reason": reason},
                )

    # --- files -----------------------------------------------------------------

    def _dir(self, name: str) -> Path:
        return self.run_dir / name

    def _ensure_dir(self, name: str) -> None:
        self._dir(name).mkdir(mode=DIR_MODE, exist_ok=True)

    def _message(self, message_id: str) -> SteerMessage | None:
        path = self._dir(MESSAGES_DIR) / f"{message_id}.json"
        if not path.exists():
            return None
        return SteerMessage.model_validate_json(path.read_bytes())

    def _save(self, message: SteerMessage) -> SteerMessage:
        write_model(self._dir(MESSAGES_DIR) / f"{message.message_id}.json", message)
        return message

    def _save_input(self, request: InputRequest) -> InputRequest:
        write_model(self._dir(INPUTS_DIR) / f"{request.request_id}.json", request)
        return request
