"""Compact supervision: what ``claude_poll`` returns, and when it returns.

A poll is an *observation* of the task as it is now, never a page of its
history (``claude_transcript`` pages history, by its own cursor). It
carries:

- the current state, from ``status.json`` and the inbox: status, one bounded
  progress line, the last real activity, what tool (if any) is running,
  every pending question, and, once the task has ended, its result;
- with an observation token, the *meaningful* changes since that
  observation: grouped denials, material warnings, steering-delivery and
  question changes. Routine activity is only counted.

**Observation tokens** are opaque, task-scoped positions in ``events.jsonl``
(``obs1.`` + base64url of ``<task_id>:<seq>:<offset>``). The server keeps no
reader state: reading never consumes anything, any number of readers can
hold their own tokens, a retry with the same token sees the same changes
(plus newer ones), and tokens stay valid across server restarts. A token
that does not decode, belongs to another task, or no longer points at an
event is not an error: the poll *resynchronises*, returning the current
snapshot and a fresh token, and says so (``token_status``). Pending
questions come from the inbox, not from the token range, so advancing a
token can never dismiss an unanswered question.

Consistency: the returned token is the log position the changes were read
up to, fixed *before* the state is read, so the snapshot is never older
than the token; nothing between the old token and the new one is skipped.

**Wake rules** (:func:`wake_reason`): a pending question and a terminal
state always return at once, whatever ``until`` says (a question can block
the very outcome a terminal wait is waiting for). ``attention`` (the
default) also wakes on a new distinct denial, a material warning, or
steering that could not be delivered. ``progress`` additionally wakes on
any routine activity (the explicit checkpoint mode). Nothing else wakes a
poll; at its deadline it returns the current snapshot.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .events import (
    ACTIVITY_KINDS,
    Cursor,
    Event,
    EventLog,
    InvalidCursorError,
    progress_line,
)
from .models import (
    DeliveryState,
    InputRequest,
    PermissionDenial,
    RunState,
    RunStatus,
    TerminationReason,
)

TOKEN_PREFIX = "obs1."
QUESTIONS_SHOWN = 5
QUESTION_CHARS = 2_000
CHANGES_SHOWN = 10
CHANGE_TEXT_CHARS = 300
PROGRESS_CHARS = 240
INLINE_RESULT_CHARS = 4_000
RESULT_EXCERPT_CHARS = 2_000
ERROR_CHARS = 600
NARRATION_CHARS = 600
ACTIVITY_TAIL_BYTES = 64_000

WakeReason = Literal[
    "question",
    "terminal",
    "denial",
    "warning",
    "steering",
    "progress",
    "resynchronised",
    "deadline",
]
PollUntil = Literal["attention", "terminal", "progress"]
TokenStatus = Literal["none", "current", "resynchronised"]


class InvalidTokenError(ValueError):
    """An observation token that cannot be used; the poll resynchronises."""


def encode_token(position: Cursor) -> str:
    raw = position.encode().encode()
    return TOKEN_PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_token(token: str, task_id: str) -> Cursor:
    if not token.startswith(TOKEN_PREFIX):
        raise InvalidTokenError("not an observation token")
    body = token[len(TOKEN_PREFIX) :]
    try:
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode()
        return Cursor.parse(raw, task_id)
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        message = str(exc) if isinstance(exc, InvalidCursorError) else "malformed"
        raise InvalidTokenError(message.replace("cursor", "token")) from exc


def clip(text: str | None, limit: int) -> tuple[str | None, int | None, bool]:
    """``(shown, full length, truncated)`` for an optional text."""
    if text is None:
        return None, None, False
    if len(text) <= limit:
        return text, len(text), False
    return text[:limit], len(text), True


class Retrieval(BaseModel):
    """The MCP call that returns what was left out here."""

    tool: str
    arguments: dict[str, str | int]


class Question(BaseModel):
    """A pending question. ``question`` is cut to :data:`QUESTION_CHARS`."""

    request_id: str
    question: str
    question_chars: int
    question_truncated: bool = False
    choices: list[str] = Field(default_factory=list)
    asked_at: datetime

    @classmethod
    def of(cls, request: InputRequest) -> Question:
        shown, chars, cut = clip(request.question, QUESTION_CHARS)
        return cls(
            request_id=request.request_id,
            question=shown or "",
            question_chars=chars or 0,
            question_truncated=cut,
            choices=request.choices,
            asked_at=request.asked_at,
        )


DenialSource = Literal["runtime", "inferred_from_tool_error"]


class DenialGroup(BaseModel):
    """Refused calls with the same tool and request, since the token.

    ``source`` is ``runtime`` when the CLI reported the denial itself, and
    ``inferred_from_tool_error`` when only the call's error text said so.
    ``new`` is false when the same refusal was already reported before the
    token (a repeat: it does not wake a poll). ``first_seq``/``last_seq``
    are the ``permission_denied`` events; ``claude_transcript`` shows them.
    """

    tool: str
    requested: str | None = None
    count: int
    source: DenialSource = "runtime"
    runtime_reason: str | None = None
    runtime_message: str | None = None
    new: bool = True
    first_seq: int
    last_seq: int


class SteeringChange(BaseModel):
    message_id: str
    state: DeliveryState
    reason: str | None = None


class QuestionChange(BaseModel):
    request_id: str
    state: Literal["answered", "declined", "expired"]
    reason: str | None = None


class Warning(BaseModel):
    """A notice or lifecycle event worth seeing. ``category`` is ``notice``,
    ``rate_limit``, ``api_retry`` (counted, does not wake), ``cancel_requested``
    or ``handoff``. Repeats of one text are grouped in ``count``."""

    category: str
    text: str | None = None
    count: int = 1
    last_seq: int


class Changes(BaseModel):
    """Meaningful changes between the caller's token and the new one.

    ``events`` counts every event in that range and ``activity`` the routine
    ones (narration, tool calls and results). Each list keeps at most
    :data:`CHANGES_SHOWN` entries; ``omitted`` counts the rest and
    ``retrieve`` reads the whole range with ``claude_transcript``.
    """

    since_seq: int
    through_seq: int
    events: int = 0
    activity: int = 0
    denials: list[DenialGroup] = Field(default_factory=list)
    steering: list[SteeringChange] = Field(default_factory=list)
    questions: list[QuestionChange] = Field(default_factory=list)
    warnings: list[Warning] = Field(default_factory=list)
    omitted: int = 0
    retrieve: Retrieval | None = None


class RunningTool(BaseModel):
    """The tool call the agent is waiting on: started, no result yet."""

    tool: str
    input: str | None = None
    since: datetime


class Outcome(BaseModel):
    """A finished task's result, bounded for supervision.

    ``final_text`` is the agent's final answer (its claim, not verified
    fact): in full up to :data:`INLINE_RESULT_CHARS`, otherwise its first
    :data:`RESULT_EXCERPT_CHARS` characters with ``final_text_truncated``.
    ``retrieve`` is the ``claude_status`` call that returns the full text,
    in chunks; it is set whenever anything here was cut. A failed or
    cancelled task can still have useful text. ``diagnostics`` are
    mechanically detected inconsistencies (an empty answer with no recorded
    model turn, for example); they are not judgements of the work.
    """

    outcome: RunStatus
    termination_reason: TerminationReason | None = None
    final_text: str | None = None
    final_text_chars: int | None = None
    final_text_truncated: bool = False
    error: str | None = None
    last_narration: str | None = None
    handoff_status: str | None = None
    handoff_text: str | None = None
    handoff_text_chars: int | None = None
    handoff_truncated: bool = False
    session_id: str | None = None
    resumable: bool = False
    invocation_cost_usd: float | None = None
    diagnostics: list[str] = Field(default_factory=list)
    retrieve: Retrieval | None = None


class Observation(BaseModel):
    """One compact supervision snapshot (the result of ``claude_poll``).

    Pass ``observation_token`` to the next poll to get only the changes
    after this one; it never needs draining and never expires.
    """

    task_id: str
    status: RunStatus
    termination_reason: TerminationReason | None = None
    wake_reason: WakeReason
    observation_token: str
    token_status: TokenStatus
    resync_reason: str | None = None
    waited_seconds: float
    progress: str | None = None
    last_activity_at: datetime | None = None
    running_tool: RunningTool | None = None
    turns_used: int | None = None
    max_turns: int | None = None
    cancel_requested_at: datetime | None = None
    pending_questions: list[Question] = Field(default_factory=list)
    pending_questions_total: int = 0
    permission_denials_total: int = 0
    changes: Changes | None = None
    result: Outcome | None = None


# --- scanning the token range -------------------------------------------------


class _DenialAcc:
    def __init__(self, event: Event, new: bool) -> None:
        detail = event.detail or {}
        message = detail.get("runtime_message")
        reason = detail.get("runtime_reason")
        source = detail.get("source")
        self.group = DenialGroup(
            tool=event.tool or "unknown",
            requested=clip(event.text, CHANGE_TEXT_CHARS)[0],
            count=1,
            source="inferred_from_tool_error"
            if source == "inferred_from_tool_error"
            else "runtime",
            runtime_reason=reason if isinstance(reason, str) else None,
            runtime_message=clip(message, CHANGE_TEXT_CHARS)[0]
            if isinstance(message, str)
            else None,
            new=new,
            first_seq=event.seq,
            last_seq=event.seq,
        )


class Scanner:
    """Accumulates the meaningful changes after a token, incrementally.

    ``known_denials`` are ``(tool, requested)`` pairs already reported before
    the token: a later denial of the same pair is a repeat, not new.
    """

    def __init__(
        self, log: EventLog, start: Cursor, known_denials: set[tuple[str, str | None]]
    ) -> None:
        self.log = log
        self.start = start
        self.position = start
        self.known_denials = known_denials
        self.events = 0
        self.activity = 0
        self._denials: dict[tuple[str, str | None], _DenialAcc] = {}
        self._denied_ids: dict[str, tuple[str, str | None]] = {}
        self._steering: dict[str, SteeringChange] = {}
        self._questions: dict[str, QuestionChange] = {}
        self._warnings: dict[tuple[str, str | None], Warning] = {}
        self.wakes: set[WakeReason] = set()

    def advance(self) -> Cursor:
        end = self.log.end()
        for position, event in self.log.read(self.position, end.offset):
            self._take(event)
            self.position = position
        return self.position

    def _take(self, event: Event) -> None:
        self.events += 1
        if event.kind in ACTIVITY_KINDS and event.kind != "input_requested":
            self.activity += 1
            self.wakes.add("progress")
        match event.kind:
            case "permission_denied":
                self._deny(event)
            case (
                "message_not_delivered"
                | "message_delivered"
                | "message_in_conversation"
            ):
                state: DeliveryState = {
                    "message_not_delivered": "not_delivered",
                    "message_delivered": "delivered",
                    "message_in_conversation": "in_conversation",
                }[event.kind]  # type: ignore[assignment]
                reason = (event.detail or {}).get("reason")
                self._steering.pop(event.message_id or "", None)
                self._steering[event.message_id or ""] = SteeringChange(
                    message_id=event.message_id or "",
                    state=state,
                    reason=clip(reason, CHANGE_TEXT_CHARS)[0]
                    if isinstance(reason, str)
                    else None,
                )
                if state == "not_delivered":
                    self.wakes.add("steering")
            case "input_answered" | "input_declined" | "input_expired":
                settled = event.kind.removeprefix("input_")
                reason = (event.detail or {}).get("reason")
                self._questions[event.request_id or ""] = QuestionChange(
                    request_id=event.request_id or "",
                    state=settled,  # type: ignore[arg-type]
                    reason=clip(reason, CHANGE_TEXT_CHARS)[0]
                    if isinstance(reason, str)
                    else None,
                )
                if settled == "expired":
                    self.wakes.add("question")
            case "notice":
                category = str((event.detail or {}).get("category") or "notice")
                self._warn(category, event)
                if category != "api_retry":
                    self.wakes.add("warning")
            case "cancel_requested":
                self._warn("cancel_requested", event)
            case "handoff_started":
                self._warn("handoff", event)
            case _:
                pass

    def _deny(self, event: Event) -> None:
        key = (event.tool or "unknown", event.text)
        if event.tool_use_id:
            seen = self._denied_ids.get(event.tool_use_id)
            if seen is not None:
                # One call reported twice: the runtime confirming a denial
                # first inferred from its error text.
                group = self._denials[seen].group
                if (event.detail or {}).get("source") != "inferred_from_tool_error":
                    group.source = "runtime"
                return
            if (event.detail or {}).get("confirms_inferred"):
                return  # confirms a call reported before the token
            self._denied_ids[event.tool_use_id] = key
        found = self._denials.get(key)
        if found is None:
            new = key not in self.known_denials
            self._denials[key] = _DenialAcc(event, new)
            if new:
                self.wakes.add("denial")
            return
        group = found.group
        group.count += 1
        group.last_seq = event.seq

    def _warn(self, category: str, event: Event) -> None:
        text = clip(event.text, CHANGE_TEXT_CHARS)[0]
        key = (category, None if category == "api_retry" else text)
        found = self._warnings.get(key)
        if found is None:
            self._warnings[key] = Warning(
                category=category, text=text, last_seq=event.seq
            )
        else:
            found.count += 1
            found.text = text
            found.last_seq = event.seq

    def changes(self) -> Changes:
        lists = {
            "denials": [d.group for d in self._denials.values()],
            "steering": list(self._steering.values()),
            "questions": list(self._questions.values()),
            "warnings": list(self._warnings.values()),
        }
        omitted = sum(max(0, len(v) - CHANGES_SHOWN) for v in lists.values())
        return Changes(
            since_seq=self.start.seq,
            through_seq=self.position.seq - 1,
            events=self.events,
            activity=self.activity,
            denials=lists["denials"][:CHANGES_SHOWN],
            steering=lists["steering"][-CHANGES_SHOWN:],
            questions=lists["questions"][-CHANGES_SHOWN:],
            warnings=lists["warnings"][-CHANGES_SHOWN:],
            omitted=omitted,
            retrieve=Retrieval(
                tool="claude_transcript",
                arguments={"task_id": self.log.task_id, "cursor": self.start.encode()},
            )
            if omitted
            else None,
        )


def known_denials(
    denials: list[PermissionDenial], before_seq: int
) -> set[tuple[str, str | None]]:
    return {
        (d.tool, d.requested)
        for d in denials
        if d.event_seq is not None and d.event_seq < before_seq
    }


def wake_reason(
    state: RunState, pending: int, scanner: Scanner | None, until: PollUntil
) -> WakeReason | None:
    """Why a poll should return now, or ``None`` to keep waiting."""
    if state.is_terminal:
        return "terminal"
    if pending or (scanner is not None and "question" in scanner.wakes):
        return "question"
    if until == "terminal" or scanner is None:
        return None
    for reason in ("denial", "steering", "warning"):
        if reason in scanner.wakes:
            return reason  # type: ignore[return-value]
    if until == "progress" and "progress" in scanner.wakes:
        return "progress"
    return None


def recent_activity(
    log: EventLog, state: RunState
) -> tuple[RunningTool | None, str | None, datetime | None]:
    """From the end of the log: the tool call still running (started, no
    result yet), and a progress line and activity time newer than the
    state's.

    The worker appends an event before its throttled status write, so the
    log can be ahead of ``status.json``; only strictly newer values are
    used, so progress never rolls back, and a question or an end keeps its
    own summary.
    """
    tail = log.tail(ACTIVITY_TAIL_BYTES)
    finished: set[str] = set()
    running: RunningTool | None = None
    for event in reversed(tail):
        if event.kind == "tool_result" and event.tool_use_id:
            finished.add(event.tool_use_id)
        elif event.kind == "tool_use":
            if event.tool_use_id not in finished:
                running = RunningTool(
                    tool=event.tool or "unknown",
                    input=clip(event.text, CHANGE_TEXT_CHARS)[0],
                    since=event.at,
                )
            break
        elif event.kind in ("turn_result", "finished"):
            break
    line = None
    if state.status == "working":
        for event in reversed(tail):
            found = progress_line(event)
            if found is None:
                continue
            if state.status_message_at is None or event.at > state.status_message_at:
                line = found
            break
    activity = [e.at for e in tail if e.is_activity]
    newest = max(activity) if activity else None
    if newest is not None and state.last_activity_at is not None:
        newest = newest if newest > state.last_activity_at else None
    return running, line, newest


def bounded_progress(text: str | None) -> str | None:
    if text is None or len(text) <= PROGRESS_CHARS:
        return text
    return text[: PROGRESS_CHARS - 1] + "…"


def result_route(task_id: str, part: str) -> Retrieval:
    return Retrieval(
        tool="claude_status",
        arguments={"task_id": task_id, "view": "result", "part": part},
    )


def outcome_of(
    run_dir: Path,
    state: RunState,
    *,
    final_text: str | None,
    last_narration: str | None,
    session_id: str | None,
    resumable: bool,
    diagnostics: list[str],
) -> Outcome:
    """The bounded :class:`Outcome` of a terminal task."""
    full = final_text
    inline = full is not None and len(full) <= INLINE_RESULT_CHARS
    shown = full if inline else (full[:RESULT_EXCERPT_CHARS] if full else None)
    handoff = state.handoff
    handoff_shown, handoff_chars, handoff_cut = clip(
        handoff.text if handoff else None, RESULT_EXCERPT_CHARS
    )
    if handoff is not None and handoff.text_chars:
        handoff_chars = handoff.text_chars
        handoff_cut = handoff_cut or handoff.text_chars > len(handoff_shown or "")
    error, _, error_cut = clip(state.error, ERROR_CHARS)
    narration = clip(last_narration, NARRATION_CHARS)[0]
    # One route: the final text first, then whatever else was cut (the
    # other parts take the same call with their own ``part``).
    retrieve = None
    if full is not None and not inline:
        retrieve = result_route(state.task_id, "final_text")
    elif error_cut:
        retrieve = result_route(state.task_id, "error")
    elif handoff_cut:
        retrieve = result_route(state.task_id, "handoff")
    return Outcome(
        outcome=state.status,
        termination_reason=state.termination_reason,
        final_text=shown,
        final_text_chars=len(full) if full is not None else None,
        final_text_truncated=full is not None and not inline,
        error=error,
        last_narration=narration,
        handoff_status=handoff.status if handoff else None,
        handoff_text=handoff_shown,
        handoff_text_chars=handoff_chars,
        handoff_truncated=handoff_cut,
        session_id=session_id,
        resumable=resumable,
        invocation_cost_usd=state.invocation_cost_usd,
        diagnostics=diagnostics,
        retrieve=retrieve,
    )


def diagnostics_of(state: RunState, final_text: str | None) -> list[str]:
    """Mechanically detectable inconsistencies in a finished task's record."""
    found: list[str] = []
    if state.status == "completed" and not (final_text or "").strip():
        found.append("completed_without_final_text")
    if state.is_terminal and state.turns_used == 0:
        found.append("no_model_turn_recorded")
    found.extend(state.diagnostics)
    return found
