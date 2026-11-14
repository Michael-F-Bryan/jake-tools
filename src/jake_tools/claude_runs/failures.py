"""What a failed provider call tells the caller to do next.

A delegated task that ends ``failed``/``sdk_error`` used to carry only the
SDK's generic sentence. The runtime's own words sit elsewhere: the result's
``final_text``, its ``errors``, the exception's text. :func:`classify_result`
and :func:`classify_exception` read those and produce one :class:`Failure`:

- ``usage_limit``: the runtime reported that an account usage limit was
  reached (``You've hit your 5-hour limit``).
- ``authentication_or_access``: the runtime reported an authentication or
  access failure (``Failed to authenticate. API Error: 403 …``). Only the
  *report* is established. Credentials, the account, a proxy or WAF, or the
  network may each be at fault, and nothing here decides which.
- ``sdk_transport``: the Claude Agent SDK reported that the CLI process
  or its message stream failed (a failed connection, a non-zero CLI exit,
  an oversized or unparseable message), or the stream ended without a
  result. It says where the failure was reported, not why: nothing here
  shows whether the task's own work was affected.
- ``unknown``: anything else, including text that states two kinds of
  failure, and SDK exceptions that are not a transport report (a missing
  ``claude`` executable, a CLI error result). Unknown is a result, not a
  gap: the evidence is kept as it came.

Classification looks only at the runtime's own words, which are short; it
never reads a field's *name* as a verdict. A result whose ``subtype`` is
``success`` but whose ``is_error`` is true (the shape both reported failures
arrived in) classifies by what it says.

Facts and advice are separate. ``kind``, ``summary``, ``resets_at`` and
``evidence`` are what was reported; ``advice`` is inferred (``basis`` says
so) and names the next safe action. Nothing here retries, resumes, queues,
switches an account or changes a credential, and no advice grants authority.

``resets_at`` is set only when the reset wording itself carries a full date
(with year), a time and an explicit timezone (``UTC``, an IANA name or an
ISO offset) that name exactly one future instant within
:data:`MAX_RESET_AHEAD`. A bare time of day (``resets 5:30am (UTC)``) does not
establish a date: which day a limit resets depends on the limit's window,
which the text does not give, so it stays words in ``reset_text``, as do
abbreviations such as ``PST`` and wall-clock times a DST change skips or
repeats. No dollar cost is turned into an allowance.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field

FailureKind = Literal[
    "usage_limit", "authentication_or_access", "sdk_transport", "unknown"
]

NextStep = Literal[
    "wait_for_reset", "repair_authentication", "inspect_then_resume", "investigate"
]

REPORT_CHARS = 2000
"""Longest text read as a runtime report; a longer one is not an error line."""

EVIDENCE_CHARS = 600
"""Evidence text kept in a :class:`Failure` (``final_text`` holds the rest)."""

EVIDENCE_ERRORS = 5

MAX_RESET_AHEAD = timedelta(days=31)
"""A reset further out than this is not read as a reset time."""


class FailureEvidence(BaseModel):
    """What the runtime or SDK actually said, as it arrived (bounded).

    ``source`` is where it came from: the turn's ``result`` message, an
    ``exception`` the SDK raised, or the ``stream`` ending. ``subtype``,
    ``is_error`` and ``stop_reason`` are the result's own fields, so a
    caller can see ``subtype='success'`` beside ``is_error=true``. ``text``
    is the result text or exception text, cut to :data:`EVIDENCE_CHARS` (the
    task's ``final_text`` and ``error`` keep it in full). ``matched`` is the
    phrase that decided the kind, ``None`` when nothing recognised did.
    """

    model_config = ConfigDict(frozen=True)

    source: Literal["result", "exception", "stream"]
    subtype: str | None = None
    is_error: bool | None = None
    stop_reason: str | None = None
    errors: list[str] = Field(default_factory=list)
    text: str | None = None
    text_chars: int | None = None
    text_truncated: bool = False
    exception_type: str | None = None
    matched: str | None = None


class Advice(BaseModel):
    """The inferred next safe action. It is a recommendation, never an act."""

    model_config = ConfigDict(frozen=True)

    next_step: NextStep
    text: str
    basis: Literal["inferred"] = "inferred"


class Failure(BaseModel):
    """A classified provider/SDK failure: ``kind`` and evidence, then advice.

    ``summary`` restates what was reported, in one line. ``resets_at`` (UTC)
    is set only for a ``usage_limit`` whose reset time was reliably
    interpreted (see the module doc); ``reset_text`` is the reset wording as
    reported, whether or not it could be interpreted. ``http_status`` is the
    status the report names, not a diagnosis. ``observed_at`` is when the
    worker saw it.
    """

    model_config = ConfigDict(frozen=True)

    kind: FailureKind
    summary: str
    observed_at: datetime
    resets_at: datetime | None = None
    reset_text: str | None = None
    http_status: int | None = None
    evidence: FailureEvidence
    advice: Advice


class Cooldown(BaseModel):
    """A reported usage limit for one account context, as last observed.

    ``resets_at`` is the instant the report named; ``observed_at`` is when a
    worker saw it, and ``task_id`` the task that saw it. This is a report,
    not a measurement: it says nothing of remaining allowance, and a caller
    may still start work.
    """

    model_config = ConfigDict(frozen=True)

    account_context: str
    resets_at: datetime
    observed_at: datetime
    task_id: str
    advisory: bool = True


# --- recognising reports ---------------------------------------------------------

_USAGE_LIMIT = (
    re.compile(r"you(?:'|’)ve hit your (?:[\w-]+ )*limit\b", re.IGNORECASE),
    re.compile(r"(?:claude(?: ai)? )?usage limit reached\b", re.IGNORECASE),
)
_AUTHENTICATION = (
    re.compile(r"failed to authenticate\b", re.IGNORECASE),
    re.compile(r"(?:401|403)\b"),
    re.compile(r"invalid api key\b", re.IGNORECASE),
    re.compile(r"oauth token has expired\b", re.IGNORECASE),
)
_API_ERROR_PREFIX = re.compile(r"^api error:\s*", re.IGNORECASE)
_HTTP_STATUS = re.compile(r"\bAPI Error: (\d{3})\b")
_AUTH_ANYWHERE = re.compile(
    r"failed to authenticate|api error: (?:401|403)|invalid api key"
    r"|oauth token has expired",
    re.IGNORECASE,
)
_LIMIT_ANYWHERE = re.compile(
    r"you(?:'|’)ve hit your (?:[\w-]+ )*limit|usage limit reached", re.IGNORECASE
)
_RESETS = re.compile(r"\bresets?\s+([^\n]{1,80})", re.IGNORECASE)
_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_RESET_ISO = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})$"
)
_RESET_DATED = re.compile(
    r"^(?:(?P<iso>\d{4}-\d{2}-\d{2})|(?P<month>[A-Za-z]{3,9})\.? "
    r"(?P<day>\d{1,2}),? (?P<year>\d{4}))[,T ]\s*(?:at\s+)?"
    r"(?:(?P<h12>\d{1,2})(?::(?P<m12>\d{2}))?\s*(?P<half>am|pm)"
    r"|(?P<h24>\d{2}):(?P<m24>\d{2}))\s*\((?P<zone>[^)]{1,40})\)$",
    re.IGNORECASE,
)


def _recognise(text: str | None) -> tuple[FailureKind, str] | None:
    """The kind a runtime report states and the phrase that says it.

    The report must open with the statement (after an optional ``API Error:``)
    and be short, so an agent's prose that quotes an error is not read as
    one. If the same report also states the *other* kind anywhere, it is
    ambiguous and recognises nothing.
    """
    report = (text or "").strip()
    if not report or len(report) > REPORT_CHARS:
        return None
    bare = _API_ERROR_PREFIX.sub("", report)
    groups: list[tuple[FailureKind, tuple[re.Pattern[str], ...]]] = [
        ("usage_limit", _USAGE_LIMIT),
        ("authentication_or_access", _AUTHENTICATION),
    ]
    found: tuple[FailureKind, str] | None = None
    for kind, patterns in groups:
        for pattern in patterns:
            if match := pattern.match(report) or pattern.match(bare):
                found = (kind, match.group(0))
                break
    if found is None:
        return None
    other = _AUTH_ANYWHERE if found[0] == "usage_limit" else _LIMIT_ANYWHERE
    return None if other.search(report) else found


def parse_reset(text: str, observed_at: datetime) -> tuple[str | None, datetime | None]:
    """The reset wording in ``text`` and the instant it reliably names.

    The wording is whatever follows ``resets``, even when it cannot be
    interpreted. The instant is set only for a full date with year, a time
    and a timezone that is ``UTC``/``GMT``, an IANA name or an ISO offset
    (``2026-11-17T05:30:00Z``, ``Nov 17, 2026, 5:30am (UTC)``), and only
    when it falls after ``observed_at`` and within :data:`MAX_RESET_AHEAD`.
    A bare time of day, a date without a year, an abbreviation and a
    wall-clock time a DST change skips or repeats give no instant.
    """
    found = _RESETS.search(text)
    if found is None:
        return None, None
    wording = found.group(1).strip().rstrip(".")
    if observed_at.tzinfo is None:
        return wording, None
    instant = _instant(wording)
    if instant is None or not (observed_at < instant <= observed_at + MAX_RESET_AHEAD):
        return wording, None
    return wording, instant.astimezone(UTC)


def _instant(wording: str) -> datetime | None:
    if _RESET_ISO.match(wording):
        try:
            return datetime.fromisoformat(wording)
        except ValueError:
            return None
    dated = _RESET_DATED.match(wording)
    if dated is None:
        return None
    zone = _zone(dated["zone"].strip())
    if zone is None:
        return None
    try:
        if dated["iso"]:
            year, month, day = (int(p) for p in dated["iso"].split("-"))
        else:
            name = dated["month"].lower()
            months = [
                m for m in _MONTHS if m == name or (len(name) == 3 and m[:3] == name)
            ]
            if len(months) != 1:
                return None
            year, month, day = (
                int(dated["year"]),
                _MONTHS.index(months[0]) + 1,
                int(dated["day"]),
            )
        if dated["half"]:
            hour, minute = int(dated["h12"]), int(dated["m12"] or 0)
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + (12 if dated["half"].lower() == "pm" else 0)
        else:
            hour, minute = int(dated["h24"]), int(dated["m24"])
        local = datetime(year, month, day, hour, minute, tzinfo=zone)
    except ValueError:
        return None
    if local.replace(fold=0).utcoffset() != local.replace(fold=1).utcoffset():
        return None  # a repeated wall-clock time
    if local.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != local.replace(
        tzinfo=None
    ):
        return None  # a skipped one
    return local


def _zone(name: str) -> ZoneInfo | None:
    if name.upper() in ("UTC", "GMT"):
        return ZoneInfo("UTC")
    if "/" not in name or not re.fullmatch(r"[A-Za-z]+(?:/[A-Za-z0-9_+-]+)+", name):
        return None  # an abbreviation (PST, CST, IST) names no one zone
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError, ValueError:
        return None


# --- building a Failure -----------------------------------------------------------


def classify_result(
    *,
    subtype: str,
    is_error: bool,
    text: str | None,
    errors: list[str],
    stop_reason: str | None,
    observed_at: datetime,
) -> Failure:
    """Classify an error result by what it says, whatever its ``subtype``."""
    found = [r for t in (text, *errors) if (r := _recognise(t))]
    # Reports of two kinds in one envelope are ambiguous, as within a report.
    recognised = found[0] if len({kind for kind, _ in found}) == 1 else None
    clipped, chars, cut = _clip(text)
    evidence = FailureEvidence(
        source="result",
        subtype=subtype,
        is_error=is_error,
        stop_reason=stop_reason,
        errors=[_clip(e)[0] or "" for e in errors[:EVIDENCE_ERRORS]],
        text=clipped,
        text_chars=chars,
        text_truncated=cut,
        matched=recognised[1] if recognised else None,
    )
    return _failure(recognised, evidence, report=text, errors=errors, now=observed_at)


def classify_exception(exc: BaseException, *, observed_at: datetime) -> Failure:
    """Classify an exception: by what it says, else by an SDK transport type.

    An SDK result error carries the CLI's result text and ``errors``; those
    are read like a result's. Other exceptions are ``unknown`` unless they
    are one of the SDK's transport/parser/process errors (:func:`is_transport_error`).
    """
    text = str(exc).strip() or None
    reports = [
        text,
        getattr(exc, "result", None),
        *(getattr(exc, "errors", None) or []),
    ]
    found = [r for t in reports if isinstance(t, str) and (r := _recognise(t))]
    recognised = found[0] if len({kind for kind, _ in found}) == 1 else None
    if recognised is None and not found and is_transport_error(exc):
        recognised = ("sdk_transport", type(exc).__name__)
    clipped, chars, cut = _clip(text)
    evidence = FailureEvidence(
        source="exception",
        text=clipped,
        text_chars=chars,
        text_truncated=cut,
        exception_type=type(exc).__name__,
        matched=recognised[1] if recognised else None,
    )
    texts = [t for t in reports if isinstance(t, str)]
    return _failure(recognised, evidence, report=None, errors=texts, now=observed_at)


def classify_stream_end(*, observed_at: datetime) -> Failure:
    """The CLI's stream closed without a final result message."""
    evidence = FailureEvidence(
        source="stream", text="the stream ended without a result message"
    )
    return _failure(("sdk_transport", "no result"), evidence, None, [], observed_at)


TRANSPORT_ERRORS = frozenset(
    {"CLIConnectionError", "CLIJSONDecodeError", "MessageParseError", "ProcessError"}
)


def is_transport_error(exc: BaseException) -> bool:
    """Whether ``exc`` is exactly one of the SDK's transport-boundary errors.

    A failed connection, a non-zero CLI exit, a message the SDK could not
    frame or parse. Matched on the exact type and its package, so this module
    does not import the SDK (``claude.py`` is the only importer) and its
    subclasses are not swept in: ``CLINotFoundError`` (no ``claude``
    executable) and ``ResultError`` (the CLI's own error result) say
    something else and stay ``unknown`` unless their text is recognised.
    """
    cls = type(exc)
    return (
        cls.__module__.partition(".")[0] == "claude_agent_sdk"
        and cls.__name__ in TRANSPORT_ERRORS
    )


def _clip(text: str | None) -> tuple[str | None, int | None, bool]:
    if text is None:
        return None, None, False
    if len(text) <= EVIDENCE_CHARS:
        return text, len(text), False
    return text[: EVIDENCE_CHARS - 1] + "…", len(text), True


def _failure(
    recognised: tuple[FailureKind, str] | None,
    evidence: FailureEvidence,
    report: str | None,
    errors: list[str],
    now: datetime,
) -> Failure:
    kind: FailureKind = recognised[0] if recognised else "unknown"
    base = {"observed_at": now, "evidence": evidence}
    # The reports that were recognised carry any reset wording or status.
    source = next((r for r in (report, *errors) if _recognise(r) is not None), None)
    if recognised is None:
        source = None
    first = (source or "").strip().splitlines()[0][:200] if source else ""
    if kind == "usage_limit":
        wording, resets_at = parse_reset(source or "", now)
        if resets_at is not None:
            when = f"It reports a reset at {resets_at:%Y-%m-%d %H:%M} UTC."
        elif wording:
            when = (
                f"The reset wording {wording!r} does not name one date and "
                "time reliably, so none is claimed."
            )
        else:
            when = "No reset time was reported."
        return Failure(
            kind=kind,
            summary=f"Usage limit reported: {first}",
            resets_at=resets_at,
            reset_text=wording,
            advice=Advice(
                next_step="wait_for_reset",
                text=(
                    "The runtime reported that an account usage limit was reached; "
                    "starting more work before it resets is expected to fail the "
                    f"same way. {when} Nothing was retried or resumed, and no "
                    "remaining allowance is known."
                ),
            ),
            **base,
        )
    if kind == "authentication_or_access":
        status = _HTTP_STATUS.search(source or "")
        return Failure(
            kind=kind,
            summary=f"Authentication or access failure reported: {first}",
            http_status=int(status.group(1)) if status else None,
            advice=Advice(
                next_step="repair_authentication",
                text=(
                    "The runtime reported an authentication or access failure. "
                    "The report does not say whether credentials, the account, "
                    "a proxy or web firewall, or the network is at fault. "
                    "Check the worker's Claude authentication and access before "
                    "starting more work; no credential, account or provider "
                    "was changed and nothing was retried."
                ),
            ),
            **base,
        )
    if kind == "sdk_transport":
        what = evidence.exception_type or "the stream ending"
        return Failure(
            kind=kind,
            summary=f"Claude SDK/CLI reported a failure: {what}",
            advice=Advice(
                next_step="inspect_then_resume",
                text=(
                    "The SDK reported that the CLI process or its message stream "
                    "failed; that does not show whether the task's work was "
                    "affected or why it failed. Inspect the recorded transcript, "
                    "session and any files (claude_status, claude_transcript) "
                    "before deciding whether to resume; do not regenerate "
                    "completed work blindly. Nothing was retried or resumed."
                ),
            ),
            **base,
        )
    return Failure(
        kind="unknown",
        summary="Failure not recognised; the original evidence is preserved",
        advice=Advice(
            next_step="investigate",
            text=(
                "This failure matches no known report, so no cause or reset "
                "time is claimed. Read evidence, error and final_text, and "
                "inspect the task's work, before deciding what to do."
            ),
        ),
        **base,
    )
