"""Advisory usage-limit cooldowns, shared by runs of one declared account context.

When a worker observes a ``usage_limit`` failure whose reset wording names one
future instant (a full date with year, time and zone; see
:mod:`.failures`), it records that instant for its run's ``account_context``,
so another conversation can see a known limit before launching more work.

Narrow by design:

- **The context is declared, not discovered.** ``claude.account_context`` is
  an operator's non-secret label, copied into ``spec.json`` when a task is
  launched so a later config change cannot move a run into another context.
  Nothing here inspects credentials, runs the CLI, or decides that two labels
  share a quota. No label, no record and no lookup.
- **Only an observed, dated reset persists.** An undated reset stays text on
  the failure; no expiry is invented. The record is the observation as
  reported: ``resets_at`` and when it was ``observed_at``.
- **Advisory.** A start reports a live cooldown beside its result; it never
  blocks, queues, retries or resumes. An expired record is ignored (and
  purged by the next write); an unreadable one reads as no cooldown.
- **A later success clears it, and stays cleared.** A run whose prompt was
  sent *after* the observation and which got a model turn answered shows the
  limit is no longer in force. The context's file keeps that prompt's time as
  a watermark (``answered_after``), so an observation made before it that is
  written late (a slow worker) is stale and dropped, not resurrected. A
  record newer than the prompt is left alone. A watermark is kept for
  :data:`WATERMARK_RETENTION` and then purged with its file.

Files live in ``<runs_dir>/.cooldowns/<sha256(label)[:32]>.json`` (the label
is hashed for the filename, and kept in the record and checked on read).
Writes are read-compare-write under ``.cooldowns/.lock`` with an atomic
replace; reads need no lock because a reader sees the old file or the new
one. The lock is a leaf: nothing is taken while it is held.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .failures import Cooldown, Failure
from .rundir import DIR_MODE, file_lock, read_spec, utc_now, write_model

log = logging.getLogger(__name__)

COOLDOWNS_DIR = ".cooldowns"
LOCK_FILE = ".lock"
WATERMARK_RETENTION = timedelta(days=7)
"""How long a context's answered-prompt watermark outlives its cooldown: far
longer than any task's deadline, so no late write can arrive after it goes."""


class _State(BaseModel):
    """One context's file: its cooldown, if any, and the answered-prompt watermark."""

    model_config = ConfigDict(frozen=True)

    account_context: str
    cooldown: Cooldown | None = None
    answered_after: datetime | None = None


def for_failed_run(run_dir: Path, failure: Failure | None) -> Cooldown | None:
    """The cooldown now known for a task that failed on a usage limit.

    The context is the one the task was launched under (``spec.json``), and
    the answer is whatever is on record now, which may be a newer
    observation than the task's own, or none (expired, cleared, undated).
    """
    if failure is None or failure.kind != "usage_limit":
        return None
    try:
        context = read_spec(run_dir).account_context
    except OSError, ValueError:
        return None
    return active(run_dir.parent, context)


def _path(runs_dir: Path, context: str) -> Path:
    digest = hashlib.sha256(context.encode()).hexdigest()[:32]
    return runs_dir / COOLDOWNS_DIR / f"{digest}.json"


def _read(path: Path, context: str) -> _State | None:
    try:
        state = _State.model_validate_json(path.read_bytes())
    except OSError, ValueError:
        return None
    return state if state.account_context == context else None


def active(
    runs_dir: Path, context: str | None, *, now: datetime | None = None
) -> Cooldown | None:
    """The unexpired cooldown recorded for ``context``, if any."""
    if context is None:
        return None
    state = _read(_path(runs_dir, context), context)
    if state is None or state.cooldown is None:
        return None
    return state.cooldown if state.cooldown.resets_at > (now or utc_now()) else None


def record(
    runs_dir: Path,
    context: str | None,
    *,
    resets_at: datetime,
    observed_at: datetime,
    task_id: str,
    now: datetime | None = None,
) -> Cooldown | None:
    """Persist an observed cooldown; the newest observation of a context wins.

    Returns what is now on record for ``context``: ``None`` when there is no
    label, the reset has passed, or the observation is stale because a prompt
    sent at or after ``observed_at`` was answered since.
    """
    moment = now or utc_now()
    if context is None or resets_at <= moment:
        return None
    new = Cooldown(
        account_context=context,
        resets_at=resets_at,
        observed_at=observed_at,
        task_id=task_id,
    )
    directory = runs_dir / COOLDOWNS_DIR
    directory.mkdir(mode=DIR_MODE, exist_ok=True)
    with file_lock(directory / LOCK_FILE):
        _purge_locked(directory, moment)
        path = _path(runs_dir, context)
        state = _read(path, context) or _State(account_context=context)
        if state.answered_after is not None and observed_at <= state.answered_after:
            return None  # a later success already showed this limit is over
        current = active(runs_dir, context, now=moment)
        if current is not None and current.observed_at >= observed_at:
            return current
        write_model(path, state.model_copy(update={"cooldown": new}))
    return new


def clear(
    runs_dir: Path,
    context: str | None,
    *,
    interaction_at: datetime,
    now: datetime | None = None,
) -> bool:
    """Note an answered prompt sent at ``interaction_at``; drop what it outdates.

    The watermark moves forward only. A cooldown observed before
    ``interaction_at`` is removed; one observed at or after it is a newer
    fact and stays. Returns whether a cooldown was removed.
    """
    if context is None:
        return False
    moment = now or utc_now()
    directory = runs_dir / COOLDOWNS_DIR
    directory.mkdir(mode=DIR_MODE, exist_ok=True)
    with file_lock(directory / LOCK_FILE):
        _purge_locked(directory, moment)
        path = _path(runs_dir, context)
        state = _read(path, context) or _State(account_context=context)
        mark = max(filter(None, (state.answered_after, interaction_at)))
        cooldown = state.cooldown
        removed = (
            cooldown is not None
            and cooldown.resets_at > moment
            and cooldown.observed_at < interaction_at
        )
        if cooldown is not None and cooldown.observed_at < interaction_at:
            cooldown = None
        write_model(
            path,
            state.model_copy(update={"cooldown": cooldown, "answered_after": mark}),
        )
        return removed


def _purge_locked(directory: Path, now: datetime) -> None:
    """Remove files with nothing left to say: no live cooldown, watermark old."""
    for path in directory.glob("*.json"):
        try:
            state = _State.model_validate_json(path.read_bytes())
        except OSError:
            continue
        except ValueError:
            state = None  # unreadable: not a cooldown, not a watermark
        live = state is not None and (
            (state.cooldown is not None and state.cooldown.resets_at > now)
            or (
                state.answered_after is not None
                and state.answered_after + WATERMARK_RETENTION > now
            )
        )
        if not live:
            with contextlib.suppress(OSError):
                path.unlink()
