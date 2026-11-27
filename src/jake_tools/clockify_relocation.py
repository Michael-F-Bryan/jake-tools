"""Moving a Jira-keyed Clockify task to the project Jira places it under.

Clockify's API cannot re-parent a task: ``PUT .../projects/{p}/tasks/{t}``
(``UpdateTaskRequest``) has no ``projectId``. Time entries can be re-pointed
(``PUT /time-entries/{id}``, ``UpdateTimeEntryRequest``, a full replace of
start/end/billable/description/projectId/taskId/tagIds/customFields/type). So
a relocation is a recoverable replacement, never a delete:

1. create a *staging* task in the destination project, named with a marker
   that carries the Jira key and the source task ID and does not claim the
   key (:func:`staged_task_name`), copying the source's writable task fields;
2. re-point every time entry from the source task to the staging task,
   echoing every writable field and comparing the whole record read back;
3. retire the source (:func:`retired_task_name`, DONE): it keeps its name
   after the marker and points at the staging task, so it no longer claims
   the key but the interrupted state stays discoverable;
4. promote the staging task to the canonical ``KEY summary`` name, last.

A fresh preview recognises only these exact marker relations as an
interrupted relocation; any other duplicate stays a conflict.

This module holds the record-level rules (markers, fingerprints, payloads,
comparisons). Planning and apply orchestration live in
:mod:`jake_tools.clockify_jira_sync`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .clockify import JsonObject, TaskRateKind

_ID = r"[A-Za-z0-9_-]{1,128}"
_KEY = r"[A-Z][A-Z0-9]+-\d+"
_STAGED = re.compile(rf"\[relocating ({_KEY}) from ({_ID})(?: #([0-9a-f]{{16}}))?\] ")
_RETIRED = re.compile(rf"\[relocated ({_KEY}) to ({_ID})\] ")


class RelocationMarker(BaseModel):
    """What a staging or retired task's name says about its relocation.

    ``other_task_id`` is the source task for a staging task, and the staging
    (replacement) task for a retired one. ``baseline`` is a staging task's
    :func:`relocation_baseline`, recorded before any entry moved; ``None`` on
    a staging task means a person removed it to accept the current records.
    """

    model_config = ConfigDict(frozen=True)

    state: Literal["staged", "retired"]
    jira_key: str
    other_task_id: str
    baseline: str | None = None


def staged_task_name(
    jira_key: str, source_task_id: str, name: str, *, baseline: str
) -> str:
    return f"[relocating {jira_key} from {source_task_id} #{baseline}] {name}"


def retired_task_name(jira_key: str, staged_task_id: str, name: str) -> str:
    return f"[relocated {jira_key} to {staged_task_id}] {name}"


def relocation_marker(name: str) -> RelocationMarker | None:
    for state, pattern in (("staged", _STAGED), ("retired", _RETIRED)):
        if match := pattern.match(name):
            return RelocationMarker(
                state=state,  # type: ignore[arg-type]
                jira_key=match.group(1),
                other_task_id=match.group(2),
                baseline=match.group(3) if state == "staged" else None,
            )
    return None


def record_fingerprint(record: JsonObject) -> str:
    """SHA-256 over the record's canonical JSON: any field change changes it."""
    canonical = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def relocation_baseline(source_task: JsonObject, entries: Iterable[JsonObject]) -> str:
    """What a relocation must preserve, as 16 hex digits for the marker.

    Covers the source task's fields other than identity (name, status, ...)
    and every entry it carries minus its placement. Recorded in the staging
    task's name before the first entry moves, it is the durable baseline a
    later preview checks: entries already moved, or moved by a write whose
    outcome was uncertain, resume only if everything still matches it.
    """
    canonical = {
        "task": {
            key: value
            for key, value in source_task.items()
            if key not in TASK_IDENTITY_FIELDS
        },
        "entries": {
            str(entry["id"]): record_fingerprint(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in ENTRY_PLACEMENT_FIELDS
                }
            )
            for entry in entries
        },
    }
    return record_fingerprint(canonical)[:16]


def changed_fields(
    before: JsonObject, after: JsonObject, *, ignore: Iterable[str]
) -> list[str]:
    """Top-level fields that differ, unknown ones included (fail closed)."""
    skip = set(ignore)
    keys = (before.keys() | after.keys()) - skip
    return sorted(key for key in keys if before.get(key) != after.get(key))


# --- Time entries ---------------------------------------------------------------

# Fields a relocation is allowed to change on an entry. Everything else,
# including read-only derived fields such as hourlyRate/costRate, must read
# back exactly as it was.
ENTRY_PLACEMENT_FIELDS = ("projectId", "taskId")
_MOVABLE_ENTRY_TYPES = {"REGULAR", "BREAK"}


def entry_blocker(entry: JsonObject) -> str | None:
    """Why this entry cannot be moved safely, or ``None``.

    A running timer, a locked (approved) entry, an entry type the update API
    does not accept, or a shape this package cannot echo back field for field.
    """
    entry_id = entry.get("id")
    interval = entry.get("timeInterval")
    if not isinstance(interval, dict) or not isinstance(interval.get("start"), str):
        return f"time entry {entry_id} has no readable start time"
    if interval.get("end") is None:
        return (
            f"time entry {entry_id} is a running timer; stop it in Clockify, "
            "then preview again"
        )
    if not isinstance(interval.get("end"), str):
        return f"time entry {entry_id} has no readable end time"
    if entry.get("isLocked") is True:
        return (
            f"time entry {entry_id} is locked (approved or in a locked period); "
            "unlock or withdraw it in Clockify, then preview again"
        )
    entry_type = entry.get("type")
    if entry_type not in (None, "", *_MOVABLE_ENTRY_TYPES):
        return (
            f"time entry {entry_id} has type {entry_type!r}, which the Clockify "
            "time-entry update API does not accept"
        )
    if not isinstance(entry.get("billable"), bool):
        return f"time entry {entry_id} has no readable billable flag"
    if not isinstance(entry.get("description", ""), str):
        return f"time entry {entry_id} has no readable description"
    tags = entry.get("tagIds") or []
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        return f"time entry {entry_id} has unreadable tags"
    fields = entry.get("customFieldValues") or []
    if not isinstance(fields, list) or not all(
        isinstance(f, dict) and isinstance(f.get("customFieldId"), str) for f in fields
    ):
        return f"time entry {entry_id} has unreadable custom field values"
    return None


def entry_move_payload(
    entry: JsonObject, *, project_id: str, task_id: str
) -> JsonObject:
    """The full ``UpdateTimeEntryRequest`` for ``entry`` with a new placement.

    Every writable field is echoed from the entry as read (the update is a
    full replace). Only call on an entry :func:`entry_blocker` accepted.
    """
    interval = entry["timeInterval"]
    assert isinstance(interval, dict)
    custom_values = entry.get("customFieldValues") or []
    assert isinstance(custom_values, list)
    payload: JsonObject = {
        "start": interval["start"],
        "end": interval["end"],
        "billable": entry["billable"],
        "description": entry.get("description", ""),
        "projectId": project_id,
        "taskId": task_id,
        "tagIds": list(entry.get("tagIds") or []),  # type: ignore[arg-type]
        "customFields": [
            {"customFieldId": value["customFieldId"], "value": value.get("value")}
            for value in custom_values
            if isinstance(value, dict)
        ],
    }
    if entry.get("type"):
        payload["type"] = entry["type"]
    return payload


# --- Tasks ----------------------------------------------------------------------

# Task fields a replacement legitimately differs in. ``duration`` is the time
# tracked on the task, derived from its entries, so it moves with them.
TASK_IDENTITY_FIELDS = ("id", "projectId", "name", "status", "duration")
# What TaskRequestV1 (create) accepts besides name/status. It has no billable.
_CREATE_FIELDS = ("assigneeIds", "userGroupIds", "estimate", "budgetEstimate")
# What UpdateTaskRequest accepts besides name/status.
_UPDATE_FIELDS = ("billable", *_CREATE_FIELDS)
TASK_RATE_FIELDS: tuple[tuple[str, TaskRateKind], ...] = (
    ("hourlyRate", "hourly-rate"),
    ("costRate", "cost-rate"),
)


def task_create_payload(source: JsonObject, *, name: str) -> JsonObject:
    payload: JsonObject = {"name": name, "status": "ACTIVE"}
    payload.update(
        (field, source[field])
        for field in _CREATE_FIELDS
        if source.get(field) is not None
    )
    return payload


def task_update_payload(task: JsonObject, *, name: str, status: str) -> JsonObject:
    """A full ``UpdateTaskRequest`` that changes only name and status."""
    payload: JsonObject = {"name": name, "status": status}
    payload.update(
        (field, task[field]) for field in _UPDATE_FIELDS if task.get(field) is not None
    )
    return payload


def task_rate_amount(task: JsonObject, field: str) -> int | None:
    rate = task.get(field)
    if isinstance(rate, dict) and isinstance(rate.get("amount"), int):
        return rate["amount"]
    return None
