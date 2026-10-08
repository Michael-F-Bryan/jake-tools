"""Result contracts for the ``session_timeline`` tool (phase 3).

These are the shapes the design fixes; the adapter that fills them is not
written yet. Nothing here fabricates: a record's role, type and provenance
are whatever the store recorded, and an unknown provenance is reported as
``"unknown"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionRecord(BaseModel):
    """One timestamped record within the requested interval."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    session_id: str
    parent_session_id: str | None = None
    recorded_at: datetime
    display_at: datetime
    role: str
    record_type: str
    source: str = "unknown"
    platform: str | None = None
    content: str | dict[str, Any] | None = None
    truncated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionLineage(BaseModel):
    """Parent/child structure for every session the result touched."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    parent_session_id: str | None = None
    child_session_ids: tuple[str, ...] = ()
    started_at: datetime | None = None
    ended_at: datetime | None = None
    platform: str | None = None


class StoreCoverage(BaseModel):
    """What the adapter saw while serving one request."""

    model_config = ConfigDict(frozen=True)

    store_path_fingerprint: str
    schema_fingerprint: str
    journal_mode: str
    observed_at: datetime
    busy_retries: int = 0
    snapshot_copy_used: bool = False
    notes: tuple[str, ...] = ()


class SessionTimeline(BaseModel):
    """The ``session_timeline`` tool's result."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime
    timezone: str
    records: tuple[SessionRecord, ...]
    sessions: tuple[SessionLineage, ...]
    count: int
    next_cursor: str | None = None
    coverage: StoreCoverage
