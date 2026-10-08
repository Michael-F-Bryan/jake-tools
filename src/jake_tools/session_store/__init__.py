"""Read-only adapter over a Hermes session store (phase 3; models only today).

All store access will live here; the rest of the package sees typed records
and a ``SessionStoreError`` hierarchy. Nothing else imports ``sqlite3``.
The adapter itself, its fingerprint table and its fixtures are phase 3 work
and are not started until the Hermes schema and lock semantics have been
read; only the result contracts in :mod:`jake_tools.session_store.models`
exist now.
"""

from __future__ import annotations

from .models import (
    SessionLineage,
    SessionRecord,
    SessionTimeline,
    StoreCoverage,
)

__all__ = ["SessionLineage", "SessionRecord", "SessionTimeline", "StoreCoverage"]
