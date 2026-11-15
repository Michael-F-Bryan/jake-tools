"""Publish an existing HTML file as a private Claude Artifact and read it back.

See :mod:`.service` for the route and its caveats, and ``docs/claude-tools.md``
for the evidence.
"""

from .models import (
    ArtifactPublishResult,
    ArtifactReadResult,
    ArtifactUpdateResult,
    Readback,
    SourceReceipt,
)
from .service import SourceError, publish, read, update

__all__ = [
    "ArtifactPublishResult",
    "ArtifactReadResult",
    "ArtifactUpdateResult",
    "Readback",
    "SourceError",
    "SourceReceipt",
    "publish",
    "read",
    "update",
]
