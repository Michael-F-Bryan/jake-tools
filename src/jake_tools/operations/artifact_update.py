"""``artifact_update``: publish the next version of an existing private artifact.

A separate ``artifact_update`` capability from ``artifact``: it overwrites the
live content of something that already exists, so granting publish/read does
not grant it. Also needs ``artifacts.enabled``. :mod:`jake_tools.artifacts`
does the work.
"""

from __future__ import annotations

from ..artifacts import ArtifactUpdateResult, SourceError, update
from ..config import Config
from . import Handler, Operation
from .errors import ToolError

DESCRIPTION = (
    "Publish the contents of an existing .html/.htm file (absolute path as "
    "jake-tools sees it, UTF-8, at most 1 MB) as the NEXT VERSION of an existing "
    "private Claude Artifact you own, keeping its URL and id. You must pass "
    "expected_version_id, the live version you believe it is at (from "
    "artifact_read or artifact_publish). Before anything is written, a separate "
    "read checks that the artifact is yours, private and at that version; a "
    "stale version fails with plan_stale and writes nothing. Only private "
    "artifacts are updated: nothing else is touched and sharing is never "
    "changed. Then a separate read verifies the result: same artifact, still "
    "private, new version, and the hosted file (Claude Code's fixed document "
    "shell removed) equals the local file byte for byte. `verified` is true only "
    "then; otherwise `problems` says why and the artifact HAS changed. The write "
    "is denied unless its own read of the target was observed as yours, private "
    "and at that version. Not an atomic compare-and-swap: an edit landing between "
    "that read and the write cannot be detected. Never retries; after an UNKNOWN "
    "error read the artifact before doing anything. The file's contents are "
    "sent to Anthropic. Uses an undocumented Claude Code mechanism that can "
    "change without notice."
)


def build(config: Config) -> Handler:
    async def artifact_update(
        url: str, file_path: str, expected_version_id: str
    ) -> ArtifactUpdateResult:
        try:
            return await update(url, file_path, expected_version_id)
        except SourceError as exc:
            raise ToolError(
                "invalid_argument", str(exc), detail={"argument": exc.argument}
            ) from exc

    return artifact_update


OPERATION = Operation(
    name="artifact_update",
    capability="artifact_update",
    skill="artifact-publishing",
    description=DESCRIPTION,
    build=build,
    available=lambda config: config.artifacts_enabled.value,
)
