"""``artifact_publish``: publish an existing HTML file as a NEW private Claude Artifact.

Always listed. It sends the file's contents to claude.ai through an
undocumented Claude Code mechanism, so authority is held by the guard, not by
hiding the tool. :mod:`jake_tools.artifacts` does the work.
"""

from __future__ import annotations

from ..artifacts import ArtifactPublishResult, SourceError, publish
from ..config import Config
from . import Handler, Operation
from .errors import ToolError

DESCRIPTION = (
    "Publish an existing, self-contained .html/.htm file (absolute path as "
    "jake-tools sees it, UTF-8, at most 1 MB) as a NEW artifact on claude.ai "
    "through the signed-in Claude account, then read it back in a separate "
    "session. The file's contents leave this machine for Anthropic; it is "
    "never edited or authored. Returns the artifact URL, artifact id, version, "
    "the source file's size and sha256, and a readback: the visibility the "
    "service reported, and the hosted file's size and sha256 with Claude "
    "Code's fixed document shell removed. `verified` is true only if that "
    "hosted file equals the local file byte for byte and the readback "
    "reported the artifact as private; otherwise `problems` says why and the "
    "artifact still exists. New artifacts start private and this tool never "
    "shares one or changes sharing. It creates a new artifact every call (use "
    "artifact_update for a new version of an existing one) and never retries. Uses an undocumented Claude Code mechanism "
    "that can change without notice."
)


def build(config: Config) -> Handler:
    async def artifact_publish(file_path: str) -> ArtifactPublishResult:
        try:
            return await publish(file_path)
        except SourceError as exc:
            raise ToolError(
                "invalid_argument", str(exc), detail={"argument": exc.argument}
            ) from exc

    return artifact_publish


OPERATION = Operation(
    name="artifact_publish",
    skill="artifact-publishing",
    description=DESCRIPTION,
    build=build,
)
