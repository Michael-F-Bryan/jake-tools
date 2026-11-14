"""``artifact_read``: read-only check of one artifact's hosted content and visibility."""

from __future__ import annotations

from ..artifacts import ArtifactReadResult, SourceError, read
from ..config import Config
from . import Handler, Operation
from .errors import ToolError

DESCRIPTION = (
    "Read one Claude Artifact back, read-only: nothing is published, edited, "
    "shared or deleted. url is the artifact link (https://claude.ai/artifact/<id>). "
    "Returns the id, live version id, whether you own it, the visibility the "
    "service reports now (e.g. private), and the hosted file's size and sha256 "
    "with Claude Code's fixed document shell removed (the whole hosted "
    "response is not the file). Optional file_path (absolute .html path) adds "
    "matches_file; expected_version_id adds version_matches; output_path (a "
    "new file in an existing directory) saves the hosted file, shell removed. "
    "Reading sends the URL, not file contents, to Claude and costs a short "
    "Claude session. Uses an undocumented Claude Code mechanism."
)


def build(config: Config) -> Handler:
    async def artifact_read(
        url: str,
        file_path: str | None = None,
        expected_version_id: str | None = None,
        output_path: str | None = None,
    ) -> ArtifactReadResult:
        try:
            return await read(
                url,
                file_path=file_path,
                expected_version_id=expected_version_id,
                output_path=output_path,
            )
        except SourceError as exc:
            raise ToolError(
                "invalid_argument", str(exc), detail={"argument": exc.argument}
            ) from exc

    return artifact_read


OPERATION = Operation(
    name="artifact_read",
    capability="artifact",
    skill="artifact-publishing",
    description=DESCRIPTION,
    build=build,
    available=lambda config: config.artifacts_enabled.value,
)
