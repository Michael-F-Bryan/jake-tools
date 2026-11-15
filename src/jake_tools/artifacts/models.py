"""What the artifact operations return."""

from __future__ import annotations

from pydantic import BaseModel, Field

MECHANISM = (
    "Claude Code's Artifact tool, enabled for this session only by the "
    "undocumented CLAUDE_CODE_ARTIFACT environment variable. Not a documented "
    "or supported Anthropic option; it may change without notice."
)


class SourceReceipt(BaseModel):
    """The local file as it was when the call started."""

    path: str
    bytes: int
    sha256: str


class Readback(BaseModel):
    """What the artifact service held, read in its own session.

    ``visibility`` is what the read reported for this artifact now (for
    example ``private``); it is evidence of the current state, not the
    documented default that new artifacts start private. ``bytes``/``sha256``
    describe the hosted file with Claude Code's fixed document shell removed;
    both are ``None`` when the shell was not recognised, in which case nothing
    is claimed about the content.
    """

    url: str
    artifact_id: str | None = None
    version_id: str | None = None
    owned_by_you: bool | None = None
    visibility: str | None = None
    private: bool | None = None
    header: str | None = None
    shell_removed: bool = False
    bytes: int | None = None
    sha256: str | None = None


class ArtifactReadResult(Readback):
    """``artifact_read``: a read-only check of one artifact."""

    matches_file: bool | None = None
    """Whether the hosted file, shell removed, equals ``file_path`` byte for
    byte (``None`` when no file was given or the hosted content could not be
    isolated)."""
    version_matches: bool | None = None
    """Whether the live version is ``expected_version_id`` (``None`` when none
    was given)."""
    file: SourceReceipt | None = None
    saved_to: str | None = None
    cost_usd: float | None = None
    mechanism: str = MECHANISM


class ArtifactPublishResult(BaseModel):
    """``artifact_publish``: identity, URL and an independent readback.

    ``verified`` is true only when a separate read session found the same
    artifact and version, isolated the hosted file inside the service's fixed
    document shell, and the file's bytes equal the local file (itself
    unchanged during the call), and the audit of the publish call found
    nothing unexpected. The whole hosted response is not the local file: it is
    that file inside the shell (see ``parse.SHELL_START``). Otherwise ``problems`` says
    why. The artifact exists either way.
    """

    url: str
    artifact_id: str | None = None
    version: int
    version_id: str
    verified: bool
    retry_safe: bool = False
    """Always false: the artifact exists, and publishing again would create a
    second one. Fix `problems`, or delete the artifact by hand first."""
    private_confirmed: bool
    """The readback reported the artifact as private. A shared artifact, or a
    read that did not say, is ``False``."""
    problems: list[str] = Field(default_factory=list)
    denied_before_running: list[str] = Field(default_factory=list)
    """Calls the session attempted that the guard refused before they ran
    (``tool: reason``). Informational: a refused call had no effect."""
    source: SourceReceipt
    readback: Readback
    cost_usd: float | None = None
    mechanism: str = MECHANISM


class ArtifactUpdateResult(BaseModel):
    """``artifact_update``: a new version of an existing private artifact.

    ``before`` is what a separate read session saw *before* any write (owner,
    private, the live version the caller expected); ``readback`` is another
    separate read *after*. ``verified`` is true only when the artifact is the
    same one, still reported with the same visibility (private), now at the
    published version, its hosted file (pinned shell removed) equals the local
    file byte for byte, the update session viewed the expected version, and
    nothing unexpected was refused or escaped. This is **not** an atomic
    compare-and-swap: the native read and write are separate calls, so a
    concurrent edit between them is detected afterwards (``problems``), not
    prevented. Updating never retries; ``problems`` says what to look at.
    """

    url: str
    artifact_id: str | None = None
    previous_version_id: str
    version: int
    version_id: str
    verified: bool
    retry_safe: bool = False
    """Always false: the artifact changed. Read it (``artifact_read``) and
    decide from its live version; do not repeat the update blindly."""
    private_confirmed: bool
    problems: list[str] = Field(default_factory=list)
    denied_before_running: list[str] = Field(default_factory=list)
    source: SourceReceipt
    before: Readback
    readback: Readback
    cost_usd: float | None = None
    mechanism: str = MECHANISM
