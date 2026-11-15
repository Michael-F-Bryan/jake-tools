"""Pre-execution authority for the artifact sessions.

A model can be steered by what it reads: the HTML it publishes, a hosted
page, a tool result. A prompt that says "only publish" and an audit afterwards
are therefore not authority: by the time an audit sees a ``share`` or
``delete`` call, it has run. These guards are checked *before* each tool call
through :attr:`jake_tools.claude.AgentSpec.tool_guard` (a ``PreToolUse`` hook,
which the CLI runs for calls an allow rule approves), independently of the
prompt. It judges the RAW model-supplied input: any key not in the permitted
set, including a ``__*`` key, is denied; private runtime metadata is never
broadly allowed. They permit the exact calls one operation needs and nothing else:

* publish: ``Read`` of the one file, and one ``Artifact`` ``publish`` of that
  file, with no ``url`` (so never an update of something else), ``edits`` or
  other inputs;
* read: one ``Artifact`` ``read`` of the one URL, and no ``Read``.

Every other tool, action, path, URL or second publish is denied. A guard also
records what it was asked, so the service can detect a call it never saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parse import artifact_short_id, parse_header

PUBLISH_KEYS = frozenset({"action", "file_path", "title", "description", "icon"})
"""The only inputs a new publish may carry. ``url`` (update), ``edits`` and
anything else are refused."""
MAX_FILE_READS = 3


@dataclass
class ArtifactGuard:
    """One session's authority; callable as a :data:`~jake_tools.claude.ToolGuard`."""

    file_path: Path | None = None
    """The one file, already resolved to its canonical absolute path."""
    read_url: str | None = None
    update_url: str | None = None
    """For an update session only: the one existing artifact it may read and
    then publish a new version of. Never set for a new publish."""
    expected_version_id: str | None = None
    """For an update session: the version the caller expects to be live."""
    observed_ok: bool = False
    """Whether this session's own ``read`` of the target was *observed* (its
    tool result parsed) as owned by you, private and at the expected version.
    An update ``publish`` is denied until it is true."""
    observed_problem: str | None = None
    consulted: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    denied: list[tuple[str, str]] = field(default_factory=list)
    """``(tool, reason)`` for every call refused before it ran."""
    publishes_allowed: int = 0
    reads_allowed: int = 0
    file_reads_allowed: int = 0

    @classmethod
    def for_publish(cls, file_path: Path) -> ArtifactGuard:
        return cls(file_path=file_path)

    @classmethod
    def for_update(
        cls, file_path: Path, url: str, expected_version_id: str
    ) -> ArtifactGuard:
        """A separate grant: the exact file onto the exact existing artifact. It
        can read that artifact (the tool refuses an update until the live
        version has been viewed) and publish once, but only AFTER that read has
        been observed to show an artifact you own, private, at the expected
        version (:meth:`observe_read`)."""
        return cls(
            file_path=file_path, update_url=url, expected_version_id=expected_version_id
        )

    def observe_read(
        self, tool_input: dict[str, Any], text: str, is_error: bool
    ) -> None:
        """Feed this session's own ``read`` result; called as results stream in.

        Anything missing, unparseable, shared, not yours, another artifact or a
        different version leaves ``observed_ok`` false, so a later publish is
        denied before it runs. If the result arrives after the publish call, the
        publish was already denied: the failure direction is closed."""
        if self.update_url is None or tool_input.get("action") != "read":
            return
        if not _same_artifact(tool_input.get("url"), self.update_url):
            return
        header = None if is_error else parse_header(text)
        if header is None:
            self.observed_ok = False
            self.observed_problem = (
                "the read of the target returned no recognisable header"
            )
        elif not header.owned_by_you or header.visibility != "private":
            self.observed_ok = False
            self.observed_problem = (
                f"the target is not a private artifact you own "
                f"(visibility {header.visibility!r}, owned by you {header.owned_by_you})"
            )
        elif header.version_id != self.expected_version_id:
            self.observed_ok = False
            self.observed_problem = (
                f"the target is at version {header.version_id!r}, not the expected "
                f"{self.expected_version_id!r}"
            )
        else:
            self.observed_ok = True
            self.observed_problem = None

    @classmethod
    def for_read(cls, url: str) -> ArtifactGuard:
        return cls(read_url=url)

    def __call__(self, tool: str, tool_input: dict[str, Any]) -> str | None:
        self.consulted.append((tool, dict(tool_input)))
        reason = self._decide(tool, tool_input)
        if reason is not None:
            self.denied.append((tool, reason))
        return reason

    def _decide(self, tool: str, tool_input: dict[str, Any]) -> str | None:
        if self.file_path is not None:
            return self._publish_session(tool, tool_input, self.file_path)
        if self.read_url is not None:
            return self._read_session(tool, tool_input, self.read_url)
        return "no authority was granted to this session"

    def _publish_session(
        self, tool: str, tool_input: dict[str, Any], path: Path
    ) -> str | None:
        if tool == "Read":
            if set(tool_input) != {"file_path"} or not _same_path(
                tool_input["file_path"], path
            ):
                return f"only {path} may be read in this session"
            if self.file_reads_allowed >= MAX_FILE_READS:
                return "the file has already been read"
            self.file_reads_allowed += 1
            return None
        if tool != "Artifact":
            return f"{tool} is not permitted in an artifact publish session"
        action = tool_input.get("action", "publish")
        if self.update_url is not None and action == "read":
            return self._update_read(tool_input, self.update_url)
        if action != "publish":
            what = "one update" if self.update_url else "one new publish"
            return f"Artifact action {action!r} is not permitted: only {what}"
        allowed_keys = PUBLISH_KEYS | ({"url"} if self.update_url else frozenset())
        extra = sorted(set(tool_input) - allowed_keys)
        if extra:
            return f"Artifact publish input {', '.join(extra)} is not permitted"
        if self.update_url is not None and not _same_artifact(
            tool_input.get("url"), self.update_url
        ):
            return f"only {self.update_url} may be updated"
        if not _same_path(tool_input.get("file_path"), path):
            return f"only {path} may be published"
        if self.publishes_allowed >= 1:
            return "only one publish is permitted; it is never retried"
        if self.update_url is not None and not self.observed_ok:
            return (
                "an update is only permitted after this session's read of the "
                "target was observed as yours, private and at the expected version"
                + (f" ({self.observed_problem})" if self.observed_problem else "")
            )
        self.publishes_allowed += 1
        return None

    def _update_read(self, tool_input: dict[str, Any], url: str) -> str | None:
        if set(tool_input) != {"action", "url"} or not _same_artifact(
            tool_input.get("url"), url
        ):
            return f"only Artifact action 'read' of {url} is permitted"
        if self.reads_allowed >= 1:
            return "the artifact has already been read in this session"
        self.reads_allowed += 1
        return None

    def _read_session(
        self, tool: str, tool_input: dict[str, Any], url: str
    ) -> str | None:
        if tool != "Artifact":
            return f"{tool} is not permitted in an artifact read session"
        if tool_input.get("action") != "read" or set(tool_input) != {"action", "url"}:
            return "only Artifact action 'read' of the one URL is permitted"
        if not _same_artifact(tool_input.get("url"), url):
            return f"only {url} may be read"
        if self.reads_allowed >= 1:
            return "the artifact has already been read in this session"
        self.reads_allowed += 1
        return None


def _same_path(value: Any, path: Path) -> bool:
    """Whether ``value`` is exactly the canonical path the session was given.

    No normalisation: the tool resolves symlinks itself, so ``/x/link/../c.html``
    is lexically ``/x/c.html`` but names another file once ``link`` is followed.
    The caller resolves ``path`` once, hands the model that exact string, and
    anything else is refused."""
    return isinstance(value, str) and value == str(path)


def _same_artifact(given: Any, url: str) -> bool:
    """Whether ``given`` is a link to the same artifact (same short id)."""
    wanted = artifact_short_id(url)
    return (
        isinstance(given, str)
        and wanted is not None
        and artifact_short_id(given) == wanted
    )
