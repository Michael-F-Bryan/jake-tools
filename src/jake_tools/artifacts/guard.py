"""Pre-execution authority for the artifact sessions.

A model can be steered by what it reads: the HTML it publishes, a hosted
page, a tool result. A prompt that says "only publish" and an audit afterwards
are therefore not authority: by the time an audit sees a ``share`` or
``delete`` call, it has run. These guards are checked *before* each tool call
through :attr:`jake_tools.claude.AgentSpec.tool_guard` (a ``PreToolUse`` hook
and a ``can_use_tool`` callback, two independent points in the same session,
with no allow rule), independently of the prompt. They permit the exact calls one operation needs and nothing else:

* publish: ``Read`` of the one file, and one ``Artifact`` ``publish`` of that
  file, with no ``url`` (so never an update of something else), ``edits`` or
  other inputs;
* read: one ``Artifact`` ``read`` of the one URL, and no ``Read``.

Every other tool, action, path, URL or second publish is denied. A guard also
records what it was asked, so the service can detect a call it never saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parse import artifact_short_id

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
    consulted: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    denied: list[tuple[str, str]] = field(default_factory=list)
    """``(tool, reason)`` for every call refused before it ran."""
    decisions: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    """``tool_use_id`` -> (fingerprint of the exact request, answer). Two
    enforcement layers (the ``PreToolUse`` hook and ``can_use_tool``) ask about
    the same call and must get the same answer without it being counted twice.
    The answer is reused only for the *identical* request: another hook can
    rewrite a call's input between the layers, and an approval of the safe form
    must never carry over to the changed one."""
    publishes_allowed: int = 0
    reads_allowed: int = 0
    file_reads_allowed: int = 0

    @classmethod
    def for_publish(cls, file_path: Path) -> ArtifactGuard:
        return cls(file_path=file_path)

    @classmethod
    def for_read(cls, url: str) -> ArtifactGuard:
        return cls(read_url=url)

    def __call__(
        self, tool: str, tool_input: dict[str, Any], tool_use_id: str | None = None
    ) -> str | None:
        fingerprint = _fingerprint(tool, tool_input)
        if tool_use_id is not None and tool_use_id in self.decisions:
            seen, answer = self.decisions[tool_use_id]
            if seen == fingerprint:
                return answer
            reason = "the call changed between enforcement layers; denied"
            self.denied.append((tool, reason))
            self.decisions[tool_use_id] = (seen, reason)  # and stays denied
            return reason
        self.consulted.append((tool, dict(tool_input)))
        reason = self._decide(tool, tool_input)
        if reason is not None:
            self.denied.append((tool, reason))
        if tool_use_id is not None:
            self.decisions[tool_use_id] = (fingerprint, reason)
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
        if action != "publish":
            return f"Artifact action {action!r} is not permitted: only one new publish"
        extra = sorted(set(tool_input) - PUBLISH_KEYS)
        if extra:
            return f"Artifact publish input {', '.join(extra)} is not permitted"
        if not _same_path(tool_input.get("file_path"), path):
            return f"only {path} may be published"
        if self.publishes_allowed >= 1:
            return "only one publish is permitted; it is never retried"
        self.publishes_allowed += 1
        return None

    def _read_session(
        self, tool: str, tool_input: dict[str, Any], url: str
    ) -> str | None:
        if tool != "Artifact":
            return f"{tool} is not permitted in an artifact read session"
        if tool_input.get("action") != "read" or set(tool_input) != {"action", "url"}:
            return "only Artifact action 'read' of the one URL is permitted"
        given = tool_input.get("url")
        wanted = artifact_short_id(url)
        if (
            not isinstance(given, str)
            or wanted is None
            or artifact_short_id(given) != wanted
        ):
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


def _fingerprint(tool: str, tool_input: dict[str, Any]) -> str:
    """The tool and its exact input, order-independent."""
    return json.dumps([tool, tool_input], sort_keys=True, default=str)
