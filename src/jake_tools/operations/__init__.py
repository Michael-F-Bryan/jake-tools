"""The tool operations every agent-facing adapter exposes.

One module per operation. Each defines ``OPERATION``: its name, its
transport-neutral description, the packaged skill that explains it, and
``build(config)``, which returns the async handler for one resolved
:class:`~jake_tools.config.Config`. The handler's parameters are the
operation's arguments; it returns a Pydantic result or raises
:class:`~.errors.ToolError`.

The one agent-facing adapter, :mod:`jake_tools.mcp`, serves them over stdio
to every MCP client (Hermes included). It duplicates no operation's logic,
schema or description; it adds only a pointer to the skill (a ``skill://``
resource).

Keep handlers thin: validate arguments, raise ``ToolError``, and delegate to
a package module (``clockify_jira_sync``, ``claude_runs``). Blocking HTTP
clients run under ``anyio.to_thread.run_sync`` so a slow apply cannot stall
``claude_status``; so do the delegated-task operations, which take file
locks and read the process table. A thread call is not abandoned on
cancellation: a start or steer that began is finished and recorded.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel

from ..config import Config

Handler = Callable[..., Awaitable[BaseModel]]

SKILL_GUIDES: dict[str, str] = {
    "claude-delegation": (
        "The guide defines task scope, permissions, supervision, cancellation and "
        "recovery rules."
    ),
    "clockify-time": (
        "The guide defines workspace scope, pagination, write verification and "
        "uncertain-write recovery rules."
    ),
    "clockify-jira-sync": (
        "The guide defines scope, conflict handling, partial-write semantics and "
        "recovery rules."
    ),
    "artifact-publishing": (
        "The guide covers what a publish and a readback prove, the document shell "
        "around hosted content, what visibility the readback does and does not "
        "show, the undocumented mechanism, and how to handle each error."
    ),
    "newsletters": (
        "The guide defines the Microsoft identity and authority used, list coverage, "
        "the publication boundary, write verification, partial-outcome recovery and "
        "how to handle each error."
    ),
    "transcription": (
        "The guide explains the transcript schema, what speaker and overlap fields "
        "do and do not promise, what polishing sends to Claude, changes and leaves "
        "unverified, and how to handle each error."
    ),
}
"""What each packaged skill gives a caller, for the adapter's skill pointer."""


def _always(config: Config) -> bool:
    return True


@dataclass(frozen=True)
class Operation:
    """One agent-facing tool, independent of how it is served."""

    name: str
    skill: str
    description: str
    build: Callable[[Config], Handler]
    omit_none: bool = False
    """Omit null result fields and use compact JSON."""
    available: Callable[[Config], bool] = _always
    """Whether ``config`` exposes the operation at all (``transcribe`` needs
    an endpoint). An unavailable operation is not registered."""

    @property
    def skill_guide(self) -> str:
        return SKILL_GUIDES[self.skill]
