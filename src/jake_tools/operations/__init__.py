"""The tool operations every agent-facing adapter exposes.

One module per operation. Each defines ``OPERATION``: its name, the
capability that gates it, its transport-neutral description, the packaged
skill that explains it, and ``build(config)``, which returns the async
handler for one resolved :class:`~jake_tools.config.Config`. The handler's
parameters are the operation's arguments; it returns a Pydantic result or
raises :class:`~.errors.ToolError`.

Adapters only register: :mod:`jake_tools.mcp` serves them over stdio and
:mod:`jake_tools.hermes` installs them as native Hermes tools. Neither
duplicates an operation's logic, schema or description; each adds only its
own pointer to the skill (an MCP resource, or a Hermes plugin skill).

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
from typing import Literal

from pydantic import BaseModel

from ..config import Config

Capability = Literal["claude", "clockify", "transcription"]
"""What an agent can be granted. Each operation belongs to exactly one."""

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
    "transcription": (
        "The guide explains the transcript schema, what speaker and overlap fields "
        "do and do not promise, and how to handle each error."
    ),
}
"""What each packaged skill gives a caller, for the adapters' skill pointers."""


def _always(config: Config) -> bool:
    return True


@dataclass(frozen=True)
class Operation:
    """One agent-facing tool, independent of how it is served."""

    name: str
    capability: Capability
    skill: str
    description: str
    build: Callable[[Config], Handler]
    available: Callable[[Config], bool] = _always
    """Whether ``config`` exposes the operation at all (``transcribe`` needs
    an endpoint). An unavailable operation is not registered."""

    @property
    def skill_guide(self) -> str:
        return SKILL_GUIDES[self.skill]
