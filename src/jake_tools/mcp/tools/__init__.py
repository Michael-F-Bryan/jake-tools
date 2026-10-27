"""One module per tool. Each exposes ``register(app, config)``.

Keep handlers thin: validate arguments, raise :class:`~..errors.ToolError`,
and delegate to a package module (``clockify_jira_sync``, ``claude_runs``).
Blocking HTTP clients run under ``anyio.to_thread.run_sync`` so a slow apply
cannot stall ``claude_status``; so do the delegated-task operations, which
take file locks and read the process table. A thread call is not abandoned
on cancellation: a start or steer that began is finished and recorded.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ...config import Config
from . import (
    claude_cancel,
    claude_list,
    claude_poll,
    claude_start,
    claude_status,
    claude_steer,
    claude_transcript,
    claude_update,
    clockify_create_time_entry,
    clockify_jira_sync,
    clockify_time_entries,
    clockify_workspace_catalogue,
    ping,
    transcribe,
)


def register_all(app: MCPServer, config: Config) -> None:
    ping.register(app, config)
    clockify_jira_sync.register(app, config)
    clockify_time_entries.register(app, config)
    clockify_workspace_catalogue.register(app, config)
    clockify_create_time_entry.register(app, config)
    claude_start.register(app, config)
    claude_status.register(app, config)
    claude_poll.register(app, config)
    claude_list.register(app, config)
    claude_transcript.register(app, config)
    claude_steer.register(app, config)
    claude_update.register(app, config)
    claude_cancel.register(app, config)
    # Registers nothing unless transcription.api_base_url is configured.
    transcribe.register(app, config)
