"""One module per tool. Each exposes ``register(app, config)``.

Keep handlers thin: validate arguments, raise :class:`~..errors.ToolError`,
and delegate to a package module (``clockify_jira_sync``, ``claude_runs``).
Blocking HTTP clients run under ``anyio.to_thread.run_sync`` so a slow apply
cannot stall ``claude_status``.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ...config import Config
from . import ping


def register_all(app: FastMCP, config: Config) -> None:
    ping.register(app, config)
