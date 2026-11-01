"""Register the shared operations, plus the MCP-only ``ping``, on the server.

The operations live in :mod:`jake_tools.operations`; this module adds only
the MCP pointer to each operation's skill, a ``skill://`` resource.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ...config import Config
from ...operations import Operation
from ...operations.catalogue import OPERATIONS
from ..server import structured_tool
from . import ping


def register_all(app: MCPServer, config: Config) -> None:
    ping.register(app, config)
    for operation in OPERATIONS:
        if operation.available(config):
            structured_tool(
                app, name=operation.name, description=mcp_description(operation)
            )(operation.build(config))


def mcp_description(operation: Operation) -> str:
    return (
        f"{operation.description.rstrip()}\n\nBefore using this tool, read "
        f"`skill://{operation.skill}` via this server's `read_resource` capability, "
        "unless its contents are already loaded in the current context. "
        f"{operation.skill_guide} Reuse the loaded guide for subsequent calls."
    )
