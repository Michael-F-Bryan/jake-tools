"""``ping``: the smoke-test tool. Proves the transport and result plumbing."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from ...config import Config
from ..server import structured_tool


class PingResult(BaseModel):
    ok: bool = True
    version: str
    pid: int
    server_time: datetime


def register(app: FastMCP, config: Config) -> None:
    @structured_tool(
        app,
        name="ping",
        description=(
            "Check that the jake-tools server is up. Returns the package "
            "version, the server's PID and the current UTC time."
        ),
    )
    async def ping() -> PingResult:
        return PingResult(
            version=_package_version(), pid=os.getpid(), server_time=datetime.now(UTC)
        )


def _package_version() -> str:
    try:
        return version("jake-tools")
    except PackageNotFoundError:  # pragma: no cover - only outside an install
        return "unknown"
