"""Shared fixtures for the real MCP server over stdio."""

from __future__ import annotations

# --- MCP server harness --------------------------------------------------------
#
# The stdio tests drive the real server: ``sys.executable -m jake_tools.mcp``
# under a hermetic environment (HOME and the XDG directories under a temp
# dir, PATH, and nothing else), through the ``mcp`` client library. Nothing
# in the server is monkeypatched; credentials are absent unless a test adds
# them to the environment it passes.
import os
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_SERVE_ARGS: tuple[str, ...] = ("-m", "jake_tools.mcp", "serve")


def isolated_mcp_env(home: Path, **extra: str) -> dict[str, str]:
    """A hermetic server environment rooted at ``home``.

    Every XDG variable points under ``home`` so the server can never read or
    write the developer's real config or state. No secret is present unless
    the caller passes one in ``extra``.
    """
    home.mkdir(parents=True, exist_ok=True)
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_DIRS": str(home / "etc-xdg"),
        "PYTHONUNBUFFERED": "1",
    }
    env.update(extra)
    return env


@asynccontextmanager
async def start_mcp_server(
    env: Mapping[str, str], *, args: tuple[str, ...] = MCP_SERVE_ARGS
) -> AsyncIterator[ClientSession]:
    """Spawn the real server over stdio and yield an initialised client session."""
    params = StdioServerParameters(
        command=sys.executable, args=list(args), env=dict(env)
    )
    async with (
        stdio_client(params, errlog=sys.stderr) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        yield session


@pytest.fixture
def mcp_home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def mcp_env(mcp_home: Path) -> dict[str, str]:
    return isolated_mcp_env(mcp_home)


@pytest.fixture
def mcp_env_factory() -> Callable[..., dict[str, str]]:
    return isolated_mcp_env


@pytest.fixture
def mcp_server() -> Callable[..., AbstractAsyncContextManager[ClientSession]]:
    """``async with mcp_server(env) as session:`` against the real server."""
    return start_mcp_server


# --- a fake ``claude`` CLI on PATH ---------------------------------------------
#
# ``tests/fake_claude.py`` speaks the CLI's stream-json protocol. Putting it
# first on PATH as ``claude`` lets the real server spawn the real worker,
# which drives the real SDK, which runs this instead of the real CLI (the
# SDK prefers ``shutil.which("claude")``). Deterministic mechanics only.

FAKE_CLAUDE = Path(__file__).resolve().parent / "fake_claude.py"


def install_fake_claude(root: Path) -> tuple[Path, Path]:
    """Write ``root/bin/claude``; returns ``(bin_dir, state_dir)``."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    state = root / "fake-claude-state"
    state.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "claude"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_CLAUDE}" "$@"\n')
    script.chmod(0o755)
    return bin_dir, state


@pytest.fixture
def fake_claude_state(tmp_path: Path) -> Path:
    return tmp_path / "fake" / "fake-claude-state"


@pytest.fixture
def fake_env(mcp_home: Path, tmp_path: Path) -> dict[str, str]:
    """The hermetic server environment with the fake CLI first on PATH."""
    bin_dir, state = install_fake_claude(tmp_path / "fake")
    env = isolated_mcp_env(mcp_home, FAKE_CLAUDE_STATE=str(state))
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    return env
