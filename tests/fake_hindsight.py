"""A stand-in for Hindsight 0.10.1's single-bank MCP route, over real MCP.

It is a real streamable-HTTP MCP server (the ``mcp`` SDK) behind the same
``/mcp/<bank_id>/`` path the pinned server uses, so the real Claude CLI, the
real SDK and the real worker talk to it unchanged. It reproduces only what
was observed on 0.10.1 (see ``jake_tools.hindsight``): the bank comes from the
path, no tool takes a ``bank_id``, and the route also carries destructive
tools. It records every tool call with the bank it arrived on, and never
persists anything.

It proves mechanics only. It says nothing about recall quality.
"""

from __future__ import annotations

import contextlib
import contextvars
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from mcp.server.mcpserver import MCPServer

FACT = "the shared memory marker is turquoise-heron-42"

_bank: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "fake_hindsight_bank", default=None
)


@dataclass
class FakeHindsight:
    url: str
    calls: list[tuple[str | None, str, dict[str, Any]]] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)
    """The path of every HTTP request, as received."""

    def called(self, tool: str) -> list[tuple[str | None, dict[str, Any]]]:
        return [(bank, args) for bank, name, args in self.calls if name == tool]


def _build(state: FakeHindsight) -> MCPServer:
    mcp = MCPServer("hindsight-mcp-server")

    def note(tool: str, **args: Any) -> None:
        state.calls.append((_bank.get(), tool, args))

    @mcp.tool(description="Search memories for the bank. Read-only.")
    def recall(query: str, max_tokens: int = 4096) -> str:
        note("recall", query=query)
        return f"Memory found: {FACT}"

    @mcp.tool(description="Synthesise an answer from the bank. Read-only.")
    def reflect(query: str) -> str:
        note("reflect", query=query)
        return f"Reflection: {FACT}"

    @mcp.tool(description="Store one fact in the bank.")
    def retain(content: str, context: str = "general") -> str:
        note("retain", content=content)
        return "queued"

    @mcp.tool(description="Delete the whole bank. Destructive.")
    def delete_bank() -> str:
        note("delete_bank")
        return "deleted"

    @mcp.tool(description="Remove every memory in the bank. Destructive.")
    def clear_memories(type: str | None = None) -> str:
        note("clear_memories")
        return "cleared"

    return mcp


class _Route:
    """``/mcp/<bank>/...`` to the MCP app at ``/``, remembering the bank."""

    def __init__(self, app: Any, state: FakeHindsight) -> None:
        self.app, self.state = app, state

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope["path"]
        self.state.requests.append(path)
        parts = path.split("/")  # ['', 'mcp', bank, ...]
        if len(parts) < 3 or parts[1] != "mcp" or not parts[2]:
            await _respond(send, 404)
            return
        scope = dict(scope, path="/" + "/".join(parts[3:]).lstrip("/") or "/")
        scope["raw_path"] = scope["path"].encode()
        token = _bank.set(parts[2])
        try:
            await self.app(scope, receive, send)
        finally:
            _bank.reset(token)


async def _respond(send: Any, status: int) -> None:
    await send({"type": "http.response.start", "status": status, "headers": []})
    await send({"type": "http.response.body", "body": b""})


@contextlib.contextmanager
def fake_hindsight() -> Iterator[FakeHindsight]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    state = FakeHindsight(url=f"http://127.0.0.1:{port}")
    app = _Route(
        _build(state).streamable_http_app(
            streamable_http_path="/", json_response=True, stateless_http=True
        ),
        state,
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the fake Hindsight server did not start")
        time.sleep(0.02)
    try:
        yield state
    finally:
        server.should_exit = True
        thread.join(timeout=10)
