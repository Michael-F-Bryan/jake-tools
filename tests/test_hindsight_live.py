"""A real Claude worker using the parent's shared Hindsight memory.

``slow`` (``uv run pytest --slow -k hindsight_live``). The real stdio server,
worker, SDK and ``claude`` CLI with the cheapest model; only Hindsight is
replaced by ``fake_hindsight.py``, a real MCP server on the single-bank route
shape observed on 0.10.1 that records the bank and tool of each call. No real
memory bank is touched, and nothing is written anywhere.

What this establishes, against the real runtime:

- a task that asks for no built-in tools still has its memory tools exposed
  and approved (so the explicit grant is independent of ``tools``);
- the call arrives on the configured bank's route;
- a destructive tool on the same route is exposed but never run;
- ``readwrite`` and ``read`` differ in whether ``retain`` runs.

It cannot say whether the model uses memory *well*, nor anything about the
real server's recall quality.

Environment: as ``test_claude_tasks_live``: the developer's real ``HOME``
holds the CLI's credentials; no user MCP server or settings file is read.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import pytest
from fake_hindsight import FACT, fake_hindsight
from mcp import ClientSession
from test_claude_tasks_live import (  # noqa: F401 - live_env is a fixture
    CHEAP,
    Evidence,
    call,
    ended,
    live_env,
    poll_until,
)

pytestmark = pytest.mark.slow

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]

BANK = "bank-under-test"
MARKER = FACT.rsplit(" ", 1)[1]


async def _run(
    mcp_server: ServerFactory,
    env: dict[str, str],
    brief: str,
    workdir: Path,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = Evidence(name)
    async with mcp_server(env) as session:
        started = await call(
            session,
            evidence,
            "claude_start",
            brief=brief,
            working_directory=str(workdir),
            tools=[],
            **CHEAP,
        )
        final = await poll_until(session, evidence, started["task_id"], ended)
        status = await call(
            session, evidence, "claude_status", task_id=started["task_id"]
        )
    return final, status


async def test_live_worker_recalls_from_the_configured_bank_with_no_builtin_tools(
    mcp_server: ServerFactory,
    live_env: dict[str, str],  # noqa: F811
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    with fake_hindsight() as memory:
        env = dict(
            live_env,
            JAKE_TOOLS_HINDSIGHT_API_URL=memory.url,
            JAKE_TOOLS_HINDSIGHT_BANK_ID=BANK,
        )
        final, status = await _run(
            mcp_server,
            env,
            "Call the mcp__hindsight__recall tool with the query 'shared memory "
            "marker'. Then call mcp__hindsight__delete_bank once. Use no other "
            "tool. Finish with a final line of exactly MARKER=<the marker value "
            "recall returned, verbatim>.",
            workdir,
            "hindsight-recall",
        )

    assert final["status"] == "completed", final
    assert MARKER in status["result"]["final_text"]
    exposed = status["limits"]["exposed_tools"]
    assert "mcp__hindsight__recall" in exposed
    assert not [name for name in exposed if not name.startswith("mcp__")]
    # The call arrived on the configured bank's single-bank route.
    assert memory.called("recall"), memory.calls
    assert {bank for bank, _ in memory.called("recall")} == {BANK}
    assert all(path.startswith(f"/mcp/{BANK}/") for path in memory.requests)
    # Exposed on the route, never approved: not one destructive call ran.
    assert memory.called("delete_bank") == []
    assert memory.called("clear_memories") == []


@pytest.mark.parametrize(("access", "retains"), [("readwrite", True), ("read", False)])
async def test_live_retain_runs_only_with_readwrite_access(
    mcp_server: ServerFactory,
    live_env: dict[str, str],  # noqa: F811
    tmp_path: Path,
    access: str,
    retains: bool,
) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    with fake_hindsight() as memory:
        env = dict(
            live_env,
            JAKE_TOOLS_HINDSIGHT_API_URL=memory.url,
            JAKE_TOOLS_HINDSIGHT_BANK_ID=BANK,
            JAKE_TOOLS_HINDSIGHT_ACCESS=access,
        )
        final, _ = await _run(
            mcp_server,
            env,
            "Call mcp__hindsight__retain once with content 'unit test fact' and "
            "then reply DONE. If the call is refused, say REFUSED. Use no other tool.",
            workdir,
            f"hindsight-retain-{access}",
        )

    assert final["status"] == "completed", final
    assert bool(memory.called("retain")) is retains, memory.calls
    assert {bank for bank, _ in memory.called("retain")} <= {BANK}
