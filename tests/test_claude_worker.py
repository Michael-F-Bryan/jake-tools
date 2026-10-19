"""The worker's driver against a fake SDK client: outcome versus cleanup timing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import anyio
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage

from jake_tools.claude import ClaudeAgent
from jake_tools.claude_runs.events import EventLog
from jake_tools.claude_runs.inbox import Inbox
from jake_tools.claude_runs.models import RunSpec
from jake_tools.claude_runs.rundir import create_run_dir, new_task_id
from jake_tools.claude_runs.worker import Driver, Progress


class SlowToCloseClient:
    """Answers at once, then takes ``close_seconds`` to shut down (shielded,
    like the SDK's own disconnect)."""

    def __init__(self, options: ClaudeAgentOptions, close_seconds: float) -> None:
        self.close_seconds = close_seconds

    async def connect(self, prompt: object = None) -> None:
        return None

    async def query(self, prompt: object, session_id: str = "default") -> None:
        async for _ in prompt:  # type: ignore[union-attr]
            pass

    async def receive_messages(self) -> AsyncIterator[Message]:
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            total_cost_usd=0.01,
            usage={"input_tokens": 1, "output_tokens": 1},
            result="all done",
        )

    async def disconnect(self) -> None:
        with anyio.CancelScope(shield=True):
            await anyio.sleep(self.close_seconds)


def _driver(tmp_path: Path, *, timeout: float, close_seconds: float) -> Driver:
    spec = RunSpec(
        task_id=new_task_id(),
        created_at=datetime.now(UTC),
        working_directory=tmp_path,
        model="m",
        max_turns=5,
        timeout_seconds=timeout,
        max_budget_usd=1,
        schema_version=2,
    )
    run_dir = create_run_dir(tmp_path / "runs", spec, "brief")
    agent = ClaudeAgent(
        connect=lambda options: SlowToCloseClient(options, close_seconds)
    )
    events = EventLog(run_dir)
    return Driver(
        agent, spec, "brief", Inbox(run_dir, events), events, Progress(run_dir)
    )


async def test_a_deadline_during_shutdown_does_not_undo_a_finished_task(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path, timeout=0.5, close_seconds=1.5)

    outcome = await driver.run()

    assert outcome.status == "completed"
    assert outcome.final_text == "all done"
    assert driver.resolved == outcome


async def test_a_cancel_during_shutdown_keeps_the_resolved_outcome(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path, timeout=30, close_seconds=1.0)

    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as tg:
            tg.start_soon(driver.run)
            while driver.resolved is None:
                await anyio.sleep(0.01)
            scope.cancel()  # a SIGTERM while the CLI is being closed

    assert driver.resolved is not None
    assert driver.resolved.status == "completed"
