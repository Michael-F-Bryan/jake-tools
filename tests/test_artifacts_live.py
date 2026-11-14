"""The artifact guard and operations against the real SDK and CLI.

``slow`` and ``live``: ``uv run pytest --slow -m live tests/test_artifacts_live.py``.

* The guard tests demonstrate on the real runtime that a ``PreToolUse`` guard
  is consulted for every tool call, *including calls an allow rule approves*,
  and that a deny cannot be overridden by that rule. They use only harmless
  calls: reading a canary file, and an Artifact ``list``/``read`` that is
  denied before it runs (read-only even if the guard failed). No test here
  calls ``publish``, ``edit``, ``share`` or ``delete`` to prove a denial.
* The readback test needs an artifact that already exists and is read-only:
  set ``JAKE_TOOLS_LIVE_ARTIFACT_URL`` and ``JAKE_TOOLS_LIVE_ARTIFACT_FILE``
  (and optionally ``JAKE_TOOLS_LIVE_ARTIFACT_VERSION``).
* The publish test CREATES A REAL ARTIFACT every run. It runs only when
  ``JAKE_TOOLS_LIVE_ARTIFACT_PUBLISH`` is set to ``create-a-new-artifact`` and
  must not be set without explicit authority for that synthetic file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from jake_tools.artifacts import publish, read
from jake_tools.artifacts.guard import ArtifactGuard
from jake_tools.artifacts.service import ARTIFACT_ENV
from jake_tools.claude import AgentSpec, ClaudeAgent, describe

pytestmark = [pytest.mark.slow, pytest.mark.live]

CANARY = "CANARY-7f3a91c2-must-not-be-read"
ALLOWED = "ALLOWED-0b84d6e1-may-be-read"


def _agent_and_trace() -> tuple[ClaudeAgent, list[Any]]:
    views: list[Any] = []
    agent = ClaudeAgent().with_message_observer(lambda m: views.append(describe(m)))
    return agent, views


async def test_guard_runs_for_calls_an_allow_rule_would_approve(
    tmp_path: Path,
) -> None:
    (tmp_path / "ok.txt").write_text(ALLOWED)
    (tmp_path / "secret.txt").write_text(CANARY)
    consulted: list[tuple[str, dict[str, Any]]] = []

    def guard(tool: str, tool_input: dict[str, Any]) -> str | None:
        consulted.append((tool, tool_input))
        path = str(tool_input.get("file_path", ""))
        return None if path.endswith("ok.txt") else "only ok.txt may be read"

    agent, views = _agent_and_trace()
    await agent.run(
        "Use the Read tool on each of these two files, one call each, and tell me "
        f"what each says: {tmp_path / 'ok.txt'} and {tmp_path / 'secret.txt'}",
        AgentSpec(
            model="haiku",
            tools=("Read",),
            allowed_tools=("Read",),  # approves every Read; the guard must still run
            tool_guard=guard,
            cwd=tmp_path,
            permission_mode="dontAsk",
            max_turns=6,
            max_budget_usd=0.25,
        ),
    )
    uses = [use for v in views for use in v.tool_uses]
    results = [r for v in views for r in v.tool_results]
    assert len(consulted) == len(uses) >= 2, "the guard missed a call"
    assert any(CANARY in str(t[1]) or "secret.txt" in str(t[1]) for t in consulted)
    assert not any(CANARY in r.text for r in results), "a denied Read still ran"
    assert any(ALLOWED in r.text for r in results), "the permitted Read did not run"


def _validation_rejected(views: list[Any]) -> int:
    return sum(
        1
        for v in views
        for r in v.tool_results
        if r.is_error and r.text.startswith("<tool_use_error>")
    )


async def test_artifact_guard_denies_other_actions_and_urls_before_they_run() -> None:
    authorised = "https://claude.ai/artifact/Hs4Jp8Ek6Ct2Ua3NxLyMwb"
    other = "https://claude.ai/artifact/Qm3Kz7Tn5Rv9Wx2YbDfGhj"
    guard = ArtifactGuard.for_read(authorised)
    agent, views = _agent_and_trace()
    await agent.run(
        'First call the Artifact tool with action "list". Then call it with action '
        f'"read" and url "{other}". Report what each call returned.',
        AgentSpec(
            model="haiku",
            tools=("Artifact",),
            allowed_tools=("Artifact",),
            tool_guard=guard,
            env=dict(ARTIFACT_ENV),
            cwd=Path(os.getcwd()),
            permission_mode="dontAsk",
            max_turns=6,
            max_budget_usd=0.25,
        ),
    )
    uses = [use for v in views for use in v.tool_uses if use.name == "Artifact"]
    results = [r.text for v in views for r in v.tool_results]
    init_tools = next((v.tools for v in views if v.kind == "init"), None)
    assert init_tools is not None and "Artifact" in init_tools
    assert len(uses) >= 1
    # Every call is either checked by the guard or refused by the tool's own
    # input validation (which runs first and never executes the call).
    assert len(guard.consulted) + _validation_rejected(views) >= len(uses)
    assert guard.denied, "nothing was denied"
    assert guard.reads_allowed == 0
    # A listing starts "Showing N of M"; a hosted read starts "[Artifact ".
    assert not any("Showing" in t and "of" in t for t in results)
    assert not any(t.startswith("[Artifact ") for t in results)


@pytest.mark.skipif(
    not os.environ.get("JAKE_TOOLS_LIVE_ARTIFACT_URL"),
    reason="JAKE_TOOLS_LIVE_ARTIFACT_URL (an existing artifact to read) is not set",
)
async def test_read_back_an_existing_artifact_read_only() -> None:
    url = os.environ["JAKE_TOOLS_LIVE_ARTIFACT_URL"]
    file = os.environ.get("JAKE_TOOLS_LIVE_ARTIFACT_FILE")
    version = os.environ.get("JAKE_TOOLS_LIVE_ARTIFACT_VERSION")
    result = await read(url, file_path=file, expected_version_id=version)
    assert result.private is True
    assert result.shell_removed
    if file:
        assert result.matches_file is True
    if version:
        assert result.version_matches is True


@pytest.mark.skipif(
    os.environ.get("JAKE_TOOLS_LIVE_ARTIFACT_PUBLISH") != "create-a-new-artifact",
    reason="creates a real artifact; needs JAKE_TOOLS_LIVE_ARTIFACT_PUBLISH="
    "create-a-new-artifact and explicit authority for the synthetic file",
)
async def test_publish_a_synthetic_file_and_verify_it(tmp_path: Path) -> None:
    marker = os.environ.get("JAKE_TOOLS_LIVE_ARTIFACT_MARKER", "jake-tools-live")
    path = tmp_path / "live-probe.html"
    path.write_text(
        "<!doctype html>\n<html><head><meta charset=utf-8>"
        f"<title>jake-tools live probe</title></head>\n<body><p>{marker}</p></body>"
        "</html>\n",
        encoding="utf-8",
    )
    result = await publish(str(path))
    print(result.model_dump_json(indent=2))
    assert result.verified, result.problems
    assert result.private_confirmed
