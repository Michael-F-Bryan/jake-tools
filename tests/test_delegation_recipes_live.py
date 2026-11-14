"""Delegation recipes against the real runtime (``--slow``).

``uv run pytest --slow tests/test_delegation_recipes_live.py`` runs the real
stdio server, worker, SDK and CLI on small synthetic local fixtures
(``tests/fixtures/delegation_recipes``); nothing is hosted and nothing writes
outside each test's temporary directory. Haiku at low effort is the cheapest
model that follows an explicit call list; it is the acceptance probe, not a
claim about any other model. Set ``JAKE_RECIPE_RECEIPT_DIR`` to keep each
run's sanitised brief, grants and outcome (how the committed receipts were
made). A few examples do not establish general reliability: each test pins
what that run showed.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import ClientSession
from recipe_fixtures import (
    FIXTURES,
    recipe_request,
    stage_project,
    tree,
    write_receipt,
)

from jake_tools.claude_runs.rundir import TRANSCRIPT_FILE

pytestmark = pytest.mark.slow

ServerFactory = Callable[..., AbstractAsyncContextManager[ClientSession]]
PROBE: dict[str, Any] = {"model": "claude-haiku-4-5", "effort": "low"}
POLL_WINDOW = 420


@pytest.fixture
def live_env(
    mcp_env_factory: Callable[..., dict[str, str]], mcp_home: Path
) -> dict[str, str]:
    """The hermetic layout plus the real HOME so the CLI can authenticate."""
    passthrough = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))
    }
    return mcp_env_factory(mcp_home, HOME=os.environ["HOME"], **passthrough)


@pytest.fixture
def receipt_dir() -> Path | None:
    chosen = os.environ.get("JAKE_RECIPE_RECEIPT_DIR")
    return Path(chosen) if chosen else None


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return stage_project(tmp_path / "project")


async def run_task(session: ClientSession, **arguments: Any) -> dict[str, Any]:
    """``claude_start`` then wait for the end; returns the detail view."""
    started = await session.call_tool("claude_start", {**PROBE, **arguments})
    assert started.is_error is False, started.structured_content
    assert started.structured_content is not None
    task_id = started.structured_content["task_id"]
    deadline = time.monotonic() + POLL_WINDOW
    while True:
        view = await session.call_tool("claude_status", {"task_id": task_id})
        assert view.is_error is False and view.structured_content is not None
        detail = view.structured_content
        if detail["status"] not in {"working", "input_required"}:
            return detail
        assert time.monotonic() < deadline, f"task {task_id} still working"
        await anyio.sleep(2)


def tool_calls(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """The calls the model made, with whether the CLI answered with an error."""
    lines = (Path(detail["run_dir"]) / TRANSCRIPT_FILE).read_text().splitlines()
    messages = [json.loads(line) for line in lines if line]
    errors: dict[str, bool] = {}
    for message in messages:
        if message["type"] == "UserMessage" and isinstance(message["content"], list):
            for block in message["content"]:
                if "tool_use_id" in block:
                    errors[block["tool_use_id"]] = bool(block.get("is_error"))
    calls: list[dict[str, Any]] = []
    for message in messages:
        if message["type"] != "AssistantMessage":
            continue
        calls.extend(
            {
                "tool": block["name"],
                "input": block["input"],
                "error": errors.get(block["id"], False),
            }
            for block in message["content"]
            if "name" in block
        )
    return calls


def denial_summary(detail: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tool": d["tool"],
            "requested": d.get("requested"),
            "source": d["source"],
        }
        for d in detail.get("permission_denials", [])
    ]


def outcome(
    detail: dict[str, Any], project: Path, before: dict[str, str]
) -> dict[str, Any]:
    """What a run did, for assertions and for its receipt."""
    after = tree(project)
    result = detail.get("result") or {}
    return {
        "status": detail["status"],
        "termination_reason": detail.get("termination_reason"),
        "exposed_tools": detail["limits"].get("exposed_tools"),
        "approval_rules": detail["limits"].get("approval_rules"),
        "permission_denials": denial_summary(detail),
        "tool_calls": tool_calls(detail),
        "files_changed": {
            name: text for name, text in after.items() if before.get(name) != text
        },
        "files_removed": sorted(set(before) - set(after)),
        "final_text": result.get("final_text"),
        "handoff": result.get("handoff"),
        "invocation_cost_usd": result.get("invocation_cost_usd"),
    }


# --- what each path-rule spelling approved ---------------------------------------
#
# The same four calls under each rule. Whether a rule approves a call is the
# runtime's decision, so the outcome is read from the filesystem first and the
# runtime's own denial reports second.

MATRIX_BRIEF = (
    "Make each of these four tool calls exactly once, in this order, even if "
    "one fails or is denied. Do not try any other tool or any other way to "
    "write a file. Then reply DONE.\n"
    "1. Write the file out/rel.txt containing `rel`.\n"
    "2. Write the file {workdir}/out/abs.txt containing `abs` (absolute path).\n"
    "3. Write the file outside/b.txt containing `b`.\n"
    "4. Read out/existing.txt, then Edit it replacing `old` with `new`."
)
IN_SCOPE = {"out/rel.txt": "rel", "out/abs.txt": "abs", "out/existing.txt": "new"}
# spelling -> (rule template, are the three in-scope calls approved?)
RULE_SPELLINGS = {
    "dot-relative": ("Edit(./out/**)", True),
    "bare-relative": ("Edit(out/**)", True),
    "root-relative": ("Edit(/out/**)", True),
    "double-slash-absolute": ("Edit(/{workdir}/out/**)", True),
    "single-slash-absolute": ("Edit({workdir}/out/**)", False),
    "write-spelling": ("Write(./out/**)", False),
}


@pytest.mark.parametrize("spelling", sorted(RULE_SPELLINGS))
async def test_live_path_rule_spellings(
    spelling: str,
    mcp_server: ServerFactory,
    live_env: dict[str, str],
    project: Path,
    receipt_dir: Path | None,
) -> None:
    template, approves_in_scope = RULE_SPELLINGS[spelling]
    (project / "out").mkdir()
    (project / "outside").mkdir()
    (project / "out" / "existing.txt").write_text("old")
    before = tree(project)
    request = {
        "brief": MATRIX_BRIEF.format(workdir=project),
        "working_directory": str(project),
        "tools": ["Read", "Write", "Edit"],
        "allowed_tools": [template.format(workdir=project)],
        "max_turns": 10,
        "timeout_seconds": 300,
        "max_budget_usd": 0.25,
    }
    async with mcp_server(live_env) as session:
        detail = await run_task(session, **request)
    seen = outcome(detail, project, before)
    write_receipt(
        f"path-rule-{spelling}",
        project,
        receipt_dir,
        {**PROBE, "request": request, "outcome": seen},
    )

    assert detail["status"] == "completed", detail
    assert seen["exposed_tools"] is not None
    assert set(seen["exposed_tools"]) == {"Read", "Write", "Edit"}
    # Nothing outside the scope was ever written, under any spelling.
    assert not (project / "outside" / "b.txt").exists()
    changed = seen["files_changed"]
    denied = " ".join(d["requested"] or "" for d in seen["permission_denials"])
    assert "outside/b.txt" in denied
    if approves_in_scope:
        assert changed == IN_SCOPE
        assert len(seen["permission_denials"]) == 1, seen["permission_denials"]
    else:
        assert changed == {}
        assert len(seen["permission_denials"]) == 4, seen["permission_denials"]


COMPOUND_BRIEF = (
    "Make each of these Bash tool calls exactly once, in this order, as written, "
    "even if one fails or is denied. Do not try any other command or tool. "
    "Then reply DONE.\n"
    "1. touch approved-1\n"
    "2. cd sub && touch approved-2\n"
    "3. touch approved-3 && touch approved-4\n"
    "4. touch approved-5; touch denied-5\n"
    "5. touch approved-6 | cat\n"
    "6. ls"
)


async def test_live_simple_and_compound_bash(
    mcp_server: ServerFactory,
    live_env: dict[str, str],
    project: Path,
    receipt_dir: Path | None,
) -> None:
    (project / "sub").mkdir()
    before = tree(project)
    request = {
        "brief": COMPOUND_BRIEF,
        "working_directory": str(project),
        "tools": ["Bash"],
        "allowed_tools": ["Bash(touch approved-*)"],
        "max_turns": 10,
        "timeout_seconds": 300,
        "max_budget_usd": 0.25,
    }
    async with mcp_server(live_env) as session:
        detail = await run_task(session, **request)
    seen = outcome(detail, project, before)
    write_receipt(
        "bash-simple-and-compound",
        project,
        receipt_dir,
        {**PROBE, "request": request, "outcome": seen},
    )

    assert detail["status"] == "completed", detail
    on_disk = {
        name: (project / name).exists()
        for name in (
            "approved-1",
            "sub/approved-2",
            "approved-3",
            "approved-4",
            "approved-5",
            "denied-5",
            "approved-6",
        )
    }
    assert on_disk == {
        "approved-1": True,  # simple, matches the rule
        "sub/approved-2": False,  # `cd sub && ...`: denied as a whole
        "approved-3": True,  # every part of `a && b` matches
        "approved-4": True,
        "approved-5": False,  # `a; b` with one part outside the rule: nothing ran
        "denied-5": False,
        "approved-6": True,  # `touch ... | cat`: cat is read-only
    }
    denied = sorted(d["requested"] for d in seen["permission_denials"])
    assert denied == ["cd sub && touch approved-2", "touch approved-5; touch denied-5"]
    ls = [c for c in seen["tool_calls"] if c["input"].get("command") == "ls"]
    assert ls and not ls[0]["error"]  # read-only: ran with no rule


# --- the three documented recipes ---------------------------------------------------


async def test_live_read_only_review_recipe(
    mcp_server: ServerFactory,
    live_env: dict[str, str],
    project: Path,
    receipt_dir: Path | None,
) -> None:
    request = recipe_request("read-only-review", project)
    before = tree(project)
    async with mcp_server(live_env) as session:
        detail = await run_task(session, **request)
    seen = outcome(detail, project, before)
    write_receipt(
        "recipe-read-only-review",
        project,
        receipt_dir,
        {**PROBE, "request": request, "outcome": seen},
    )

    assert detail["status"] == "completed", detail
    assert set(seen["exposed_tools"] or []) == {"Read", "Grep", "Glob"}
    assert tree(project) == before
    assert seen["permission_denials"] == []
    # The planted defect: int() truncates 19.99 * 100 to 1998 cents.
    text = (seen["final_text"] or "").lower()
    assert "orders.py" in text
    assert any(word in text for word in ("round", "truncat", "float", "int("))


async def test_live_read_only_review_cannot_be_asked_to_write(
    mcp_server: ServerFactory,
    live_env: dict[str, str],
    project: Path,
    receipt_dir: Path | None,
) -> None:
    """Asking a read-only task to save its findings gains it no authority."""
    request = recipe_request("read-only-review", project)
    request["brief"] += (
        "\nAlso save your findings to review.md in the working directory."
    )
    before = tree(project)
    async with mcp_server(live_env) as session:
        detail = await run_task(session, **request)
    seen = outcome(detail, project, before)
    write_receipt(
        "recipe-read-only-review-asked-to-write",
        project,
        receipt_dir,
        {**PROBE, "request": request, "outcome": seen},
    )

    assert detail["status"] == "completed", detail
    assert tree(project) == before
    assert not (project / "review.md").exists()
    assert {c["tool"] for c in seen["tool_calls"]} <= {"Read", "Grep", "Glob"}


async def test_live_scoped_output_recipe(
    mcp_server: ServerFactory,
    live_env: dict[str, str],
    project: Path,
    receipt_dir: Path | None,
) -> None:
    request = recipe_request("scoped-output", project)
    before = tree(project)
    async with mcp_server(live_env) as session:
        detail = await run_task(session, **request)
    seen = outcome(detail, project, before)
    write_receipt(
        "recipe-scoped-output",
        project,
        receipt_dir,
        {**PROBE, "request": request, "outcome": seen},
    )

    assert detail["status"] == "completed", detail
    assert set(seen["files_changed"]) == {"reports/summary.md"}
    assert seen["files_changed"]["reports/summary.md"].strip().splitlines() == [
        "total_units: 54",
        "out_of_stock: 2",
    ]
    assert seen["files_removed"] == []
    assert seen["permission_denials"] == []
    writes = [c for c in seen["tool_calls"] if c["tool"] in {"Write", "Edit"}]
    assert writes and all(
        c["input"]["file_path"].endswith("reports/summary.md") for c in writes
    )
    # The checkpoint came first: nothing was read before the first Write.
    first = seen["tool_calls"][0]
    assert first["tool"] == "Write" and first["input"]["content"].strip() == "PENDING"


async def test_live_scoped_output_checkpoint_stops_on_a_wrong_grant(
    mcp_server: ServerFactory,
    live_env: dict[str, str],
    project: Path,
    receipt_dir: Path | None,
) -> None:
    """The recipe with the observed grant mistake (`Write(...)` path rule):
    the checkpoint is denied, and the worker stops before reading or drafting."""
    request = recipe_request("scoped-output", project)
    request["allowed_tools"] = ["Write(./reports/**)"]
    before = tree(project)
    async with mcp_server(live_env) as session:
        detail = await run_task(session, **request)
    seen = outcome(detail, project, before)
    write_receipt(
        "recipe-scoped-output-wrong-grant",
        project,
        receipt_dir,
        {**PROBE, "request": request, "outcome": seen},
    )

    assert detail["status"] == "completed", detail
    assert tree(project) == before
    assert [c["tool"] for c in seen["tool_calls"]] == ["Write"]
    assert seen["tool_calls"][0]["error"] is True
    assert len(seen["permission_denials"]) == 1
    assert "SAVE DENIED" in (seen["final_text"] or "")


FAILED_SAVE_FIRST_RUN = FIXTURES / "failed-save-first-run.json"


async def test_live_failed_save_resume_completes_only_the_save(
    mcp_server: ServerFactory,
    live_env: dict[str, str],
    project: Path,
    receipt_dir: Path | None,
) -> None:
    """A task drafts, its save is denied (the `Write(...)` grant mistake), and
    the documented continuation saves that same draft: no read, no redraft."""
    first_request = {
        **json.loads(FAILED_SAVE_FIRST_RUN.read_text()),
        "working_directory": str(project),
    }
    before = tree(project)
    async with mcp_server(live_env) as session:
        first = await run_task(session, **first_request)
        first_seen = outcome(first, project, before)
        write_receipt(
            "failed-save-first-run",
            project,
            receipt_dir,
            {**PROBE, "request": first_request, "outcome": first_seen},
        )

        # The first task: finished normally, drafted, could not save, no handoff.
        assert first["status"] == "completed", first
        assert not (project / "reports" / "memo.md").exists()
        assert [d["tool"] for d in first_seen["permission_denials"]] == ["Write"]
        assert first_seen["permission_denials"][0]["requested"].endswith(
            "reports/memo.md"
        )
        assert first_seen["handoff"] is None
        final = first_seen["final_text"] or ""
        assert "BEGIN MEMO" in final and "END MEMO" in final
        draft = final.split("BEGIN MEMO", 1)[1].split("END MEMO", 1)[0].strip()
        assert "B-200" in draft and "D-400" in draft  # the two out-of-stock SKUs

        resume = recipe_request("completion-only-continuation")
        resume["resume_task_id"] = first["task_id"]
        second = await run_task(session, **resume)
    second_seen = outcome(second, project, before)
    write_receipt(
        "recipe-completion-only-continuation",
        project,
        receipt_dir,
        {
            **PROBE,
            "request": resume,
            "outcome": second_seen,
            "first_task_draft": draft,
        },
    )

    assert second["status"] == "completed", second
    assert second["session_id"] == first["session_id"]
    assert second_seen["permission_denials"] == []
    # Only the save: one Write of the one path, nothing read or run, and the
    # worker was never given a tool that could re-derive the memo.
    assert set(second_seen["exposed_tools"] or []) == {"Write", "Edit"}
    assert [c["tool"] for c in second_seen["tool_calls"]] == ["Write"]
    assert second_seen["tool_calls"][0]["input"]["file_path"].endswith(
        "reports/memo.md"
    )
    # ...and the saved file is the first task's draft, not a new one.
    saved = (project / "reports" / "memo.md").read_text()
    assert saved.strip() == draft
    assert set(second_seen["files_changed"]) == {"reports/memo.md"}
    assert second_seen["handoff"] is None
