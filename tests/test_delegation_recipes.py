"""Hermetic checks on the delegation recipes (no model call).

They prove the documented recipes are well-formed ``claude_start`` requests,
that the saved receipts are receipts of *these* recipes, and that MCP and
native consumers read one file. They cannot show what the runtime approves or
that a model follows a brief: that is ``test_delegation_recipes_live.py``
(``--slow``), whose sanitised outcomes the receipts keep.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from conftest import isolated_mcp_env, start_mcp_server
from mcp.types import TextResourceContents
from recipe_fixtures import (
    FIXTURES,
    RECEIPTS,
    RULE_SPELLINGS,
    WORKDIR_PLACEHOLDER,
    load_recipes,
    recipe_request,
    recipes_path,
)

from jake_tools.claude_runs.control import StartRequest
from jake_tools.hermes.plugin import SKILLS_DIR

RECIPE_NAMES = {"read-only-review", "scoped-output", "completion-only-continuation"}
WRITE_TOOLS = {"Write", "Edit", "Bash", "NotebookEdit"}


def _start_request(arguments: dict[str, Any], workdir: Path) -> StartRequest:
    if "working_directory" in arguments:
        arguments = {**arguments, "working_directory": str(workdir)}
    return StartRequest.model_validate(
        {
            **arguments,
            "tools": tuple(arguments.get("tools", ())),
            "allowed_tools": tuple(arguments["allowed_tools"]),
        }
    )


def test_every_documented_recipe_is_a_valid_start_request_with_explicit_grants(
    tmp_path: Path,
) -> None:
    recipes = load_recipes()
    assert set(recipes) == RECIPE_NAMES
    for name, arguments in recipes.items():
        request = _start_request(arguments, tmp_path)
        assert request.allowed_tools, f"{name}: grants must be written out"
        assert request.tools, f"{name}: no tools is not a recipe"
        for limit in ("max_turns", "timeout_seconds", "max_budget_usd"):
            assert arguments.get(limit), f"{name}: no {limit} stopping condition"
        assert "stop" in arguments["brief"], f"{name}: no stop-on-denial rule"
        # Grants are exactly as wide as the rules written, never a bare write tool.
        for rule in arguments["allowed_tools"]:
            if rule.split("(")[0] in WRITE_TOOLS:
                assert "(" in rule, f"{name}: {rule} approves every call"
            assert not rule.startswith("Write("), f"{name}: {rule} approves nothing"


def test_read_only_recipe_has_no_write_authority_and_the_others_have_one_directory() -> (
    None
):
    recipes = load_recipes()
    review = recipes["read-only-review"]
    assert not WRITE_TOOLS & set(review["tools"])
    assert not WRITE_TOOLS & {rule.split("(")[0] for rule in review["allowed_tools"]}
    assert review["working_directory"] == WORKDIR_PLACEHOLDER

    assert recipes["scoped-output"]["allowed_tools"] == ["Edit(./reports/**)"]
    resume = recipes["completion-only-continuation"]
    assert resume["allowed_tools"] == ["Edit(./reports/**)"]
    assert "working_directory" not in resume  # a resume reuses its parent's
    assert "resume_task_id" in resume
    # The save step cannot re-derive anything: no way to read or run.
    assert set(resume["tools"]) == {"Write", "Edit"}


def _comparable(request: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in request.items() if k != "resume_task_id"}


@pytest.mark.parametrize("name", sorted(RECIPE_NAMES))
def test_receipt_matches_the_documented_recipe_exactly(name: str) -> None:
    receipt = json.loads((RECEIPTS / f"recipe-{name}.json").read_text())
    assert _comparable(receipt["request"]) == _comparable(recipe_request(name))
    assert receipt["cli_version"].startswith("2.")
    assert receipt["model"] and receipt["outcome"]["status"] == "completed"


def test_first_run_of_the_failed_save_fixture_matches_its_receipt() -> None:
    fixture = json.loads((FIXTURES / "failed-save-first-run.json").read_text())
    receipt = json.loads((RECEIPTS / "failed-save-first-run.json").read_text())
    assert receipt["request"] == {**fixture, "working_directory": WORKDIR_PLACEHOLDER}
    # The mistaken grant the recipe corrects, and the denial it produced.
    assert fixture["allowed_tools"] == ["Write(./reports/**)"]
    outcome = receipt["outcome"]
    assert [d["tool"] for d in outcome["permission_denials"]] == ["Write"]
    assert outcome["files_changed"] == {}
    assert outcome["handoff"] is None


def test_continuation_receipt_shows_one_save_of_the_first_tasks_draft() -> None:
    receipt = json.loads(
        (RECEIPTS / "recipe-completion-only-continuation.json").read_text()
    )
    outcome = receipt["outcome"]
    assert [c["tool"] for c in outcome["tool_calls"]] == ["Write"]
    assert outcome["permission_denials"] == []
    (saved,) = outcome["files_changed"].values()
    assert saved.strip() == receipt["first_task_draft"]
    assert outcome["handoff"] is None


def test_scoped_output_receipts_show_the_checkpoint_first_and_stopping_on_denial() -> (
    None
):
    done = json.loads((RECEIPTS / "recipe-scoped-output.json").read_text())["outcome"]
    first = done["tool_calls"][0]
    assert (first["tool"], first["input"]["content"]) == ("Write", "PENDING")
    assert done["permission_denials"] == []
    wrong = json.loads(
        (RECEIPTS / "recipe-scoped-output-wrong-grant.json").read_text()
    )["outcome"]
    assert [c["tool"] for c in wrong["tool_calls"]] == ["Write"]
    assert len(wrong["permission_denials"]) == 1 and wrong["files_changed"] == {}


def test_path_rule_receipts_and_the_documented_table_agree() -> None:
    doc = recipes_path().read_text()
    for spelling, (template, approved) in RULE_SPELLINGS.items():
        rule = template.format(workdir=WORKDIR_PLACEHOLDER)
        assert f"`{rule}`" in doc, rule
        receipt = json.loads((RECEIPTS / f"path-rule-{spelling}.json").read_text())
        request = receipt["request"]
        assert request["allowed_tools"] == [
            template.format(workdir=WORKDIR_PLACEHOLDER)
        ]
        outcome = receipt["outcome"]
        assert bool(outcome["files_changed"]) is approved
        assert "outside/b.txt" not in outcome["files_changed"]
        assert len(outcome["permission_denials"]) == (1 if approved else 4)


def test_compound_bash_receipt_matches_the_documented_table() -> None:
    receipt = json.loads((RECEIPTS / "bash-simple-and-compound.json").read_text())
    outcome = receipt["outcome"]
    assert receipt["request"]["allowed_tools"] == ["Bash(touch approved-*)"]
    assert sorted(d["requested"] for d in outcome["permission_denials"]) == [
        "cd sub && touch approved-2",
        "touch approved-5; touch denied-5",
    ]
    doc = recipes_path().read_text()
    for command in (
        "touch approved-1",
        "touch approved-3 && touch approved-4",
        "cd sub && touch approved-2",
        "touch approved-5; touch denied-5",
    ):
        assert f"`{command}`" in doc


def test_receipts_are_sanitised_and_small() -> None:
    files = sorted(RECEIPTS.glob("*.json"))
    assert files
    for path in files:
        text = path.read_text()
        assert not re.search(r"/(tmp|opt|home|Users|nix|workspace)/", text), path.name
        assert "api_key" not in text.lower() and "token" not in text.lower(), path.name
        assert path.stat().st_size < 12_000, path.name


def test_skill_points_at_the_recipes_and_they_ship_beside_it() -> None:
    skill = SKILLS_DIR / "claude-delegation"
    assert recipes_path() == skill / "references" / "recipes.md"
    assert "references/recipes.md" in (skill / "SKILL.md").read_text()


async def test_mcp_serves_the_same_recipes_file_the_native_plugin_registers(
    tmp_path: Path,
) -> None:
    """One file, two routes: the MCP resource and the plugin skill's reference
    are both that file, so there is no second maintained copy."""
    from test_hermes_plugin import RecordingContext, config_for

    from jake_tools.hermes.plugin import register_operations

    uri = "skill://claude-delegation/references/recipes.md"
    async with start_mcp_server(isolated_mcp_env(tmp_path / "home")) as session:
        listed = {str(r.uri) for r in (await session.list_resources()).resources}
        resource = await session.read_resource(uri)
    assert uri in listed
    content = resource.contents[0]
    assert isinstance(content, TextResourceContents)
    assert content.text == recipes_path().read_text()

    config = config_for(tmp_path, {})
    ctx = RecordingContext()
    register_operations(ctx, config, lambda: config)
    registered = ctx.skills["claude-delegation"]
    assert (registered.parent / "references" / "recipes.md").read_text() == (
        content.text
    )
