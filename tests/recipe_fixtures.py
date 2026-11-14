"""Shared pieces of the delegation-recipe tests (``test_delegation_recipes*.py``).

The copyable recipes live in one place, the packaged skill reference
``claude-delegation/references/recipes.md``: each is a ``claude_start``
argument object in a JSON fence preceded by a ``<!-- recipe: NAME -->``
marker. The live tests run those exact objects (only ``working_directory``,
``model`` and ``effort`` are filled in), so the documented text is the tested
text. Their sanitised outcomes are kept as receipts under
``tests/fixtures/delegation_recipes/receipts``; the hermetic tests there pin
that each receipt's brief and grants are still the documented ones.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from jake_tools.mcp.server import packaged_skills

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "delegation_recipes"
PROJECT = FIXTURES / "project"
RECEIPTS = FIXTURES / "receipts"
WORKDIR_PLACEHOLDER = "/abs/path/to/project"
MARKER = re.compile(r"<!-- recipe: ([a-z0-9-]+) -->\n```json\n(.*?)\n```", re.DOTALL)


def recipes_path() -> Path:
    skill = next(p for p in packaged_skills() if p.name == "claude-delegation")
    return skill / "references" / "recipes.md"


def load_recipes() -> dict[str, dict[str, Any]]:
    """Every documented recipe's ``claude_start`` arguments, by name."""
    found = {
        name: json.loads(body)
        for name, body in MARKER.findall(recipes_path().read_text())
    }
    assert found, "recipes.md has no `<!-- recipe: NAME -->` JSON fences"
    return found


def recipe_request(name: str, project: Path | None = None) -> dict[str, Any]:
    """A documented recipe as ``claude_start`` arguments, verbatim except that
    ``working_directory`` (when the recipe has one) names ``project``."""
    request = dict(load_recipes()[name])
    if project is not None and "working_directory" in request:
        assert request["working_directory"] == WORKDIR_PLACEHOLDER
        request["working_directory"] = str(project)
    return request


def stage_project(destination: Path) -> Path:
    """A fresh copy of the synthetic fixture project to run a recipe in."""
    shutil.copytree(PROJECT, destination)
    return destination


def tree(root: Path) -> dict[str, str]:
    """``{relative path: text}`` for every file under ``root``."""
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def cli_version() -> str:
    completed = subprocess.run(
        ["claude", "--version"], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or "unknown"


def sanitise(value: Any, workdir: Path) -> Any:
    """Replace the scratch path with the placeholder the docs use."""
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    text = text.replace(str(workdir.resolve()), WORKDIR_PLACEHOLDER)
    text = text.replace(str(workdir), WORKDIR_PLACEHOLDER)
    return json.loads(text)


def write_receipt(
    name: str, workdir: Path, directory: Path | None, receipt: dict[str, Any]
) -> None:
    """Keep a sanitised receipt when ``JAKE_RECIPE_RECEIPT_DIR`` asks for one."""
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    clean = sanitise({"cli_version": cli_version(), **receipt}, workdir)
    (directory / f"{name}.json").write_text(
        json.dumps(clean, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


# What the path-rule matrix (test_delegation_recipes_live.py) establishes.
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
