from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
UPSTREAM_RE = re.compile(r"upstream ([0-9a-f]+)", re.IGNORECASE)


def _pinned_hermes_agent_rev() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    source = data["tool"]["uv"]["sources"]["hermes-agent"]
    rev = source.get("rev")
    if not isinstance(rev, str) or not rev:
        pytest.fail("hermes-agent source in pyproject.toml has no pinned rev")
    return rev


def _hermes_upstream_rev() -> str:
    result = subprocess.run(
        ["hermes", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = UPSTREAM_RE.search(result.stdout)
    if match is None:
        pytest.fail(
            f"hermes --version did not report an upstream revision:\n{result.stdout}"
        )
    return match.group(1)


def test_hermes_cli_matches_pinned_hermes_agent_rev() -> None:
    if shutil.which("hermes") is None:
        pytest.skip("hermes executable not available")

    pinned_rev = _pinned_hermes_agent_rev()
    upstream_rev = _hermes_upstream_rev()

    assert pinned_rev.startswith(upstream_rev), (
        "hermes CLI upstream revision does not match pyproject.toml pin: "
        f"cli={upstream_rev}, pinned={pinned_rev}"
    )
