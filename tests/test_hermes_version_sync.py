from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
DUMP_COMMIT_RE = re.compile(
    r"^version:\s+.*?\[([0-9a-f]+)\]",
    re.MULTILINE | re.IGNORECASE,
)


def _pinned_hermes_agent_rev() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    source = data["tool"]["uv"]["sources"]["hermes-agent"]
    rev = source.get("rev")
    if not isinstance(rev, str) or not rev:
        pytest.fail("hermes-agent source in pyproject.toml has no pinned rev")
    return rev


def _resolve_user_hermes_executable() -> str | None:
    """Return the first ``hermes`` on PATH that is not this project's venv shim."""
    venv_bin = (Path(sys.prefix) / "bin").resolve()
    seen: set[str] = set()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        candidate = Path(directory) / "hermes"
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.parent == venv_bin:
            continue
        return str(resolved)
    return None


def _parse_installed_rev_from_dump(stdout: str) -> str:
    match = DUMP_COMMIT_RE.search(stdout)
    if match is None:
        pytest.fail(f"hermes dump did not report an installed revision:\n{stdout}")
    return match.group(1)


def _hermes_installed_rev(hermes_executable: str) -> str:
    """Return the short git SHA of the running hermes CLI install.

    ``hermes --version`` labels ``origin/main`` as "upstream", which is the
    latest tip on GitHub — not the installed checkout. ``hermes dump`` reports
    the running commit via ``git rev-parse HEAD`` (or the baked Docker SHA).
    """
    result = subprocess.run(
        [hermes_executable, "dump"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _parse_installed_rev_from_dump(result.stdout)


def test_parse_installed_rev_from_dump_accepts_both_version_formats() -> None:
    new_format = """\
--- hermes dump ---
version:          0.17.0 [44ddc552] (2026-06-30)
"""
    old_format = """\
--- hermes dump ---
version:          0.16.0 (2026.6.5) [3a28147a]
"""
    assert _parse_installed_rev_from_dump(new_format) == "44ddc552"
    assert _parse_installed_rev_from_dump(old_format) == "3a28147a"


def test_hermes_cli_matches_pinned_hermes_agent_rev() -> None:
    hermes_executable = _resolve_user_hermes_executable()
    if hermes_executable is None:
        pytest.skip("user hermes executable not available")

    pinned_rev = _pinned_hermes_agent_rev()
    installed_rev = _hermes_installed_rev(hermes_executable)

    assert pinned_rev.startswith(installed_rev), (
        "hermes CLI installed revision does not match pyproject.toml pin: "
        f"cli={installed_rev}, pinned={pinned_rev}"
    )
