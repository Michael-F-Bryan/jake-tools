"""Configuration is explicit: flags and environment variables, nothing else.

Every test runs the real entrypoints in subprocesses from a hostile working
directory: a ``.env`` and a ``config.toml`` in the working directory and in its
ancestors, ``$HOME``, ``$XDG_CONFIG_HOME`` and ``$XDG_CONFIG_DIRS`` all set
every setting to a recognisable wrong value. None of it may have any effect on
the ``jake-tools`` CLI or on ``python -m jake_tools.mcp``; only flags and the
process environment do, flags first.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from jake_tools.config import load_config
from jake_tools.mcp.options import FLAGS, Flag

WORKSPACE = "60176d9aa84b61134aaef479"
POISON = "FROM-A-FILE"

DOTENV = f"""\
JAKE_TOOLS_JIRA_PROJECT={POISON}
JAKE_TOOLS_CLOCKIFY_CLIENT={POISON}
JAKE_TOOLS_CLOCKIFY_WORKSPACES={POISON}
JAKE_TOOLS_CLAUDE_MAX_TURNS=999
JAKE_TOOLS_ARTIFACTS_ENABLED=true
JAKE_TOOLS_RUNS_DIR=/nonexistent/{POISON}
TRANSCRIPTION_API_BASE_URL=http://127.0.0.1:9
CLOCKIFY_API_KEY={POISON}
CLOCKIFY_API_BASE_URL=http://127.0.0.1:1
MICROSOFT_REFRESH_TOKEN={POISON}
MICROSOFT_CLIENT_SECRET={POISON}
JAKE_TOOLS_MICROSOFT_TENANT_ID={POISON}
JAKE_TOOLS_MICROSOFT_CLIENT_ID={POISON}
JAKE_TOOLS_MICROSOFT_ACCOUNT=poison@example.test
"""

TOML = f"""\
[clockify]
jira_project = "{POISON}"
client = "{POISON}"
workspaces = ["{POISON}"]
api_key = "{POISON}"
[claude]
max_turns = 999
model = "{POISON}"
runs_dir = "/nonexistent/{POISON}"
[artifacts]
enabled = true
[transcription]
api_base_url = "http://127.0.0.1:9"
[microsoft]
tenant_id = "{POISON}"
account = "poison@example.test"
refresh_token = "{POISON}"
"""


@dataclass(frozen=True)
class Hostile:
    """A working directory whose every possible file source is poisoned."""

    home: Path
    cwd: Path

    def env(self, **extra: str) -> dict[str, str]:
        env = {
            "HOME": str(self.home),
            "PATH": os.environ.get("PATH", ""),
            "XDG_CONFIG_HOME": str(self.home / ".config"),
            "XDG_CONFIG_DIRS": str(self.home / "etc-xdg"),
            "XDG_STATE_HOME": str(self.home / ".local" / "state"),
            "XDG_CACHE_HOME": str(self.home / ".cache"),
            "PYTHONUNBUFFERED": "1",
        }
        env.update(extra)
        return env

    def run(
        self, module: str, *args: str, **env: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            env=self.env(**env),
            cwd=self.cwd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )


@pytest.fixture
def hostile(tmp_path: Path) -> Hostile:
    root = tmp_path / "ancestor"
    home = tmp_path / "home"
    cwd = root / "project" / "deep"
    cwd.mkdir(parents=True)
    home.mkdir()
    for directory in (root, root / "project", cwd, home):
        (directory / ".env").write_text(DOTENV)
        (directory / "config.toml").write_text(TOML)
    for config_dir in (
        home / ".config" / "jake-tools",
        home / "etc-xdg" / "jake-tools",
    ):
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(TOML)
    return Hostile(home=home, cwd=cwd)


def settings_block(output: str) -> dict[str, str]:
    """``doctor``'s Settings section as ``{name: "value  [source]"}``."""
    block = output.split("Settings\n", 1)[1].split("\nChecks", 1)[0]
    return {
        match["name"]: match["rest"]
        for line in block.splitlines()
        if (match := re.match(r"  (?P<name>\S+) = (?P<rest>.*)", line))
    }


def doctor(
    hostile: Hostile, *args: str, **env: str
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    result = hostile.run("jake_tools.mcp", "doctor", *args, **env)
    assert POISON not in result.stdout + result.stderr
    return result, settings_block(result.stdout)


# --- MCP: doctor ------------------------------------------------------------


def test_mcp_ignores_toml_and_dotenv_files_everywhere(hostile: Hostile) -> None:
    result, settings = doctor(hostile)

    assert len(settings) == 32, result.stdout
    for name, rest in settings.items():
        assert rest.endswith("[default]") or rest.startswith("unset"), (name, rest)
    assert settings["clockify.workspaces"] == "(empty)  [default]"
    assert settings["claude.max_turns"] == "20  [default]"
    assert settings["artifacts.enabled"] == "False  [default]"
    assert settings["CLOCKIFY_API_KEY"] == "unset  [default]"
    assert settings["transcription.api_base_url"] == "unset  [default]"
    state = hostile.home / ".local" / "state" / "jake-tools" / "runs"
    assert settings["claude.runs_dir"] == f"{state}  [default]"
    assert "XDG config" not in result.stdout
    assert "Config files" not in result.stdout


def test_mcp_environment_sets_the_clockify_workspace_allowlist(
    hostile: Hostile,
) -> None:
    _, settings = doctor(hostile, JAKE_TOOLS_CLOCKIFY_WORKSPACES=WORKSPACE)

    assert settings["clockify.workspaces"] == (
        f"{WORKSPACE}  [env (JAKE_TOOLS_CLOCKIFY_WORKSPACES)]"
    )
    assert settings["claude.max_turns"] == "20  [default]"


def test_mcp_flags_override_the_environment(hostile: Hostile) -> None:
    _, settings = doctor(
        hostile,
        "--clockify-workspace",
        "ws-flag-1",
        "--clockify-workspace",
        "ws-flag-2",
        "--claude-max-turns",
        "5",
        "--no-artifacts",
        JAKE_TOOLS_CLOCKIFY_WORKSPACES=WORKSPACE,
        JAKE_TOOLS_CLAUDE_MAX_TURNS="9",
        JAKE_TOOLS_ARTIFACTS_ENABLED="true",
        JAKE_TOOLS_JIRA_PROJECT="ENV",
    )

    assert settings["clockify.workspaces"] == (
        "ws-flag-1, ws-flag-2  [flag (clockify.workspaces)]"
    )
    assert settings["claude.max_turns"] == "5  [flag (claude.max_turns)]"
    assert settings["artifacts.enabled"] == "False  [flag (artifacts.enabled)]"
    # No flag for this one: the environment still applies.
    assert settings["clockify.jira_project"] == ("ENV  [env (JAKE_TOOLS_JIRA_PROJECT)]")


def test_mcp_secrets_keep_file_support_and_redaction(
    hostile: Hostile, tmp_path: Path
) -> None:
    secret = tmp_path / "clockify-key"
    secret.write_text("key-from-a-secret-file\n")

    result, settings = doctor(hostile, CLOCKIFY_API_KEY_FILE=str(secret))

    assert settings["CLOCKIFY_API_KEY"] == "set  [env (CLOCKIFY_API_KEY_FILE)]"
    assert "key-from-a-secret-file" not in result.stdout + result.stderr


def test_mcp_a_bad_environment_value_fails_naming_the_variable(
    hostile: Hostile,
) -> None:
    result = hostile.run("jake_tools.mcp", "doctor", JAKE_TOOLS_CLAUDE_MAX_TURNS="x")

    assert result.returncode == 1
    assert "JAKE_TOOLS_CLAUDE_MAX_TURNS" in result.stdout


# --- MCP: serve ---------------------------------------------------------------


async def tool_names(hostile: Hostile, *args: str, **env: str) -> set[str]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "jake_tools.mcp", "serve", *args],
        env=hostile.env(**env),
        cwd=hostile.cwd,
    )
    async with (
        stdio_client(params, errlog=sys.stderr) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return {tool.name for tool in (await session.list_tools()).tools}


async def test_serve_ignores_files_and_honours_environment_then_flags(
    hostile: Hostile,
) -> None:
    # The poisoned .env/config.toml both enable transcription and artifacts.
    plain = await tool_names(hostile)
    assert "transcribe" not in plain
    assert not {"artifact_publish", "artifact_read"} & plain

    from_env = await tool_names(
        hostile,
        TRANSCRIPTION_API_BASE_URL="http://127.0.0.1:9",
        JAKE_TOOLS_ARTIFACTS_ENABLED="true",
    )
    assert {"transcribe", "artifact_publish", "artifact_read"} <= from_env

    flag_beats_env = await tool_names(
        hostile,
        "--no-artifacts",
        "--transcription-api-base-url",
        "http://127.0.0.1:9",
        JAKE_TOOLS_ARTIFACTS_ENABLED="true",
    )
    assert "transcribe" in flag_beats_env
    assert not {"artifact_publish", "artifact_read"} & flag_beats_env


# --- the jake-tools CLI ---------------------------------------------------------


def cli(hostile: Hostile, *args: str, **env: str) -> str:
    result = hostile.run("jake_tools", *args, **env)
    assert result.returncode != 0, result.stdout
    return result.stdout + result.stderr


def test_cli_ignores_dotenv_in_the_cwd_and_its_ancestors_for_clockify(
    hostile: Hostile,
) -> None:
    # The .env supplies CLOCKIFY_API_KEY; were it loaded this would reach
    # the (refused) network instead of stopping at the missing key.
    assert "Clockify API key is required" in cli(hostile, "clockify", "whoami")

    from_env = cli(
        hostile,
        "clockify",
        "whoami",
        CLOCKIFY_API_KEY="real-env-key",
        CLOCKIFY_API_BASE_URL="http://127.0.0.1:1",
    )
    assert "Clockify API key is required" not in from_env
    assert "real-env-key" not in from_env


def test_cli_flags_override_the_environment(hostile: Hostile) -> None:
    out = cli(
        hostile,
        "clockify",
        "whoami",
        "--api-key",
        "flag-key",
        "--api-base-url",
        "http://127.0.0.1:1",
        CLOCKIFY_API_KEY="env-key",
        CLOCKIFY_API_BASE_URL="http://127.0.0.1:2",
    )

    assert "port=1" in out and "port=2" not in out
    assert "flag-key" not in out and "env-key" not in out


def test_cli_files_ignores_files_and_reads_microsoft_settings_from_env(
    hostile: Hostile,
) -> None:
    # The .env and config.toml carry a refresh token, tenant, client and account.
    files = ("files", "get", "drive", "item", "--json")
    unconfigured = cli(hostile, *files)
    assert "missing_credentials" in unconfigured
    assert "MICROSOFT_REFRESH_TOKEN" in unconfigured

    guid = "11111111-1111-1111-1111-111111111111"
    configured = cli(
        hostile,
        *files,
        MICROSOFT_REFRESH_TOKEN="env-token",
        JAKE_TOOLS_MICROSOFT_TENANT_ID=guid,
        JAKE_TOOLS_MICROSOFT_CLIENT_ID=guid,
        JAKE_TOOLS_MICROSOFT_ACCOUNT="env@example.test",
        JAKE_TOOLS_MICROSOFT_AUTHORITY_URL="http://127.0.0.1:1",
    )
    assert "missing_credentials" not in configured
    assert "env-token" not in configured


def test_cli_jira_sync_project_and_client_come_from_flag_then_env(
    hostile: Hostile,
) -> None:
    result = hostile.run("jake_tools", "clockify", "jira-sync", "--help")

    assert result.returncode == 0
    assert "JAKE_TOOLS_JIRA_PROJECT" in result.stdout
    assert "JAKE_TOOLS_CLOCKIFY_CLIENT" in result.stdout
    assert POISON not in result.stdout


# --- the flag table and the dependency ---------------------------------------


def sample(flag: Flag) -> tuple[str, object]:
    """A valid raw environment value and the flag value that must beat it."""
    if flag.is_bool:
        return "true", False
    if isinstance(flag.type, click.Choice):
        return "low", "high"
    if flag.type is int:
        return "3", 4
    if flag.type is float:
        return "1.5", 2.5
    if isinstance(flag.type, click.Path):
        return "/from/env", Path("/from/flag")
    if flag.multiple:
        return "env-a,env-b", ("flag-a", "flag-b")
    return "from-env", "from-flag"


@pytest.mark.parametrize("flag", FLAGS, ids=lambda flag: flag.flag)
def test_each_flag_beats_the_environment_variable_it_documents(
    flag: Flag, tmp_path: Path
) -> None:
    raw, override = sample(flag)
    env = {"HOME": str(tmp_path), flag.env: raw}

    from_env = {s.name: s for s in load_config(env).settings()}[flag.setting]
    from_flag = {
        s.name: s
        for s in load_config(env, overrides={flag.setting: override}).settings()
    }[flag.setting]

    assert (from_env.source, from_env.origin) == ("env", flag.env)
    assert from_flag.source == "flag"
    assert from_flag.value != from_env.value
    assert not from_flag.secret  # secrets have no flag


def test_flags_never_cover_secrets() -> None:
    assert not [flag for flag in FLAGS if "token" in flag.flag or "secret" in flag.flag]


def test_nothing_in_the_source_reads_toml_or_dotenv() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "jake_tools"
    pattern = re.compile(r"tomllib|load_dotenv|^\s*(import|from)\s+dotenv", re.M)
    offenders = [
        path
        for path in sorted(root.rglob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_python_dotenv_is_not_a_dependency() -> None:
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    assert not [d for d in declared["dependencies"] if "dotenv" in d.lower()]
