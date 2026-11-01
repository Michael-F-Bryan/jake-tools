"""The native Hermes adapter's own logic, without a Hermes installation.

Hermes is the external boundary here: ``get_secret`` is a dict lookup and
``ctx`` records registrations. Everything of ours is real: operations run
against the real task service and Clockify client, project-env scripts run
in real bash, and schemas are compared with the real MCP server's. The same
seams are exercised inside the pinned Hermes runtime by
``test_hermes_runtime.py``.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from jake_tools.claude_runs.spawn import worker_environment
from jake_tools.config import Config, load_config
from jake_tools.hermes.plugin import (
    SKILLS_DIR,
    configured_capabilities,
    register_operations,
)
from jake_tools.hermes.profile import (
    SECRET_NAMES,
    profile_config,
    worker_base_environ,
)
from jake_tools.hermes.project_env import (
    WrapperChanged,
    insert_project_env,
    wants_project_env,
)
from jake_tools.hermes.tools import (
    hermes_description,
    make_handler,
    run_coroutine,
    signature_metadata,
    tool_schema,
)
from jake_tools.mcp.server import build_server, packaged_skills
from jake_tools.operations.catalogue import OPERATIONS

WORKSPACE = "0123456789abcdef01234567"


def environ(home: Path, **extra: str) -> dict[str, str]:
    return {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
        "XDG_CONFIG_DIRS": str(home / "etc-xdg"),
        **extra,
    }


class RecordingContext:
    """What Hermes' PluginContext receives from ``register_operations``."""

    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, Path] = {}

    def register_tool(self, **kwargs: Any) -> None:
        self.tools[kwargs["name"]] = kwargs

    def register_skill(self, name: str, path: Path) -> None:
        self.skills[name] = path


def config_for(home: Path, secrets: dict[str, str], **extra: str) -> Config:
    return profile_config(secrets.get, environ(home, **extra))


# --- profile configuration -------------------------------------------------


def test_secrets_come_from_the_profile_never_the_process_environment(
    tmp_path: Path,
) -> None:
    process = environ(
        tmp_path, CLOCKIFY_API_KEY="launch-profile", JIRA_EMAIL="launch@example"
    )
    config = profile_config({"CLOCKIFY_API_KEY": "this-profile"}.get, process)
    assert config.clockify_api_key.value == "this-profile"
    assert config.jira_email.value is None
    assert profile_config({}.get, process).clockify_api_key.value is None


def test_the_profile_home_selects_the_session_store(tmp_path: Path) -> None:
    config = profile_config(
        {}.get, environ(tmp_path), hermes_home=tmp_path / "profiles" / "a"
    )
    assert config.session_store_path.value == tmp_path / "profiles" / "a" / "state.db"


def test_workers_inherit_only_the_allowlist_and_never_secrets(tmp_path: Path) -> None:
    process = environ(
        tmp_path,
        OPENAI_API_KEY="hermes-provider-key",
        DISCORD_BOT_TOKEN="hermes-bot-token",
        CLOCKIFY_API_KEY="jake-secret",
        PYTHONPATH="/plugin/site-packages",
        NIX_LD="/loader",
        XDG_CACHE_HOME=str(tmp_path / "cache"),
    )
    config = profile_config({"CLOCKIFY_API_KEY": "jake-secret"}.get, process)
    assert config.worker_environ is not None
    spawned = worker_environment(config.worker_environ)
    assert {"OPENAI_API_KEY", "DISCORD_BOT_TOKEN"}.isdisjoint(spawned)
    assert SECRET_NAMES.isdisjoint(spawned)
    assert spawned["PYTHONPATH"] == "/plugin/site-packages"
    assert spawned["NIX_LD"] == "/loader"
    assert spawned["XDG_CACHE_HOME"] == str(tmp_path / "cache")
    assert worker_base_environ(process) == config.worker_environ
    assert "worker_environ" not in repr(config)
    assert "worker_environ" not in config.model_dump()


# --- tool adaptation -------------------------------------------------------


def test_hermes_schemas_match_the_mcp_servers(tmp_path: Path) -> None:
    config = load_config(
        environ(tmp_path, TRANSCRIPTION_API_BASE_URL="https://example.invalid")
    )
    mcp_tools = {
        tool.name: tool for tool in asyncio.run(build_server(config).list_tools())
    }
    for operation in OPERATIONS:
        schema = tool_schema(operation, signature_metadata(operation, config))
        assert schema["name"] == operation.name
        assert schema["parameters"] == mcp_tools[operation.name].input_schema
        assert schema["description"] == hermes_description(operation)
        assert f"`jake-tools:{operation.skill}` skill" in schema["description"]
        assert "skill://" not in schema["description"]


def _handler(name: str, config: Config, factory: Any = None) -> Any:
    operation = next(op for op in OPERATIONS if op.name == name)
    return make_handler(
        operation, signature_metadata(operation, config), factory or (lambda: config)
    )


def test_handlers_return_the_operation_result_as_json(tmp_path: Path) -> None:
    config = config_for(tmp_path, {})
    result = json.loads(_handler("claude_list", config)({"limit": 5}, task_id="t"))
    assert result["tasks"] == []
    assert "error" not in result


def test_handler_errors_carry_the_stable_code_and_message(tmp_path: Path) -> None:
    config = config_for(tmp_path, {})
    unknown = json.loads(_handler("claude_status", config)({"task_id": "nope"}))
    assert unknown["code"] == "unknown_task"
    assert unknown["error"] == unknown["message"] == "no such delegated task"
    invalid = json.loads(_handler("claude_list", config)({"limit": "many"}))
    assert invalid["code"] == "invalid_argument"
    assert invalid["detail"] == {"argument": "limit"}


def test_a_profile_without_credentials_gets_missing_credentials(
    tmp_path: Path,
) -> None:
    # The launching process has a key; the profile does not.
    config = profile_config(
        {}.get,
        environ(
            tmp_path,
            CLOCKIFY_API_KEY="launch-profile",
            JAKE_TOOLS_CLOCKIFY_WORKSPACES=WORKSPACE,
        ),
    )
    out = json.loads(_handler("clockify_workspace_catalogue", config)({}))
    assert out["code"] == "missing_credentials"
    assert out["detail"] == {"variables": ["CLOCKIFY_API_KEY"]}
    assert "launch-profile" not in json.dumps(out)


def test_configuration_failures_are_reported_not_raised(tmp_path: Path) -> None:
    def unscoped() -> Config:
        raise RuntimeError("Hermes could not read this profile's CLOCKIFY_API_KEY")

    config = config_for(tmp_path, {})
    out = json.loads(_handler("claude_list", config, unscoped)({}))
    assert out["code"] == "internal_error"
    assert "could not read this profile's CLOCKIFY_API_KEY" in out["message"]


def test_operations_run_inside_an_event_loop_with_the_callers_context() -> None:
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker")

    async def read_marker() -> Any:
        from pydantic import BaseModel

        class Seen(BaseModel):
            value: str

        return Seen(value=marker.get())

    async def caller() -> Any:
        marker.set("profile-a")
        return run_coroutine(read_marker)

    assert asyncio.run(caller()).value == "profile-a"


# --- registration ----------------------------------------------------------


def test_only_granted_capabilities_are_registered(tmp_path: Path) -> None:
    config = config_for(tmp_path, {})
    ctx = RecordingContext()
    register_operations(ctx, ("claude",), config, lambda: config)
    assert set(ctx.tools) == {op.name for op in OPERATIONS if op.capability == "claude"}
    assert {t["toolset"] for t in ctx.tools.values()} == {"jake_tools_claude"}
    assert ctx.skills == {
        "claude-delegation": SKILLS_DIR / "claude-delegation" / "SKILL.md"
    }

    nothing = RecordingContext()
    register_operations(nothing, (), config, lambda: config)
    assert nothing.tools == {} and nothing.skills == {}


def test_transcription_needs_its_endpoint(tmp_path: Path) -> None:
    without = config_for(tmp_path, {})
    ctx = RecordingContext()
    register_operations(ctx, ("transcription",), without, lambda: without)
    assert ctx.tools == {} and ctx.skills == {}

    with_endpoint = config_for(
        tmp_path, {}, TRANSCRIPTION_API_BASE_URL="https://example.invalid"
    )
    ctx = RecordingContext()
    register_operations(ctx, ("transcription",), with_endpoint, lambda: with_endpoint)
    assert set(ctx.tools) == {"transcribe"}
    assert set(ctx.skills) == {"transcription"}


def test_registered_skills_are_the_packaged_sources(tmp_path: Path) -> None:
    config = config_for(tmp_path, {}, TRANSCRIPTION_API_BASE_URL="https://x.invalid")
    ctx = RecordingContext()
    register_operations(
        ctx, ("claude", "clockify", "transcription"), config, lambda: config
    )
    assert sorted(ctx.skills) == sorted(path.name for path in packaged_skills())
    for name, path in ctx.skills.items():
        assert path.is_file() and path.parent.name == name


def test_capability_settings_are_validated() -> None:
    assert configured_capabilities(None) == ()
    assert configured_capabilities(["transcription", "claude"]) == (
        "claude",
        "transcription",
    )
    with pytest.raises(ValueError, match="unknown everything"):
        configured_capabilities(["everything"])
    with pytest.raises(ValueError, match="list of names"):
        configured_capabilities("claude")


# --- project environment insertion -----------------------------------------

BASH = shutil.which("bash")


def wrapper(directory: Path, command: str) -> str:
    """The shape of Hermes' per-command script around the cd and eval."""
    escaped = command.replace("'", "'\\''")
    return "\n".join(
        [
            "export HERMES_AGENT=true",
            f"builtin cd -- {directory} || exit 126",
            f"eval '{escaped}'",
            "__hermes_ec=$?",
            "echo after:$PWD:${PROJECT-unset}",
            "exit $__hermes_ec",
        ]
    )


@pytest.mark.skipif(BASH is None, reason="bash is required")
def test_the_loader_runs_in_the_commands_directory_and_shell(tmp_path: Path) -> None:
    project, other = tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    loader = tmp_path / "loader.sh"
    loader.write_text('export PROJECT="loaded:$PWD"\n')
    command = f"echo cmd:$PROJECT; cd {other}; echo 'quoted \"text\" kept'"
    script = insert_project_env(wrapper(project, command), str(loader))
    out = subprocess.run(
        [str(BASH), "-c", script], capture_output=True, text=True, check=True
    )
    assert out.stdout.splitlines() == [
        f"cmd:loaded:{project}",
        'quoted "text" kept',
        f"after:{other}:loaded:{project}",
    ]


@pytest.mark.skipif(BASH is None, reason="bash is required")
def test_a_failing_loader_stops_the_command(tmp_path: Path) -> None:
    loader = tmp_path / "loader.sh"
    loader.write_text("echo 'project flake failed to load' >&2; return 1\n")
    script = insert_project_env(wrapper(tmp_path, "echo ran"), str(loader))
    out = subprocess.run([str(BASH), "-c", script], capture_output=True, text=True)
    assert out.returncode == 1
    assert "ran" not in out.stdout
    assert "project flake failed to load" in out.stderr


def test_commands_containing_wrapper_lines_are_not_mistaken_for_it(
    tmp_path: Path,
) -> None:
    command = "builtin cd -- /elsewhere || exit 126\neval 'true'"
    script = insert_project_env(wrapper(tmp_path, command), "/loader.sh")
    lines = script.split("\n")
    assert lines[1] == f"builtin cd -- {tmp_path} || exit 126"
    assert lines[2] == ". /loader.sh || exit 1"


def test_an_unrecognised_wrapper_is_refused() -> None:
    with pytest.raises(WrapperChanged):
        insert_project_env("cd /x\neval 'true'", "/loader.sh")
    with pytest.raises(WrapperChanged):
        insert_project_env("builtin cd -- /x || exit 126\necho\neval 'x'", "/l.sh")


def test_only_the_agents_terminal_commands_load_the_project() -> None:
    assert wants_project_env("make", {"bounded_capture": True})
    launch = "mkdir -p /tmp && ( nohup bash -lc 'make' > /tmp/x.log 2>&1; rc=$?; ) &"
    assert wants_project_env(launch, {"rewrite_compound_background": False})
    # Internal commands: pollers, kills, file reads.
    assert not wants_project_env("tail -c +1 /tmp/x.log", {})
    assert not wants_project_env("kill 123", {"timeout": 5})
    assert not wants_project_env(
        "readlink -f /x", {"rewrite_compound_background": False}
    )
