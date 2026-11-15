"""The native Hermes adapter's own logic, without a Hermes installation.

Hermes is the external boundary here: ``get_secret`` is a dict lookup and
``ctx`` records registrations. Everything of ours is real: operations run
against the real task service and Clockify client, and schemas are compared
with the real MCP server's. The same
seams are exercised inside the pinned Hermes runtime by
``test_hermes_runtime.py``.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
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
        environ(
            tmp_path,
            TRANSCRIPTION_API_BASE_URL="https://example.invalid",
            JAKE_TOOLS_ARTIFACTS_ENABLED="true",
        )
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


def test_claude_results_are_compact_and_omit_nulls(tmp_path: Path) -> None:
    config = config_for(tmp_path, {})
    raw = _handler("claude_list", config)({})
    payload = json.loads(raw)
    assert "next_before" not in payload
    assert raw == json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def test_non_claude_results_keep_nullable_fields() -> None:
    from pydantic import BaseModel

    from jake_tools.hermes.tools import success_json
    from jake_tools.mcp.errors import success_result

    class Result(BaseModel):
        optional: str | None = None

    assert all(not op.omit_none for op in OPERATIONS if op.capability != "claude")
    assert json.loads(success_json(Result())) == {"optional": None}
    wire = success_result(Result())
    assert wire.structured_content == {"optional": None}
    content = wire.content[0]
    assert content.type == "text"
    assert json.loads(content.text) == {"optional": None}


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
    # polish_transcript sends transcript text to Claude, so `claude` grants
    # it, with the transcription guide that documents it.
    assert "polish_transcript" in ctx.tools
    assert ctx.skills == {
        "claude-delegation": SKILLS_DIR / "claude-delegation" / "SKILL.md",
        "transcription": SKILLS_DIR / "transcription" / "SKILL.md",
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
    # Local transcription alone never grants sending text to a cloud model.
    assert set(ctx.tools) == {"transcribe"}
    assert set(ctx.skills) == {"transcription"}


def test_registered_skills_are_the_packaged_sources(tmp_path: Path) -> None:
    config = config_for(
        tmp_path,
        {},
        TRANSCRIPTION_API_BASE_URL="https://x.invalid",
        JAKE_TOOLS_ARTIFACTS_ENABLED="true",
    )
    ctx = RecordingContext()
    register_operations(
        ctx,
        ("artifact", "claude", "clockify", "newsletter", "transcription"),
        config,
        lambda: config,
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


def test_artifact_tools_need_both_the_capability_and_the_setting(
    tmp_path: Path,
) -> None:
    disabled = config_for(tmp_path, {})
    enabled = config_for(tmp_path, {}, JAKE_TOOLS_ARTIFACTS_ENABLED="true")

    ctx = RecordingContext()
    register_operations(ctx, ("artifact",), disabled, lambda: disabled)
    assert ctx.tools == {}  # capability alone: the setting is off

    ctx = RecordingContext()
    register_operations(ctx, ("claude", "clockify"), enabled, lambda: enabled)
    assert "artifact_publish" not in ctx.tools  # setting alone: no capability

    ctx = RecordingContext()
    register_operations(ctx, ("artifact",), enabled, lambda: enabled)
    assert set(ctx.tools) == {"artifact_publish", "artifact_read"}
    assert set(ctx.skills) == {"artifact-publishing"}

    # Overwriting an existing artifact is its own grant, never implied.
    ctx = RecordingContext()
    register_operations(ctx, ("artifact_update",), enabled, lambda: enabled)
    assert set(ctx.tools) == {"artifact_update"}
    ctx = RecordingContext()
    register_operations(ctx, ("artifact_update",), disabled, lambda: disabled)
    assert ctx.tools == {}


def test_newsletter_reads_and_publication_are_separate_grants(tmp_path: Path) -> None:
    config = config_for(tmp_path, {})

    reads = RecordingContext()
    register_operations(reads, ("newsletter",), config, lambda: config)
    assert set(reads.tools) == {"newsletter_list", "newsletter_get"}
    assert set(reads.skills) == {"newsletters"}

    writes = RecordingContext()
    register_operations(writes, ("newsletter_publish",), config, lambda: config)
    assert set(writes.tools) == {"newsletter_create", "newsletter_update"}
    assert {t["toolset"] for t in writes.tools.values()} == {
        "jake_tools_newsletter_publish"
    }
    description = writes.tools["newsletter_create"]["schema"]["description"]
    assert "PUBLICATION ACTION" in description
    assert "jake-tools:newsletters" in description

    # Not granted unless named: an unconfigured profile gets neither.
    nothing = RecordingContext()
    register_operations(nothing, (), config, lambda: config)
    assert nothing.tools == {}


def test_microsoft_secrets_come_from_the_profile_not_the_process(
    tmp_path: Path,
) -> None:
    config = profile_config(
        {"MICROSOFT_REFRESH_TOKEN": "profile-token"}.get,
        environ(
            tmp_path,
            MICROSOFT_REFRESH_TOKEN="process-token",
            MICROSOFT_CLIENT_SECRET="process-secret",
        ),
    )
    assert config.microsoft_refresh_token.value == "profile-token"
    assert config.microsoft_client_secret.value is None
    assert {"MICROSOFT_REFRESH_TOKEN", "MICROSOFT_CLIENT_SECRET_FILE"} <= SECRET_NAMES


def test_native_newsletter_tools_read_and_publish_against_a_fake_upstream(
    tmp_path: Path,
) -> None:
    """The native handlers, run for real, end to end (list, create, get)."""
    from fake_microsoft import REFRESH_TOKEN, FakeMicrosoft

    fake = FakeMicrosoft().start()
    try:
        secrets = {"MICROSOFT_REFRESH_TOKEN": REFRESH_TOKEN}
        config = config_for(tmp_path, secrets, **fake.env())
        ctx = RecordingContext()
        register_operations(
            ctx, ("newsletter", "newsletter_publish"), config, lambda: config
        )
        file = tmp_path / "note.txt"
        file.write_bytes(b"attached")

        created = json.loads(
            ctx.tools["newsletter_create"]["handler"](
                {"title": "Hello", "body": "World", "attachments": [str(file)]}
            )
        )
        listed = json.loads(ctx.tools["newsletter_list"]["handler"]({"limit": 5}))
        missing = json.loads(ctx.tools["newsletter_get"]["handler"]({"entry_id": "99"}))
        bad_id = json.loads(
            ctx.tools["newsletter_get"]["handler"]({"entry_id": "../x"})
        )
    finally:
        fake.stop()

    assert created["verified"] is True
    assert created["entry"]["attachments"][0]["size_bytes"] == 8
    assert [e["title"] for e in listed["entries"]] == ["Hello"]
    assert listed["complete"] is True
    assert missing["code"] == "invalid_argument" and "99" in missing["error"]
    assert bad_id["code"] == "invalid_argument"
    assert REFRESH_TOKEN not in json.dumps([created, listed, missing, bad_id])
