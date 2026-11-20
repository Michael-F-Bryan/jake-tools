from __future__ import annotations

from pathlib import Path

import pytest

from jake_tools.config import (
    DEFAULT_CLAUDE_MAX_BUDGET_USD,
    DEFAULT_CLAUDE_MAX_LIVE_TASKS,
    DEFAULT_CLAUDE_MAX_TURNS,
    DEFAULT_CLAUDE_TIMEOUT_SECONDS,
    DEFAULT_CLOCKIFY_CLIENT,
    DEFAULT_JIRA_PROJECT,
    DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS,
    ConfigError,
    XdgPaths,
    load_config,
)


def _env(home: Path, **extra: str) -> dict[str, str]:
    return {"HOME": str(home), **extra}


def test_xdg_defaults_follow_the_spec_fallbacks(tmp_path: Path) -> None:
    xdg = XdgPaths.from_environ(_env(tmp_path))

    assert xdg.state_home == tmp_path / ".local" / "state"
    assert xdg.cache_home == tmp_path / ".cache"
    assert xdg.app_state_dir == tmp_path / ".local" / "state" / "jake-tools"
    assert xdg.app_cache_dir == tmp_path / ".cache" / "jake-tools"
    # There is no config directory: nothing is ever read from one.
    assert not hasattr(xdg, "config_home")
    assert not hasattr(xdg, "config_files")


def test_xdg_ignores_relative_values_and_honours_absolute_ones(tmp_path: Path) -> None:
    xdg = XdgPaths.from_environ(
        _env(
            tmp_path,
            XDG_STATE_HOME=str(tmp_path / "state"),
            XDG_CACHE_HOME="relative/cache",
        )
    )

    assert xdg.state_home == tmp_path / "state"
    assert xdg.cache_home == tmp_path / ".cache"


def test_defaults_when_nothing_is_configured(tmp_path: Path) -> None:
    config = load_config(_env(tmp_path))

    assert config.jira_project.value == DEFAULT_JIRA_PROJECT
    assert config.clockify_client.value == DEFAULT_CLOCKIFY_CLIENT
    assert config.clockify_workspaces.value == ()
    assert config.session_store_path.value is None
    assert (
        config.runs_dir.value == tmp_path / ".local" / "state" / "jake-tools" / "runs"
    )
    assert config.claude_max_turns.value == DEFAULT_CLAUDE_MAX_TURNS
    assert config.claude_timeout_seconds.value == DEFAULT_CLAUDE_TIMEOUT_SECONDS
    assert config.claude_max_budget_usd.value == DEFAULT_CLAUDE_MAX_BUDGET_USD
    assert config.claude_effort.value is None
    assert config.transcription_api_base_url.value is None
    assert config.transcription_api_token.value is None
    assert config.transcription_api_token.secret
    assert (
        config.transcription_timeout_seconds.value
        == DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS
    )
    assert not config.has_transcription_endpoint
    assert all(setting.source == "default" for setting in config.settings())
    assert not config.has_clockify_credentials
    assert config.missing_jira_variables == (
        "JIRA_BASE_URL",
        "JIRA_EMAIL",
        "JIRA_API_TOKEN",
    )


def test_precedence_is_flag_then_env_then_default(tmp_path: Path) -> None:
    config = load_config(
        _env(
            tmp_path,
            JAKE_TOOLS_CLOCKIFY_CLIENT="Env Client",
            JAKE_TOOLS_JIRA_PROJECT="ENV",
            JAKE_TOOLS_CLAUDE_MAX_TURNS="7",
        ),
        overrides={"clockify.jira_project": "FLAG"},
    )

    assert (config.jira_project.value, config.jira_project.source) == ("FLAG", "flag")
    assert (config.clockify_client.value, config.clockify_client.source) == (
        "Env Client",
        "env",
    )
    assert config.clockify_client.origin == "JAKE_TOOLS_CLOCKIFY_CLIENT"
    assert config.claude_max_turns.value == 7
    assert config.claude_max_turns.origin == "JAKE_TOOLS_CLAUDE_MAX_TURNS"
    assert config.claude_timeout_seconds.source == "default"
    assert {setting.source for setting in config.settings()} == {
        "flag",
        "env",
        "default",
    }


def test_a_none_flag_does_not_override_the_environment(tmp_path: Path) -> None:
    config = load_config(
        _env(tmp_path, JAKE_TOOLS_JIRA_PROJECT="ENV"),
        overrides={"clockify.jira_project": None},
    )

    assert (config.jira_project.value, config.jira_project.source) == ("ENV", "env")


# (setting, environment variable, raw value, resolved value): every non-secret
# deployment setting is set by exactly this variable.
ENV_MAPPINGS: tuple[tuple[str, str, str, object], ...] = (
    ("claude.runs_dir", "JAKE_TOOLS_RUNS_DIR", "/r", Path("/r")),
    ("session_store.path", "JAKE_TOOLS_SESSION_STORE", "/s.db", Path("/s.db")),
    ("clockify.jira_project", "JAKE_TOOLS_JIRA_PROJECT", "ZZ", "ZZ"),
    ("clockify.client", "JAKE_TOOLS_CLOCKIFY_CLIENT", "Acme", "Acme"),
    ("clockify.api_base_url", "CLOCKIFY_API_BASE_URL", "http://x/", "http://x/"),
    (
        "clockify.workspaces",
        "JAKE_TOOLS_CLOCKIFY_WORKSPACES",
        "60176d9aa84b61134aaef479",
        ("60176d9aa84b61134aaef479",),
    ),
    ("claude.model", "JAKE_TOOLS_CLAUDE_MODEL", "m", "m"),
    ("claude.effort", "JAKE_TOOLS_CLAUDE_EFFORT", "low", "low"),
    ("claude.max_turns", "JAKE_TOOLS_CLAUDE_MAX_TURNS", "3", 3),
    ("claude.timeout_seconds", "JAKE_TOOLS_CLAUDE_TIMEOUT_SECONDS", "12.5", 12.5),
    ("claude.max_budget_usd", "JAKE_TOOLS_CLAUDE_MAX_BUDGET_USD", "0.25", 0.25),
    ("claude.max_live_tasks", "JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS", "4", 4),
    ("claude.account_context", "JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT", "a:1", "a:1"),
    (
        "transcription.api_base_url",
        "TRANSCRIPTION_API_BASE_URL",
        "http://t",
        "http://t",
    ),
    (
        "transcription.timeout_seconds",
        "JAKE_TOOLS_TRANSCRIPTION_TIMEOUT_SECONDS",
        "42",
        42.0,
    ),
    ("microsoft.tenant_id", "JAKE_TOOLS_MICROSOFT_TENANT_ID", "t", "t"),
    ("microsoft.client_id", "JAKE_TOOLS_MICROSOFT_CLIENT_ID", "c", "c"),
    ("microsoft.account", "JAKE_TOOLS_MICROSOFT_ACCOUNT", "a@x.test", "a@x.test"),
    (
        "microsoft.account_object_id",
        "JAKE_TOOLS_MICROSOFT_ACCOUNT_OBJECT_ID",
        "o",
        "o",
    ),
    (
        "microsoft.authority_url",
        "JAKE_TOOLS_MICROSOFT_AUTHORITY_URL",
        "http://a",
        "http://a",
    ),
    (
        "newsletters.graph_root",
        "JAKE_TOOLS_NEWSLETTERS_GRAPH_ROOT",
        "http://g",
        "http://g",
    ),
    (
        "newsletters.sharepoint_root",
        "JAKE_TOOLS_NEWSLETTERS_SHAREPOINT_ROOT",
        "http://sp",
        "http://sp",
    ),
    ("files.graph_root", "JAKE_TOOLS_FILES_GRAPH_ROOT", "http://fg", "http://fg"),
    (
        "files.sharepoint_hosts",
        "JAKE_TOOLS_FILES_SHAREPOINT_HOSTS",
        "a.test,b.test",
        ("a.test", "b.test"),
    ),
    ("artifacts.enabled", "JAKE_TOOLS_ARTIFACTS_ENABLED", "true", True),
    (
        "hindsight.api_url",
        "JAKE_TOOLS_HINDSIGHT_API_URL",
        "http://127.0.0.1:8888/",
        "http://127.0.0.1:8888",
    ),
    (
        "hindsight.bank_id",
        "JAKE_TOOLS_HINDSIGHT_BANK_ID",
        "michael-personal",
        "michael-personal",
    ),
    ("hindsight.access", "JAKE_TOOLS_HINDSIGHT_ACCESS", "read", "read"),
)


def test_every_non_secret_setting_has_a_tested_environment_variable(
    tmp_path: Path,
) -> None:
    covered = {setting for setting, *_ in ENV_MAPPINGS}
    non_secret = {
        setting.name
        for setting in load_config(_env(tmp_path)).settings()
        if not setting.secret
    }
    assert covered == non_secret


@pytest.mark.parametrize(("setting", "variable", "raw", "expected"), ENV_MAPPINGS)
def test_each_setting_is_set_by_its_environment_variable(
    tmp_path: Path, setting: str, variable: str, raw: str, expected: object
) -> None:
    config = load_config(_env(tmp_path, **{variable: raw}))

    resolved = {s.name: s for s in config.settings()}[setting]
    assert resolved.value == expected
    assert (resolved.source, resolved.origin) == ("env", variable)
    assert all(s.source == "default" for s in config.settings() if s is not resolved)


def test_workspace_allowlist_from_environment_is_comma_separated(
    tmp_path: Path,
) -> None:
    config = load_config(
        _env(tmp_path, JAKE_TOOLS_CLOCKIFY_WORKSPACES="ws-a, ws-b,,ws-c ")
    )

    assert config.clockify_workspaces.value == ("ws-a", "ws-b", "ws-c")
    assert config.clockify_workspaces.source == "env"


def test_secrets_come_from_the_environment_or_a_name_file(tmp_path: Path) -> None:
    secret_file = tmp_path / "jira-token"
    secret_file.write_text("from-file\n")

    config = load_config(
        _env(
            tmp_path,
            CLOCKIFY_API_KEY="direct\n",
            CLOCKIFY_API_KEY_FILE=str(secret_file),
            JIRA_API_TOKEN_FILE=str(secret_file),
        )
    )

    assert config.clockify_api_key.value == "direct"
    assert config.clockify_api_key.origin == "CLOCKIFY_API_KEY"
    assert config.jira_api_token.value == "from-file"
    assert config.jira_api_token.origin == "JIRA_API_TOKEN_FILE"
    assert config.jira_email.value is None
    assert config.has_clockify_credentials
    assert config.missing_jira_variables == ("JIRA_BASE_URL", "JIRA_EMAIL")


def test_secret_display_never_shows_the_value(tmp_path: Path) -> None:
    config = load_config(_env(tmp_path, CLOCKIFY_API_KEY="super-secret"))

    assert config.clockify_api_key.display_value() == "set"
    assert config.jira_api_token.display_value() == "unset"
    assert "super-secret" not in config.clockify_api_key.display_source()


def test_secret_file_that_cannot_be_read_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(_env(tmp_path, CLOCKIFY_API_KEY_FILE=str(tmp_path / "missing")))

    assert "CLOCKIFY_API_KEY_FILE" in str(excinfo.value)


def test_session_store_defaults_under_hermes_home(tmp_path: Path) -> None:
    config = load_config(_env(tmp_path, HERMES_HOME=str(tmp_path / "hermes")))

    assert config.hermes_home == tmp_path / "hermes"
    assert config.session_store_path.value == tmp_path / "hermes" / "state.db"
    assert config.session_store_path.source == "default"
    assert config.session_store_path.origin == "HERMES_HOME"


def test_env_overrides_runs_dir_and_session_store(tmp_path: Path) -> None:
    config = load_config(
        _env(
            tmp_path,
            JAKE_TOOLS_RUNS_DIR="~/runs",
            JAKE_TOOLS_SESSION_STORE=str(tmp_path / "store.db"),
        )
    )

    assert config.runs_dir.value == Path("~/runs").expanduser()
    assert config.runs_dir.source == "env"
    assert config.session_store_path.value == tmp_path / "store.db"


def test_wrong_type_names_the_setting_and_where_it_came_from(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(_env(tmp_path, JAKE_TOOLS_CLAUDE_MAX_TURNS="lots"))

    message = str(excinfo.value)
    assert "claude.max_turns" in message
    assert "JAKE_TOOLS_CLAUDE_MAX_TURNS" in message
    assert "integer" in message


def test_a_bad_flag_value_names_the_flag(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="flag claude.max_turns"):
        load_config(_env(tmp_path), overrides={"claude.max_turns": "lots"})


def test_invalid_effort_lists_the_choices(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(_env(tmp_path, JAKE_TOOLS_CLAUDE_EFFORT="turbo"))

    assert "low, medium, high" in str(excinfo.value)
    assert "JAKE_TOOLS_CLAUDE_EFFORT" in str(excinfo.value)


def test_settings_listing_is_complete_and_stable(tmp_path: Path) -> None:
    config = load_config(_env(tmp_path))

    names = [setting.name for setting in config.settings()]
    assert names == [
        "CLOCKIFY_API_KEY",
        "JIRA_BASE_URL",
        "JIRA_EMAIL",
        "JIRA_API_TOKEN",
        "clockify.jira_project",
        "clockify.client",
        "clockify.api_base_url",
        "clockify.workspaces",
        "session_store.path",
        "claude.runs_dir",
        "claude.model",
        "claude.effort",
        "claude.max_turns",
        "claude.timeout_seconds",
        "claude.max_budget_usd",
        "claude.max_live_tasks",
        "claude.account_context",
        "transcription.api_base_url",
        "TRANSCRIPTION_API_TOKEN",
        "transcription.timeout_seconds",
        "microsoft.tenant_id",
        "microsoft.client_id",
        "microsoft.account",
        "microsoft.account_object_id",
        "microsoft.authority_url",
        "MICROSOFT_REFRESH_TOKEN",
        "MICROSOFT_CLIENT_SECRET",
        "newsletters.graph_root",
        "newsletters.sharepoint_root",
        "files.graph_root",
        "files.sharepoint_hosts",
        "artifacts.enabled",
        "hindsight.api_url",
        "hindsight.bank_id",
        "hindsight.access",
    ]
    assert config.runs_dir.display_value().startswith(str(tmp_path))


def test_clockify_api_base_url_defaults_to_the_public_api(tmp_path: Path) -> None:
    config = load_config(_env(tmp_path))
    assert config.clockify_api_base_url.value == "https://api.clockify.me/api/v1"

    overridden = load_config(
        _env(tmp_path, CLOCKIFY_API_BASE_URL="http://127.0.0.1:9/api/v1")
    )
    assert overridden.clockify_api_base_url.value == "http://127.0.0.1:9/api/v1"
    assert overridden.clockify_api_base_url.source == "env"


def test_transcription_settings_follow_env_and_the_token_is_a_secret(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("s3cret\n")

    unset = load_config(_env(tmp_path))
    assert unset.transcription_api_base_url.value is None
    assert not unset.has_transcription_endpoint

    from_env = load_config(
        _env(
            tmp_path,
            TRANSCRIPTION_API_BASE_URL="http://from-env:2",
            JAKE_TOOLS_TRANSCRIPTION_TIMEOUT_SECONDS="7.5",
            TRANSCRIPTION_API_TOKEN_FILE=str(token_file),
        )
    )
    assert from_env.has_transcription_endpoint
    assert from_env.transcription_api_base_url.value == "http://from-env:2"
    assert from_env.transcription_api_base_url.origin == "TRANSCRIPTION_API_BASE_URL"
    assert from_env.transcription_timeout_seconds.value == 7.5
    assert from_env.transcription_api_token.value == "s3cret"
    assert from_env.transcription_api_token.display_value() == "set"
    assert "s3cret" not in repr(from_env.transcription_api_token.display_value())


def test_artifacts_are_off_unless_explicitly_enabled(tmp_path: Path) -> None:
    assert load_config(_env(tmp_path)).artifacts_enabled.value is False

    on = load_config(_env(tmp_path, JAKE_TOOLS_ARTIFACTS_ENABLED="true"))
    assert on.artifacts_enabled.value is True and on.artifacts_enabled.source == "env"
    off = load_config(_env(tmp_path, JAKE_TOOLS_ARTIFACTS_ENABLED="false"))
    assert (
        off.artifacts_enabled.value is False and off.artifacts_enabled.source == "env"
    )


def test_the_account_context_label_is_optional_and_unset_by_default(
    tmp_path: Path,
) -> None:
    assert load_config(_env(tmp_path)).claude_account_context.value is None
    blank = load_config({**_env(tmp_path), "JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT": " "})
    assert blank.claude_account_context.value is None


def test_the_account_context_label_comes_from_the_environment(
    tmp_path: Path,
) -> None:
    env = {**_env(tmp_path), "JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT": "work.max:1"}
    config = load_config(env)
    assert config.claude_account_context.value == "work.max:1"
    assert config.claude_account_context.origin == "JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT"
    assert not config.claude_account_context.secret


@pytest.mark.parametrize("label", ["has space", "a/b", "x" * 65, "über", "a@b.com "])
def test_an_unsafe_account_context_label_is_refused(tmp_path: Path, label: str) -> None:
    env = {**_env(tmp_path), "JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT": label}
    with pytest.raises(ConfigError) as excinfo:
        load_config(env)
    assert "claude.account_context" in str(excinfo.value)


# --- claude.max_live_tasks -----------------------------------------------------


def test_max_live_tasks_defaults_to_two(tmp_path: Path) -> None:
    config = load_config(_env(tmp_path))

    assert DEFAULT_CLAUDE_MAX_LIVE_TASKS == 2
    assert config.claude_max_live_tasks.value == 2
    assert config.claude_max_live_tasks.source == "default"


def test_max_live_tasks_resolves_flag_over_env_over_default(tmp_path: Path) -> None:
    from_env = load_config(_env(tmp_path, JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS="3"))
    from_flag = load_config(
        _env(tmp_path, JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS="3"),
        overrides={"claude.max_live_tasks": 4},
    )

    assert from_env.claude_max_live_tasks.value == 3
    assert from_env.claude_max_live_tasks.origin == "JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS"
    assert (
        from_flag.claude_max_live_tasks.value,
        from_flag.claude_max_live_tasks.source,
    ) == (4, "flag")
    assert from_env.claude_max_live_tasks in from_env.settings()  # doctor prints it


@pytest.mark.parametrize("raw", ["0", "-1", "two", "2.5", "", "true", " "])
def test_max_live_tasks_rejects_anything_but_a_positive_integer_from_env(
    tmp_path: Path, raw: str
) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(_env(tmp_path, JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS=raw))

    assert "claude.max_live_tasks" in str(excinfo.value)
    assert "JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS" in str(excinfo.value)
    assert "positive integer" in str(excinfo.value)


def test_microsoft_settings_are_explicit_and_secrets_are_environment_only(
    tmp_path: Path,
) -> None:
    unset = load_config(_env(tmp_path))
    assert unset.microsoft_tenant_id.value is None
    assert unset.microsoft_account.value is None  # no default account, ever
    assert unset.microsoft_authority_url.value == "https://login.microsoftonline.com"
    assert unset.newsletters_graph_root.value == "https://graph.microsoft.com/v1.0"
    assert unset.microsoft_refresh_token.value is None
    assert unset.microsoft_client_secret.value is None

    secret_file = tmp_path / "rt"
    secret_file.write_text("from-file\n")
    from_env = load_config(
        _env(
            tmp_path,
            JAKE_TOOLS_MICROSOFT_ACCOUNT="env@example.test",
            MICROSOFT_REFRESH_TOKEN_FILE=str(secret_file),
            MICROSOFT_CLIENT_SECRET="s3cret",
        )
    )
    assert from_env.microsoft_account.value == "env@example.test"
    assert from_env.microsoft_refresh_token.value == "from-file"
    assert from_env.microsoft_refresh_token.display_value() == "set"
    assert from_env.microsoft_client_secret.display_value() == "set"
    assert "from-file" not in repr(from_env.microsoft_refresh_token.display_value())
