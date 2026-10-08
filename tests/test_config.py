from __future__ import annotations

from pathlib import Path

import pytest

from jake_tools.config import (
    DEFAULT_CLAUDE_MAX_BUDGET_USD,
    DEFAULT_CLAUDE_MAX_TURNS,
    DEFAULT_CLAUDE_TIMEOUT_SECONDS,
    DEFAULT_CLOCKIFY_CLIENT,
    DEFAULT_JIRA_PROJECT,
    ConfigError,
    XdgPaths,
    load_config,
)


def _env(home: Path, **extra: str) -> dict[str, str]:
    return {"HOME": str(home), **extra}


def test_xdg_defaults_follow_the_spec_fallbacks(tmp_path: Path) -> None:
    xdg = XdgPaths.from_environ(_env(tmp_path))

    assert xdg.config_home == tmp_path / ".config"
    assert xdg.state_home == tmp_path / ".local" / "state"
    assert xdg.cache_home == tmp_path / ".cache"
    assert xdg.config_dirs == (Path("/etc/xdg"),)
    assert xdg.config_files == (
        tmp_path / ".config" / "jake-tools" / "config.toml",
        Path("/etc/xdg/jake-tools/config.toml"),
    )


def test_xdg_ignores_relative_values_and_honours_absolute_ones(tmp_path: Path) -> None:
    xdg = XdgPaths.from_environ(
        _env(
            tmp_path,
            XDG_CONFIG_HOME="relative/config",
            XDG_STATE_HOME=str(tmp_path / "state"),
            XDG_CONFIG_DIRS=f"relative:{tmp_path / 'xdg-a'}:{tmp_path / 'xdg-b'}",
        )
    )

    assert xdg.config_home == tmp_path / ".config"
    assert xdg.state_home == tmp_path / "state"
    assert xdg.config_dirs == (tmp_path / "xdg-a", tmp_path / "xdg-b")


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
    assert all(setting.source == "default" for setting in config.settings())
    assert not config.has_clockify_credentials
    assert config.missing_jira_variables == (
        "JIRA_BASE_URL",
        "JIRA_EMAIL",
        "JIRA_API_TOKEN",
    )


def test_precedence_flag_env_config_home_config_dirs_default(tmp_path: Path) -> None:
    config_home = tmp_path / ".config" / "jake-tools"
    config_home.mkdir(parents=True)
    (config_home / "config.toml").write_text(
        '[clockify]\njira_project = "HOME"\nclient = "Home Client"\n'
        'workspaces = ["ws-home"]\n[claude]\nmax_turns = 7\n'
    )
    system_dir = tmp_path / "etc-xdg" / "jake-tools"
    system_dir.mkdir(parents=True)
    (system_dir / "config.toml").write_text(
        '[clockify]\njira_project = "SYS"\nclient = "System Client"\n'
        "[claude]\nmax_turns = 3\ntimeout_seconds = 120\n[session_store]\n"
        'path = "/srv/hermes/state.db"\n'
    )

    config = load_config(
        _env(
            tmp_path,
            XDG_CONFIG_DIRS=str(tmp_path / "etc-xdg"),
            JAKE_TOOLS_CLOCKIFY_CLIENT="Env Client",
        ),
        overrides={"clockify.jira_project": "FLAG"},
    )

    assert (config.jira_project.value, config.jira_project.source) == ("FLAG", "flag")
    assert (config.clockify_client.value, config.clockify_client.source) == (
        "Env Client",
        "env",
    )
    assert config.clockify_client.origin == "JAKE_TOOLS_CLOCKIFY_CLIENT"
    assert config.clockify_workspaces.value == ("ws-home",)
    assert config.claude_max_turns.value == 7
    assert config.claude_max_turns.source == "config"
    assert config.claude_max_turns.origin == str(config_home / "config.toml")
    # Only the system file sets these, so they come from the lowest-precedence file.
    assert config.claude_timeout_seconds.value == 120.0
    assert config.claude_timeout_seconds.origin == str(system_dir / "config.toml")
    assert config.session_store_path.value == Path("/srv/hermes/state.db")
    assert config.claude_max_budget_usd.source == "default"


def test_workspace_allowlist_from_environment_is_comma_separated(
    tmp_path: Path,
) -> None:
    config = load_config(
        _env(tmp_path, JAKE_TOOLS_CLOCKIFY_WORKSPACES="ws-a, ws-b,,ws-c ")
    )

    assert config.clockify_workspaces.value == ("ws-a", "ws-b", "ws-c")
    assert config.clockify_workspaces.source == "env"


def test_secrets_come_from_the_environment_only(tmp_path: Path) -> None:
    config_home = tmp_path / ".config" / "jake-tools"
    config_home.mkdir(parents=True)
    (config_home / "config.toml").write_text(
        '[clockify]\napi_key = "should-not-be-read"\n'
    )
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


def test_unparsable_config_file_is_a_config_error(tmp_path: Path) -> None:
    config_home = tmp_path / ".config" / "jake-tools"
    config_home.mkdir(parents=True)
    (config_home / "config.toml").write_text("[clockify\n")

    with pytest.raises(ConfigError) as excinfo:
        load_config(_env(tmp_path))

    assert str(config_home / "config.toml") in str(excinfo.value)


def test_wrong_type_names_the_setting_and_where_it_came_from(tmp_path: Path) -> None:
    config_home = tmp_path / ".config" / "jake-tools"
    config_home.mkdir(parents=True)
    (config_home / "config.toml").write_text('[claude]\nmax_turns = "lots"\n')

    with pytest.raises(ConfigError) as excinfo:
        load_config(_env(tmp_path))

    message = str(excinfo.value)
    assert "claude.max_turns" in message
    assert str(config_home / "config.toml") in message
    assert "integer" in message


def test_invalid_effort_lists_the_choices(tmp_path: Path) -> None:
    config_home = tmp_path / ".config" / "jake-tools"
    config_home.mkdir(parents=True)
    (config_home / "config.toml").write_text('[claude]\neffort = "turbo"\n')

    with pytest.raises(ConfigError) as excinfo:
        load_config(_env(tmp_path))

    assert "low, medium, high" in str(excinfo.value)


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
