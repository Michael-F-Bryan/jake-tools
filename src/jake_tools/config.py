"""Configuration for the ``jake_tools.mcp`` entrypoint.

Paths follow the XDG Base Directory specification on every platform,
including macOS, with the spec's fallbacks when a variable is unset or not
absolute. Every setting is resolved with one precedence, highest first:

1. a CLI flag (``overrides`` here; the command maps its flags onto them),
2. an environment variable,
3. ``$XDG_CONFIG_HOME/jake-tools/config.toml``,
4. each ``$XDG_CONFIG_DIRS`` entry's ``jake-tools/config.toml``, in order,
5. the built-in default.

Secrets are environment-only and are never read from ``config.toml``. Each
also accepts the Docker/systemd ``*_FILE`` convention, with the direct
variable winning if both are set. Nothing here reads ``.env`` files: the
process environment is the only environment.

Every resolved value remembers where it came from so ``doctor`` can print the
source next to the value. :class:`Config` is immutable and built once per
process by :func:`load_config`.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from .claude import DEFAULT_MODEL, EffortLevel
from .clockify import CLOCKIFY_API_ROOT

APP_DIRNAME = "jake-tools"
CONFIG_FILENAME = "config.toml"

Source = Literal["flag", "env", "config", "default"]
"""Where a setting's value was resolved from.

``config`` covers both ``$XDG_CONFIG_HOME`` and ``$XDG_CONFIG_DIRS`` files;
:attr:`Resolved.origin` names the actual file.
"""

SECRET_VARIABLES: tuple[str, ...] = (
    "CLOCKIFY_API_KEY",
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
)
"""Every environment-only secret, by variable name.

The single list: :func:`load_config` reads each (and its ``_FILE`` variant),
and the delegated-task worker is spawned with all of them removed from its
environment so a Bash-granted task can never ``env`` them into a transcript.
"""

DEFAULT_JIRA_PROJECT = "SF"
DEFAULT_CLOCKIFY_CLIENT = "Sunfish Robotics"
DEFAULT_CLAUDE_MAX_TURNS = 20
DEFAULT_CLAUDE_TIMEOUT_SECONDS = 900.0
DEFAULT_CLAUDE_MAX_BUDGET_USD = 5.0


class ConfigError(RuntimeError):
    """A config file or environment value could not be used.

    The message names the setting and where it came from, never a value.
    """


class XdgPaths(BaseModel):
    """The XDG base directories, after the spec's fallbacks."""

    model_config = ConfigDict(frozen=True)

    home: Path
    config_home: Path
    state_home: Path
    cache_home: Path
    config_dirs: tuple[Path, ...]

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> XdgPaths:
        home = _home(environ)
        return cls(
            home=home,
            config_home=_xdg_dir(environ, "XDG_CONFIG_HOME", home / ".config"),
            state_home=_xdg_dir(environ, "XDG_STATE_HOME", home / ".local" / "state"),
            cache_home=_xdg_dir(environ, "XDG_CACHE_HOME", home / ".cache"),
            config_dirs=_xdg_dirs(environ),
        )

    @property
    def app_config_dir(self) -> Path:
        return self.config_home / APP_DIRNAME

    @property
    def app_state_dir(self) -> Path:
        return self.state_home / APP_DIRNAME

    @property
    def app_cache_dir(self) -> Path:
        return self.cache_home / APP_DIRNAME

    @property
    def config_files(self) -> tuple[Path, ...]:
        """Candidate config files, highest precedence first."""
        return (
            self.app_config_dir / CONFIG_FILENAME,
            *(
                directory / APP_DIRNAME / CONFIG_FILENAME
                for directory in self.config_dirs
            ),
        )


class Resolved[T](BaseModel):
    """One setting's value plus where it was resolved from.

    ``origin`` is the flag name, the environment variable name, or the config
    file path, depending on ``source``; it is ``None`` for defaults unless the
    default itself derived from the environment (e.g. ``HERMES_HOME``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    value: T
    source: Source
    origin: str | None = None
    secret: bool = False

    def display_value(self) -> str:
        """The value as ``doctor`` may print it: secrets are never shown."""
        if self.secret:
            return "set" if self.value is not None else "unset"
        if self.value is None:
            return "unset"
        if isinstance(self.value, Path):
            return str(self.value.expanduser().absolute())
        if isinstance(self.value, tuple | list):
            return ", ".join(str(item) for item in self.value) or "(empty)"
        return str(self.value)

    def display_source(self) -> str:
        return self.source if self.origin is None else f"{self.source} ({self.origin})"


class ConfigFile(BaseModel):
    """One candidate ``config.toml`` and what loading it produced."""

    model_config = ConfigDict(frozen=True)

    path: Path
    exists: bool
    data: dict[str, Any] = {}


class Config(BaseModel):
    """Everything the MCP entrypoint can be configured with, fully resolved."""

    model_config = ConfigDict(frozen=True)

    xdg: XdgPaths
    config_files: tuple[ConfigFile, ...]
    hermes_home: Path | None

    clockify_api_key: Resolved[str | None]
    jira_base_url: Resolved[str | None]
    jira_email: Resolved[str | None]
    jira_api_token: Resolved[str | None]

    jira_project: Resolved[str]
    clockify_client: Resolved[str]
    clockify_api_base_url: Resolved[str]
    clockify_workspaces: Resolved[tuple[str, ...]]
    session_store_path: Resolved[Path | None]
    runs_dir: Resolved[Path]

    claude_model: Resolved[str]
    claude_effort: Resolved[EffortLevel | None]
    claude_max_turns: Resolved[int]
    claude_timeout_seconds: Resolved[float]
    claude_max_budget_usd: Resolved[float]

    def settings(self) -> tuple[Resolved[Any], ...]:
        """Every setting, in a stable order, for ``doctor`` to print."""
        return (
            self.clockify_api_key,
            self.jira_base_url,
            self.jira_email,
            self.jira_api_token,
            self.jira_project,
            self.clockify_client,
            self.clockify_api_base_url,
            self.clockify_workspaces,
            self.session_store_path,
            self.runs_dir,
            self.claude_model,
            self.claude_effort,
            self.claude_max_turns,
            self.claude_timeout_seconds,
            self.claude_max_budget_usd,
        )

    @property
    def has_clockify_credentials(self) -> bool:
        return self.clockify_api_key.value is not None

    @property
    def missing_jira_variables(self) -> tuple[str, ...]:
        """Which Jira environment variables are unset, in a stable order."""
        return tuple(
            name
            for name, setting in (
                ("JIRA_BASE_URL", self.jira_base_url),
                ("JIRA_EMAIL", self.jira_email),
                ("JIRA_API_TOKEN", self.jira_api_token),
            )
            if setting.value is None
        )


@dataclass(frozen=True)
class _Spec[T]:
    """How one non-secret setting is looked up."""

    name: str
    env: str | None
    table: str | None
    key: str | None
    adapter: TypeAdapter[T]


_STR = TypeAdapter(str)
_OPTIONAL_STR = TypeAdapter(str | None)
_STR_TUPLE = TypeAdapter(tuple[str, ...])
_PATH = TypeAdapter(Path)
_INT = TypeAdapter(int)
_FLOAT = TypeAdapter(float)
_EFFORT = TypeAdapter(EffortLevel | None)


def load_config(
    environ: Mapping[str, str] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Resolve every setting from ``environ`` (default: the process environment).

    ``overrides`` maps setting names (``"claude.runs_dir"``) to values a CLI
    flag supplied; they win over everything. Raises :class:`ConfigError` when
    a config file cannot be parsed or a value has the wrong type; a missing
    file or variable is never an error.
    """
    env = dict(os.environ if environ is None else environ)
    flags = dict(overrides or {})
    xdg = XdgPaths.from_environ(env)
    files = tuple(_read_config_file(path) for path in xdg.config_files)
    hermes_home = _optional_path(env.get("HERMES_HOME"))

    def resolve[T](
        spec: _Spec[T], default: T, default_origin: str | None = None
    ) -> Resolved[T]:
        return _resolve(spec, env, files, flags, default, default_origin)

    runs_dir = resolve(
        _Spec("claude.runs_dir", "JAKE_TOOLS_RUNS_DIR", "claude", "runs_dir", _PATH),
        xdg.app_state_dir / "runs",
    )
    session_store = resolve(
        _Spec(
            "session_store.path",
            "JAKE_TOOLS_SESSION_STORE",
            "session_store",
            "path",
            TypeAdapter(Path | None),
        ),
        None if hermes_home is None else hermes_home / "state.db",
        None if hermes_home is None else "HERMES_HOME",
    )

    secrets = {name: _secret(name, env) for name in SECRET_VARIABLES}
    return Config(
        xdg=xdg,
        config_files=files,
        hermes_home=hermes_home,
        clockify_api_key=secrets["CLOCKIFY_API_KEY"],
        jira_base_url=secrets["JIRA_BASE_URL"],
        jira_email=secrets["JIRA_EMAIL"],
        jira_api_token=secrets["JIRA_API_TOKEN"],
        jira_project=resolve(
            _Spec(
                "clockify.jira_project",
                "JAKE_TOOLS_JIRA_PROJECT",
                "clockify",
                "jira_project",
                _STR,
            ),
            DEFAULT_JIRA_PROJECT,
        ),
        clockify_client=resolve(
            _Spec(
                "clockify.client",
                "JAKE_TOOLS_CLOCKIFY_CLIENT",
                "clockify",
                "client",
                _STR,
            ),
            DEFAULT_CLOCKIFY_CLIENT,
        ),
        clockify_api_base_url=resolve(
            _Spec(
                "clockify.api_base_url",
                "CLOCKIFY_API_BASE_URL",
                "clockify",
                "api_base_url",
                _STR,
            ),
            CLOCKIFY_API_ROOT,
        ),
        clockify_workspaces=resolve(
            _Spec(
                "clockify.workspaces",
                "JAKE_TOOLS_CLOCKIFY_WORKSPACES",
                "clockify",
                "workspaces",
                _STR_TUPLE,
            ),
            (),
        ),
        session_store_path=session_store,
        runs_dir=runs_dir,
        claude_model=resolve(
            _Spec("claude.model", None, "claude", "model", _STR), DEFAULT_MODEL
        ),
        claude_effort=resolve(
            _Spec("claude.effort", None, "claude", "effort", _EFFORT), None
        ),
        claude_max_turns=resolve(
            _Spec("claude.max_turns", None, "claude", "max_turns", _INT),
            DEFAULT_CLAUDE_MAX_TURNS,
        ),
        claude_timeout_seconds=resolve(
            _Spec("claude.timeout_seconds", None, "claude", "timeout_seconds", _FLOAT),
            DEFAULT_CLAUDE_TIMEOUT_SECONDS,
        ),
        claude_max_budget_usd=resolve(
            _Spec("claude.max_budget_usd", None, "claude", "max_budget_usd", _FLOAT),
            DEFAULT_CLAUDE_MAX_BUDGET_USD,
        ),
    )


def _resolve[T](
    spec: _Spec[T],
    env: Mapping[str, str],
    files: tuple[ConfigFile, ...],
    flags: Mapping[str, Any],
    default: T,
    default_origin: str | None,
) -> Resolved[T]:
    if spec.name in flags and flags[spec.name] is not None:
        return Resolved(
            name=spec.name,
            value=_coerce(spec, flags[spec.name], f"flag {spec.name}"),
            source="flag",
            origin=spec.name,
        )
    if spec.env is not None and spec.env in env:
        raw: Any = env[spec.env]
        if spec.adapter is _STR_TUPLE:
            raw = tuple(item.strip() for item in raw.split(",") if item.strip())
        return Resolved(
            name=spec.name,
            value=_coerce(spec, raw, f"environment variable {spec.env}"),
            source="env",
            origin=spec.env,
        )
    if spec.table is not None and spec.key is not None:
        for file in files:
            table = file.data.get(spec.table)
            if isinstance(table, dict) and spec.key in table:
                return Resolved(
                    name=spec.name,
                    value=_coerce(spec, table[spec.key], f"{file.path}"),
                    source="config",
                    origin=str(file.path),
                )
    return Resolved(
        name=spec.name, value=default, source="default", origin=default_origin
    )


def _coerce[T](spec: _Spec[T], raw: Any, where: str) -> T:
    try:
        value = spec.adapter.validate_python(raw)
    except ValidationError as exc:
        expected = _describe_expected(spec)
        raise ConfigError(
            f"{spec.name} from {where} is not {expected}: {exc.errors()[0]['msg']}"
        ) from exc
    if isinstance(value, Path):
        return value.expanduser()  # type: ignore[return-value]
    return value


def _describe_expected(spec: _Spec[Any]) -> str:
    if spec.adapter is _STR_TUPLE:
        return "a list of strings"
    if spec.adapter is _EFFORT:
        return f"one of {', '.join(get_args(EffortLevel))}"
    if spec.adapter is _INT:
        return "an integer"
    if spec.adapter is _FLOAT:
        return "a number"
    if spec.adapter is _PATH:
        return "a path"
    return "a string"


def _secret(name: str, env: Mapping[str, str]) -> Resolved[str | None]:
    """Environment-only, with ``<NAME>_FILE`` as the Docker/systemd fallback."""
    if name in env:
        return Resolved(
            name=name,
            value=_strip_newline(env[name]),
            source="env",
            origin=name,
            secret=True,
        )
    file_variable = f"{name}_FILE"
    if file_variable in env:
        path = Path(env[file_variable]).expanduser()
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"{file_variable} points at a file that cannot be read: {exc.strerror}"
            ) from exc
        return Resolved(
            name=name,
            value=_strip_newline(content),
            source="env",
            origin=file_variable,
            secret=True,
        )
    return Resolved(name=name, value=None, source="default", secret=True)


def _strip_newline(value: str) -> str:
    return value.removesuffix("\n").removesuffix("\r")


def _read_config_file(path: Path) -> ConfigFile:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ConfigFile(path=path, exists=False)
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"cannot parse {path}: {exc}") from exc
    return ConfigFile(path=path, exists=True, data=data)


def _home(environ: Mapping[str, str]) -> Path:
    raw = environ.get("HOME")
    if raw and os.path.isabs(raw):
        return Path(raw)
    return Path.home()


def _xdg_dir(environ: Mapping[str, str], variable: str, fallback: Path) -> Path:
    raw = environ.get(variable)
    if raw and os.path.isabs(raw):
        return Path(raw)
    return fallback


def _xdg_dirs(environ: Mapping[str, str]) -> tuple[Path, ...]:
    raw = environ.get("XDG_CONFIG_DIRS")
    if not raw:
        return (Path("/etc/xdg"),)
    dirs = tuple(Path(item) for item in raw.split(os.pathsep) if os.path.isabs(item))
    return dirs or (Path("/etc/xdg"),)


def _optional_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    return Path(raw).expanduser()
