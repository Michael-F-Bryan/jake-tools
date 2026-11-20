"""Configuration for every ``jake_tools`` entrypoint.

Configuration is explicit: CLI flags and process environment variables, and
nothing else. There is no ``config.toml`` (searched or otherwise) and no
``.env`` file: the process environment is the only environment, and nothing in
the working directory or its ancestors can change a setting. Every setting is
resolved with one precedence, highest first:

1. a CLI flag (``overrides`` here; the command maps its flags onto them),
2. an environment variable (named in the setting's :class:`_Spec`),
3. the built-in default.

Paths follow the XDG Base Directory specification on every platform,
including macOS, with the spec's fallbacks when a variable is unset or not
absolute; only the state and cache directories are used (run state, caches),
never a config directory.

Secrets are environment-only. Each also accepts the Docker/systemd ``*_FILE``
convention, with the direct variable winning if both are set.

Every resolved value remembers where it came from so ``doctor`` can print the
source next to the value. :class:`Config` is immutable and built once per
process by :func:`load_config`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)

from .claude import DEFAULT_MODEL, EffortLevel
from .clockify import CLOCKIFY_API_ROOT
from .hindsight import (
    HindsightAccess,
    HindsightMemory,
    OptionalApiUrl,
    OptionalBankId,
)

APP_DIRNAME = "jake-tools"

Source = Literal["flag", "env", "default"]
"""Where a setting's value was resolved from."""

SECRET_VARIABLES: tuple[str, ...] = (
    "CLOCKIFY_API_KEY",
    "JIRA_BASE_URL",
    "JIRA_EMAIL",
    "JIRA_API_TOKEN",
    "TRANSCRIPTION_API_TOKEN",
    "MICROSOFT_REFRESH_TOKEN",
    "MICROSOFT_CLIENT_SECRET",
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
DEFAULT_CLAUDE_MAX_LIVE_TASKS = 2
"""Delegated tasks one admitting process lets run at once, unless configured.

Jake policy, not an MCP or Claude constraint. The default is the original
fixed two. See :class:`~jake_tools.claude_runs.control.CapacityExceededError`
and ``claude_runs/control.py`` for the exact (per-admitting-process) semantics.
"""
DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS = 1800.0
"""How long one transcription call may take, including inference.

The service runs one job at a time with a bounded queue, so a long recording
behind other work can legitimately take many minutes. A timeout is reported,
never retried: the service may still be processing the original request.
"""


DEFAULT_MICROSOFT_AUTHORITY_URL = "https://login.microsoftonline.com"
DEFAULT_GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
DEFAULT_SHAREPOINT_ROOT = "https://csuses.sharepoint.com/sites/leadershipteam/_api"
DEFAULT_FILES_SHAREPOINT_HOSTS = ("csuses.sharepoint.com",)
"""Where the newsletter operations talk by default. Overridable only so a
fixture upstream can stand in; a non-HTTPS value is refused unless it is a
loopback address (see ``microsoft_auth.require_secure_url``)."""


class ConfigError(RuntimeError):
    """A flag or environment value could not be used.

    The message names the setting and where it came from, never a value.
    """


class XdgPaths(BaseModel):
    """The XDG base directories, after the spec's fallbacks."""

    model_config = ConfigDict(frozen=True)

    home: Path
    state_home: Path
    cache_home: Path

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> XdgPaths:
        home = _home(environ)
        return cls(
            home=home,
            state_home=_xdg_dir(environ, "XDG_STATE_HOME", home / ".local" / "state"),
            cache_home=_xdg_dir(environ, "XDG_CACHE_HOME", home / ".cache"),
        )

    @property
    def app_state_dir(self) -> Path:
        return self.state_home / APP_DIRNAME

    @property
    def app_cache_dir(self) -> Path:
        return self.cache_home / APP_DIRNAME


class Resolved[T](BaseModel):
    """One setting's value plus where it was resolved from.

    ``origin`` is the flag's setting name or the environment variable name,
    depending on ``source``; it is ``None`` for defaults unless the default
    itself derived from the environment (e.g. ``HERMES_HOME``).
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


class Config(BaseModel):
    """Everything jake-tools can be configured with, fully resolved."""

    model_config = ConfigDict(frozen=True)

    xdg: XdgPaths
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
    claude_max_live_tasks: Resolved[int]
    """The most delegated tasks *this process* admits while they are live in
    the shared runs directory (``JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS``, a positive
    integer). Every process applies its own value to the one shared live
    count, so processes configured differently do not share a ceiling; see
    ``claude_runs/control.py``."""
    claude_account_context: Resolved[str | None]
    """An operator-declared, non-secret label for the Claude account context
    this installation runs under. Unset (the default) means no cooldown
    sharing. Nothing infers or verifies it: equal labels are the operator's
    statement that the runs share one quota."""

    transcription_api_base_url: Resolved[str | None]
    transcription_api_token: Resolved[str | None]
    transcription_timeout_seconds: Resolved[float]

    microsoft_tenant_id: Resolved[str | None]
    microsoft_client_id: Resolved[str | None]
    microsoft_account: Resolved[str | None]
    """The sign-in name (UPN) the delegated grant MUST belong to. Explicit on
    purpose: a grant for any other account is refused, never used."""
    microsoft_account_object_id: Resolved[str | None]
    """Optional immutable pin (the account's ``oid``): sign-in names can change."""
    microsoft_authority_url: Resolved[str]
    microsoft_refresh_token: Resolved[str | None]
    microsoft_client_secret: Resolved[str | None]
    newsletters_graph_root: Resolved[str]
    newsletters_sharepoint_root: Resolved[str]
    files_graph_root: Resolved[str]
    """Where the ``sharepoint_*`` file tools reach Microsoft Graph. Overridable
    only so a fixture upstream can stand in (HTTPS, or HTTP to loopback)."""
    files_sharepoint_hosts: Resolved[tuple[str, ...]]
    """The SharePoint hosts ``sharepoint_resolve`` accepts a link for. A link
    on any other host (another tenant's, a guest-shared one) is refused."""

    artifacts_enabled: Resolved[bool]
    """Whether ``artifact_publish`` and ``artifact_read`` are exposed. Off by
    default: they depend on an undocumented Claude Code mechanism."""

    hindsight_api_url: Resolved[str | None]
    hindsight_bank_id: Resolved[str | None]
    hindsight_access: Resolved[HindsightAccess]
    """The Hindsight server, bank and access that delegated workers share with
    the agent that started them (see :mod:`jake_tools.hindsight`). Unset
    (the default) means workers get no memory. Both the URL and the bank are
    needed (``doctor`` fails when only one is set), and there is no fallback
    bank: a worker is never given a default or a global one."""

    @property
    def hindsight(self) -> HindsightMemory | None:
        """The memory connection workers are launched with, or ``None``."""
        url, bank = self.hindsight_api_url.value, self.hindsight_bank_id.value
        if url is None or bank is None:
            return None
        return HindsightMemory(
            api_url=url, bank_id=bank, access=self.hindsight_access.value
        )

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
            self.claude_max_live_tasks,
            self.claude_account_context,
            self.transcription_api_base_url,
            self.transcription_api_token,
            self.transcription_timeout_seconds,
            self.microsoft_tenant_id,
            self.microsoft_client_id,
            self.microsoft_account,
            self.microsoft_account_object_id,
            self.microsoft_authority_url,
            self.microsoft_refresh_token,
            self.microsoft_client_secret,
            self.newsletters_graph_root,
            self.newsletters_sharepoint_root,
            self.files_graph_root,
            self.files_sharepoint_hosts,
            self.artifacts_enabled,
            self.hindsight_api_url,
            self.hindsight_bank_id,
            self.hindsight_access,
        )

    @property
    def has_clockify_credentials(self) -> bool:
        return self.clockify_api_key.value is not None

    @property
    def has_transcription_endpoint(self) -> bool:
        """Whether the ``transcribe`` tool is exposed at all.

        The endpoint is deliberate configuration; the token is a secret that
        may be missing, in which case the tool is exposed and reports
        ``missing_credentials`` when called.
        """
        return self.transcription_api_base_url.value is not None

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
    env: str
    adapter: TypeAdapter[T]


_STR = TypeAdapter(str)
_OPTIONAL_STR = TypeAdapter(str | None)
ACCOUNT_CONTEXT_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"
"""What an operator-declared ``claude.account_context`` label may look like."""


def _blank_is_unset(value: Any) -> Any:
    return None if isinstance(value, str) and not value.strip() else value


AccountContext = Annotated[
    Annotated[str, StringConstraints(pattern=ACCOUNT_CONTEXT_PATTERN)] | None,
    BeforeValidator(_blank_is_unset),
]
_ACCOUNT_CONTEXT = TypeAdapter(AccountContext)
_HINDSIGHT_URL = TypeAdapter(OptionalApiUrl)
_HINDSIGHT_BANK = TypeAdapter(OptionalBankId)
_HINDSIGHT_ACCESS = TypeAdapter(HindsightAccess)
_STR_TUPLE = TypeAdapter(tuple[str, ...])
_PATH = TypeAdapter(Path)
_INT = TypeAdapter(int)


def _not_a_bool(value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("a boolean is not a count")
    return value


_POSITIVE_INT = TypeAdapter(Annotated[int, Field(ge=1), BeforeValidator(_not_a_bool)])
_FLOAT = TypeAdapter(float)
_BOOL = TypeAdapter(bool)
_EFFORT = TypeAdapter(EffortLevel | None)


def load_config(
    environ: Mapping[str, str] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Resolve every setting from ``environ`` (default: the process environment).

    ``overrides`` maps setting names (``"claude.runs_dir"``) to values a CLI
    flag supplied; they win over everything. Raises :class:`ConfigError` when
    a value has the wrong type or a ``*_FILE`` secret cannot be read; a
    missing variable is never an error. No file other than a ``*_FILE``
    secret is ever read.
    """
    env = dict(os.environ if environ is None else environ)
    flags = dict(overrides or {})
    xdg = XdgPaths.from_environ(env)
    hermes_home = _optional_path(env.get("HERMES_HOME"))

    def resolve[T](
        spec: _Spec[T], default: T, default_origin: str | None = None
    ) -> Resolved[T]:
        return _resolve(spec, env, flags, default, default_origin)

    runs_dir = resolve(
        _Spec("claude.runs_dir", "JAKE_TOOLS_RUNS_DIR", _PATH),
        xdg.app_state_dir / "runs",
    )
    session_store = resolve(
        _Spec(
            "session_store.path",
            "JAKE_TOOLS_SESSION_STORE",
            TypeAdapter(Path | None),
        ),
        None if hermes_home is None else hermes_home / "state.db",
        None if hermes_home is None else "HERMES_HOME",
    )

    secrets = {name: _secret(name, env) for name in SECRET_VARIABLES}
    config = Config(
        xdg=xdg,
        hermes_home=hermes_home,
        clockify_api_key=secrets["CLOCKIFY_API_KEY"],
        jira_base_url=secrets["JIRA_BASE_URL"],
        jira_email=secrets["JIRA_EMAIL"],
        jira_api_token=secrets["JIRA_API_TOKEN"],
        jira_project=resolve(
            _Spec(
                "clockify.jira_project",
                "JAKE_TOOLS_JIRA_PROJECT",
                _STR,
            ),
            DEFAULT_JIRA_PROJECT,
        ),
        clockify_client=resolve(
            _Spec(
                "clockify.client",
                "JAKE_TOOLS_CLOCKIFY_CLIENT",
                _STR,
            ),
            DEFAULT_CLOCKIFY_CLIENT,
        ),
        clockify_api_base_url=resolve(
            _Spec(
                "clockify.api_base_url",
                "CLOCKIFY_API_BASE_URL",
                _STR,
            ),
            CLOCKIFY_API_ROOT,
        ),
        clockify_workspaces=resolve(
            _Spec(
                "clockify.workspaces",
                "JAKE_TOOLS_CLOCKIFY_WORKSPACES",
                _STR_TUPLE,
            ),
            (),
        ),
        session_store_path=session_store,
        runs_dir=runs_dir,
        claude_model=resolve(
            _Spec("claude.model", "JAKE_TOOLS_CLAUDE_MODEL", _STR), DEFAULT_MODEL
        ),
        claude_effort=resolve(
            _Spec("claude.effort", "JAKE_TOOLS_CLAUDE_EFFORT", _EFFORT), None
        ),
        claude_max_turns=resolve(
            _Spec("claude.max_turns", "JAKE_TOOLS_CLAUDE_MAX_TURNS", _INT),
            DEFAULT_CLAUDE_MAX_TURNS,
        ),
        claude_timeout_seconds=resolve(
            _Spec(
                "claude.timeout_seconds", "JAKE_TOOLS_CLAUDE_TIMEOUT_SECONDS", _FLOAT
            ),
            DEFAULT_CLAUDE_TIMEOUT_SECONDS,
        ),
        claude_max_budget_usd=resolve(
            _Spec("claude.max_budget_usd", "JAKE_TOOLS_CLAUDE_MAX_BUDGET_USD", _FLOAT),
            DEFAULT_CLAUDE_MAX_BUDGET_USD,
        ),
        claude_max_live_tasks=resolve(
            _Spec(
                "claude.max_live_tasks",
                "JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS",
                _POSITIVE_INT,
            ),
            DEFAULT_CLAUDE_MAX_LIVE_TASKS,
        ),
        claude_account_context=resolve(
            _Spec(
                "claude.account_context",
                "JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT",
                _ACCOUNT_CONTEXT,
            ),
            None,
        ),
        transcription_api_base_url=resolve(
            _Spec(
                "transcription.api_base_url",
                "TRANSCRIPTION_API_BASE_URL",
                _OPTIONAL_STR,
            ),
            None,
        ),
        transcription_api_token=secrets["TRANSCRIPTION_API_TOKEN"],
        transcription_timeout_seconds=resolve(
            _Spec(
                "transcription.timeout_seconds",
                "JAKE_TOOLS_TRANSCRIPTION_TIMEOUT_SECONDS",
                _FLOAT,
            ),
            DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS,
        ),
        microsoft_tenant_id=resolve(
            _Spec(
                "microsoft.tenant_id",
                "JAKE_TOOLS_MICROSOFT_TENANT_ID",
                _OPTIONAL_STR,
            ),
            None,
        ),
        microsoft_client_id=resolve(
            _Spec(
                "microsoft.client_id",
                "JAKE_TOOLS_MICROSOFT_CLIENT_ID",
                _OPTIONAL_STR,
            ),
            None,
        ),
        microsoft_account=resolve(
            _Spec(
                "microsoft.account",
                "JAKE_TOOLS_MICROSOFT_ACCOUNT",
                _OPTIONAL_STR,
            ),
            None,
        ),
        microsoft_account_object_id=resolve(
            _Spec(
                "microsoft.account_object_id",
                "JAKE_TOOLS_MICROSOFT_ACCOUNT_OBJECT_ID",
                _OPTIONAL_STR,
            ),
            None,
        ),
        microsoft_authority_url=resolve(
            _Spec(
                "microsoft.authority_url",
                "JAKE_TOOLS_MICROSOFT_AUTHORITY_URL",
                _STR,
            ),
            DEFAULT_MICROSOFT_AUTHORITY_URL,
        ),
        microsoft_refresh_token=secrets["MICROSOFT_REFRESH_TOKEN"],
        microsoft_client_secret=secrets["MICROSOFT_CLIENT_SECRET"],
        newsletters_graph_root=resolve(
            _Spec(
                "newsletters.graph_root",
                "JAKE_TOOLS_NEWSLETTERS_GRAPH_ROOT",
                _STR,
            ),
            DEFAULT_GRAPH_ROOT,
        ),
        newsletters_sharepoint_root=resolve(
            _Spec(
                "newsletters.sharepoint_root",
                "JAKE_TOOLS_NEWSLETTERS_SHAREPOINT_ROOT",
                _STR,
            ),
            DEFAULT_SHAREPOINT_ROOT,
        ),
        files_graph_root=resolve(
            _Spec(
                "files.graph_root",
                "JAKE_TOOLS_FILES_GRAPH_ROOT",
                _STR,
            ),
            DEFAULT_GRAPH_ROOT,
        ),
        files_sharepoint_hosts=resolve(
            _Spec(
                "files.sharepoint_hosts",
                "JAKE_TOOLS_FILES_SHAREPOINT_HOSTS",
                _STR_TUPLE,
            ),
            DEFAULT_FILES_SHAREPOINT_HOSTS,
        ),
        artifacts_enabled=resolve(
            _Spec(
                "artifacts.enabled",
                "JAKE_TOOLS_ARTIFACTS_ENABLED",
                _BOOL,
            ),
            False,
        ),
        hindsight_api_url=resolve(
            _Spec("hindsight.api_url", "JAKE_TOOLS_HINDSIGHT_API_URL", _HINDSIGHT_URL),
            None,
        ),
        hindsight_bank_id=resolve(
            _Spec("hindsight.bank_id", "JAKE_TOOLS_HINDSIGHT_BANK_ID", _HINDSIGHT_BANK),
            None,
        ),
        hindsight_access=resolve(
            _Spec("hindsight.access", "JAKE_TOOLS_HINDSIGHT_ACCESS", _HINDSIGHT_ACCESS),
            "readwrite",
        ),
    )
    return config


def _resolve[T](
    spec: _Spec[T],
    env: Mapping[str, str],
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
    if spec.env in env:
        raw: Any = env[spec.env]
        if spec.adapter is _STR_TUPLE:
            raw = tuple(item.strip() for item in raw.split(",") if item.strip())
        return Resolved(
            name=spec.name,
            value=_coerce(spec, raw, f"environment variable {spec.env}"),
            source="env",
            origin=spec.env,
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
    if spec.adapter is _POSITIVE_INT:
        return "a positive integer"
    if spec.adapter is _FLOAT:
        return "a number"
    if spec.adapter is _PATH:
        return "a path"
    if spec.adapter is _HINDSIGHT_URL:
        return "an https URL (or http to localhost) with no path"
    if spec.adapter is _HINDSIGHT_BANK:
        return "a bank id of letters, digits, . _ - (at most 128)"
    if spec.adapter is _HINDSIGHT_ACCESS:
        return "one of read, readwrite"
    if spec.adapter is _ACCOUNT_CONTEXT:
        return "a label of letters, digits and . _ : - (at most 64)"
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


def _optional_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    return Path(raw).expanduser()
