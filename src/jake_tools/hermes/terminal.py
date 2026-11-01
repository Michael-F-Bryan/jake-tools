"""The ``direnv_local`` terminal backend (Hermes runtime only).

Local execution exactly as Hermes' ``local`` backend does it, plus the
project environment of each terminal command's directory; see
:mod:`.project_env`. Select it with ``terminal.backend: direnv_local``.

What a plugin backend cannot have in the pinned Hermes: core decides PTY
sessions, stdin to background processes, yield-to-background and the
in-process ``execute_code`` kernel by the literal backend name ``local``.
Background commands still run (through the process registry's sandbox path:
output is polled from a log, and ``pty=true`` is not honoured).
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from agent.terminal_env_provider import (  # pyright: ignore[reportMissingImports]
    TerminalEnvironmentProvider,
)
from tools.environments.local import (  # pyright: ignore[reportMissingImports]
    LocalEnvironment,
)

from .profile import SECRET_NAMES
from .project_env import (
    PROJECT_ENV_SCRIPT_VARIABLE,
    WrapperChanged,
    insert_project_env,
    wants_project_env,
)

log = logging.getLogger(__name__)

BACKEND_NAME = "direnv_local"

UNSET_SECRETS = f"unset {' '.join(sorted(SECRET_NAMES))} 2>/dev/null"

_LOAD_PROJECT_ENV: ContextVar[bool] = ContextVar(
    "jake_tools_load_project_env", default=False
)


def project_env_script() -> str | None:
    """The configured loader, when it names an existing file."""
    value = os.environ.get(PROJECT_ENV_SCRIPT_VARIABLE, "").strip()
    return value if value and Path(value).is_file() else None


def probe_wrapper() -> str:
    """A command script from this Hermes' own wrapper builder, to check its shape."""
    from tools.environments.base_session_env import (  # pyright: ignore[reportMissingImports]
        _wrap_command_script,
    )

    return _wrap_command_script(
        "true",
        quoted_cwd="/probe",
        quoted_snap="/probe/snapshot.sh",
        snap_tmp_template="/probe/snapshot.sh.tmp.XXXXXXXXXX",
        passthrough_names=(),
        snapshot_ready=True,
        cwd_marker="__probe_cwd__",
        pipe_marker="__probe_pipe__",
    )


class DirenvLocalEnvironment(LocalEnvironment):  # pyright: ignore[reportUntypedBaseClass]
    """``LocalEnvironment`` whose terminal commands load the project env."""

    def __init__(self, project_env_script: str, **kwargs: Any) -> None:
        self.project_env_script = project_env_script
        super().__init__(**kwargs)

    def _run_bash(self, cmd_string: str, **kwargs: Any) -> Any:
        # Hermes' foreground run environment does not apply providers'
        # strip_env_keys (only its background/worker spawns do), and the login
        # snapshot is taken from it. Unsetting first keeps Jake's credentials
        # out of every script this environment runs, the snapshot included.
        return super()._run_bash(f"{UNSET_SECRETS}\n{cmd_string}", **kwargs)

    def execute(self, command: str, cwd: str = "", **kwargs: Any) -> dict:
        token = _LOAD_PROJECT_ENV.set(wants_project_env(command, kwargs))
        try:
            return super().execute(command, cwd, **kwargs)
        finally:
            _LOAD_PROJECT_ENV.reset(token)

    def _wrap_command(self, command: str, cwd: str) -> str:
        script = super()._wrap_command(command, cwd)
        if not _LOAD_PROJECT_ENV.get():
            return script
        return insert_project_env(script, self.project_env_script)


class DirenvLocalProvider(TerminalEnvironmentProvider):  # pyright: ignore[reportUntypedBaseClass]
    name = BACKEND_NAME
    display_name = "Local with project environments"
    is_remote = False
    is_container = False

    @property
    def description(self) -> str:
        return (
            "Run commands on this host, loading each directory's approved direnv "
            "project environment before the command."
        )

    @property
    def skip_container_guards(self) -> bool:
        return False  # host execution: dangerous-command approval stays on

    @property
    def default_cwd(self) -> str | None:
        return os.getcwd()

    @property
    def strip_env_keys(self) -> frozenset:
        # Hermes strips these from every subprocess it spawns, whichever
        # backend is selected: Jake's credentials are for its tools only.
        return SECRET_NAMES

    @property
    def env_description(self) -> str:
        return "this host, with the working directory's project environment loaded"

    def is_available(self) -> bool:
        return project_env_script() is not None

    def check_requirements(self, config: dict[str, Any]) -> bool:
        script = project_env_script()
        if script is None:
            log.error(
                "terminal.backend %s needs %s to name the project environment loader",
                BACKEND_NAME,
                PROJECT_ENV_SCRIPT_VARIABLE,
            )
            return False
        try:
            insert_project_env(probe_wrapper(), script)
        except (WrapperChanged, TypeError) as exc:
            log.error("terminal.backend %s is incompatible: %s", BACKEND_NAME, exc)
            return False
        return True

    def create_environment(
        self,
        *,
        cwd: str,
        timeout: int,
        task_id: str = "default",
        image: str | None = None,
        container_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DirenvLocalEnvironment:
        script = project_env_script()
        if script is None:
            raise RuntimeError(
                f"{PROJECT_ENV_SCRIPT_VARIABLE} does not name the project environment "
                f"loader; the {BACKEND_NAME} backend cannot run commands"
            )
        return DirenvLocalEnvironment(script, cwd=cwd, timeout=timeout)
