"""The ``direnv_local`` terminal backend (Hermes runtime only).

Local execution exactly as Hermes' ``local`` backend does it, plus the
project environment of each terminal command's directory; see
:mod:`.project_env`. Selected with ``terminal.backend: direnv_local``, it runs
only when the profile also sets ``settings.accept_degraded_direnv_local``:
pinned Hermes gives a plugin backend none of the behaviours it keys on the
literal name ``local`` (:data:`~.project_env.LOST_LOCAL_BEHAVIOURS`).
Otherwise its requirements are unmet and Hermes withholds the terminal tool
with that reason. Registered either way: its ``strip_env_keys`` keep Jake's
credentials out of the subprocesses Hermes scrubs, whichever backend runs.
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
    BACKEND_NAME,
    PROJECT_ENV_SCRIPT_VARIABLE,
    WrapperChanged,
    degraded_backend_message,
    insert_project_env,
    wants_project_env,
)

log = logging.getLogger(__name__)

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
    """``direnv_local``, usable only when ``accept_degraded`` (see module doc)."""

    name = BACKEND_NAME
    display_name = "Local with project environments (degraded)"
    is_remote = False
    is_container = False

    def __init__(self, *, accept_degraded: bool) -> None:
        super().__init__()
        self.accept_degraded = accept_degraded

    @property
    def description(self) -> str:
        return (
            "Run commands on this host, loading each directory's approved direnv "
            "project environment before the command. Degraded: no PTY, background "
            "stdin, yield-to-background, background recovery or execute_code "
            "kernel; prefer local."
        )

    @property
    def skip_container_guards(self) -> bool:
        return False  # host execution: dangerous-command approval stays on

    @property
    def default_cwd(self) -> str | None:
        return os.getcwd()

    @property
    def strip_env_keys(self) -> frozenset:
        # Hermes strips these from the subprocesses it scrubs (background,
        # PTY and helper spawns), whichever backend is selected: Jake's
        # credentials are for its tools only.
        return SECRET_NAMES

    @property
    def env_description(self) -> str:
        return "this host, with the working directory's project environment loaded"

    def is_available(self) -> bool:
        return self.accept_degraded and project_env_script() is not None

    def check_requirements(self, config: dict[str, Any]) -> bool:
        if not self.accept_degraded:
            log.error("%s", degraded_backend_message())
            return False
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
        if not self.accept_degraded:
            raise RuntimeError(degraded_backend_message())
        script = project_env_script()
        if script is None:
            raise RuntimeError(
                f"{PROJECT_ENV_SCRIPT_VARIABLE} does not name the project environment "
                f"loader; the {BACKEND_NAME} backend cannot run commands"
            )
        log.warning("%s (accepted by this profile)", degraded_backend_message())
        return DirenvLocalEnvironment(script, cwd=cwd, timeout=timeout)
