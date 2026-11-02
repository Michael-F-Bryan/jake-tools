"""Per-call configuration for the native tools, from the owning profile.

Hermes may serve several profiles from one process, so the process
environment is not the profile's: secrets come from a ``get_secret``
callable (Hermes' ``agent.secret_scope.get_secret``, which resolves the
active profile's ``.env`` and secret sources and fails closed when no
profile is in scope). Everything else Jake reads (XDG paths, non-secret
``JAKE_TOOLS_*`` settings, ``config.toml``) is process-wide, as for the MCP
server.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from ..config import SECRET_VARIABLES, Config, load_config

SecretGetter = Callable[[str], str | None]

SECRET_NAMES: frozenset[str] = frozenset(
    name for base in SECRET_VARIABLES for name in (base, f"{base}_FILE")
)
"""Every variable that carries a Jake secret or names a file holding one."""

WORKER_ENV_KEYS: frozenset[str] = frozenset(
    {
        # What Hermes passes a stdio MCP server (tools/mcp_tool_config.py),
        # so a worker sees what it saw when spawned by the MCP server.
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TERM",
        "SHELL",
        "TMPDIR",
        # The loader settings the MCP server was configured with for
        # foreign executables (a CLI the SDK may fall back to).
        "NIX_LD",
        "NIX_LD_LIBRARY_PATH",
        # The worker runs on Hermes' interpreter; an installer that adds
        # jake-tools by PYTHONPATH (Hermes' Nix package) needs it there.
        "PYTHONPATH",
    }
)


def worker_base_environ(environ: Mapping[str, str]) -> dict[str, str]:
    """The allowlisted part of ``environ`` a delegated worker may inherit.

    Hermes' own environment holds its provider keys and bot tokens; a worker
    can run Bash, so it gets only these names and ``XDG_*``. Jake's secrets
    are removed again by :func:`~jake_tools.claude_runs.spawn.worker_environment`.
    """
    return {
        key: value
        for key, value in environ.items()
        if (key in WORKER_ENV_KEYS or key.startswith("XDG_"))
        and key not in SECRET_NAMES
    }


def profile_config(
    get_secret: SecretGetter,
    environ: Mapping[str, str],
    *,
    hermes_home: Path | None = None,
) -> Config:
    """Resolve Jake's configuration for the profile ``get_secret`` serves.

    Secret variables in ``environ`` are ignored even when set: in a process
    serving several profiles they belong to whichever profile launched it.
    ``hermes_home`` is the profile's home (``HERMES_HOME`` for defaults such
    as the session store).
    """
    env = {key: value for key, value in environ.items() if key not in SECRET_NAMES}
    for name in sorted(SECRET_NAMES):
        value = get_secret(name)
        if value is not None:
            env[name] = value
    if hermes_home is not None:
        env["HERMES_HOME"] = str(hermes_home)
    return load_config(env, worker_environ=worker_base_environ(environ))
