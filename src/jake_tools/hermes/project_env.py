"""Where and when the ``direnv_local`` backend loads a project environment,
and why it is gated.

Hermes' local backend runs each command as a bash script that restores the
session snapshot, enters the command's directory with
``builtin cd -- <dir> || exit 126``, then ``eval``s the command, and finally
re-dumps exported variables into the snapshot. Sourcing the project loader
between the ``cd`` and the ``eval`` loads the right directory's environment
after the snapshot (so Hermes cannot overwrite it), in the same shell (so
``cd`` and exports persist and the next directory change unloads it), and
without touching the command Hermes previews, approves and records.

Only the agent's own terminal commands get it, as with the old
argument-rewriting hook: the foreground terminal tool (the one caller that
passes ``bounded_capture=True``) and background launches (wrapped by the
process registry). Hermes' internal commands (output pollers, kills, file
reads) run as plain local commands, so a broken or unapproved project
cannot stop it watching a background process.

The backend is degraded, and so gated. Pinned Hermes (5b32cfc) decides
several behaviours by the literal backend name ``local``, which no plugin
backend can have (built-in names are reserved), and offers no seam that loads
a project environment under ``local`` without rewriting the command it
previews and approves. :data:`LOST_LOCAL_BEHAVIOURS` lists what selecting this
backend gives up; it refuses to run until the profile accepts that
explicitly (:data:`ACCEPT_DEGRADED_SETTING`). :data:`REQUIRED_UPSTREAM_SEAM`
is the Hermes change that would make it unnecessary.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from typing import Any

BACKEND_NAME = "direnv_local"

ACCEPT_DEGRADED_SETTING = "accept_degraded_direnv_local"
"""``plugins.entries.jake-tools.settings.<this>: true`` lets the profile run
``direnv_local`` despite :data:`LOST_LOCAL_BEHAVIOURS`. Unset or false, the
backend reports its requirements unmet, so Hermes withholds the terminal tool
and says why, rather than running with less than ``local`` does."""

LOST_LOCAL_BEHAVIOURS: tuple[str, ...] = (
    "PTY sessions (pty=true runs on a pipe)",
    "stdin to background processes (process write/submit/close need a PTY)",
    "yield-to-background of a running foreground command",
    "recovery of background processes after a Hermes restart",
    "the persistent execute_code session kernel",
    "the local browser's access to private-network URLs",
    "vision reads of arbitrary host files",
    "terminal.local_persistent",
)
"""What pinned Hermes keys on the literal backend name ``local``."""

REQUIRED_UPSTREAM_SEAM = (
    "a Hermes hook that sources a plugin-supplied prelude for the local "
    "backend's agent-issued commands, after the snapshot is restored and the "
    "directory entered and before the command runs, in both the foreground "
    "wrapper and background/PTY spawns, without changing the command that is "
    "previewed, approved or recorded"
)


def degraded_backend_message() -> str:
    """Why ``direnv_local`` will not run, for Hermes' log and the user."""
    return (
        f"terminal.backend {BACKEND_NAME} is degraded in this Hermes: it loses "
        f"{'; '.join(LOST_LOCAL_BEHAVIOURS)}. Use terminal.backend local, or set "
        f"plugins.entries.jake-tools.settings.{ACCEPT_DEGRADED_SETTING}: true to "
        f"accept that. A fix needs {REQUIRED_UPSTREAM_SEAM}."
    )


PROJECT_ENV_SCRIPT_VARIABLE = "JAKE_TOOLS_PROJECT_ENV_SCRIPT"
"""Names the shell file to source in the command's directory. The installer
(nixfiles) owns it: it runs direnv with nix-direnv and fails when an
``.envrc`` is present but its environment did not load."""

_CD_LINE = re.compile(r"builtin cd -- .+ \|\| exit 126")
_EVAL_LINE = re.compile(r"eval '")
_BACKGROUND_LAUNCH = re.compile(r"mkdir -p \S+ && \( nohup bash -lc ")


class WrapperChanged(RuntimeError):
    """Hermes' command wrapper no longer has the shape this backend relies on."""


def insert_project_env(script: str, project_env_script: str) -> str:
    """``script`` with the loader sourced right after it enters the directory.

    A loader failure exits with status 1 before the command runs; the
    loader has already said why on stderr. Raises :class:`WrapperChanged`
    rather than guessing when the wrapper's ``cd``/``eval`` lines are not
    where they were.
    """
    lines = script.split("\n")
    cd_index = next(
        (index for index, line in enumerate(lines) if _CD_LINE.fullmatch(line)), None
    )
    eval_index = next(
        (index for index, line in enumerate(lines) if _EVAL_LINE.match(line)), None
    )
    if cd_index is None or eval_index is None or eval_index != cd_index + 1:
        raise WrapperChanged(
            "Hermes' terminal command wrapper changed shape; the direnv_local "
            "backend cannot place the project environment safely"
        )
    loader = f". {shlex.quote(project_env_script)} || exit 1"
    return "\n".join([*lines[: cd_index + 1], loader, *lines[cd_index + 1 :]])


def wants_project_env(command: str, execute_kwargs: Mapping[str, Any]) -> bool:
    """Whether an ``execute()`` call is one of the agent's terminal commands."""
    if execute_kwargs.get("bounded_capture") is True:
        return True
    return execute_kwargs.get("rewrite_compound_background") is False and bool(
        _BACKGROUND_LAUNCH.match(command)
    )
