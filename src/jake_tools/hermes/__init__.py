"""The native Hermes plugin: Jake's operations, skills and terminal backend.

Installed as the ``jake-tools`` entry point in the ``hermes_agent.plugins``
group, so Hermes discovers it in whatever Python environment its installer
(nixfiles) provides. It loads only when the profile lists ``jake-tools`` in
``plugins.enabled``. ``register(ctx)``:

- registers the shared :mod:`jake_tools.operations` for the capabilities
  named in ``plugins.entries.jake-tools.settings.capabilities`` (none when
  unset), as native tools of the same names as the MCP server's;
- registers the packaged skills those operations point to, from the same
  ``skills/`` sources the MCP server serves, as ``jake-tools:<skill>``;
- registers the ``direnv_local`` terminal backend, whose ``strip_env_keys``
  keep Jake's credentials out of the subprocesses Hermes scrubs. Selecting it
  is degraded in pinned Hermes (no PTY, background stdin, yield-to-background,
  background recovery or ``execute_code`` kernel), so it refuses to run unless
  ``settings.accept_degraded_direnv_local`` is true; see
  :mod:`jake_tools.hermes.project_env`.

Handlers run inside Hermes. Credentials come from the owning profile through
``agent.secret_scope.get_secret`` on every call, are consumed by the
operation, and are never returned. Delegated Claude workers stay separate
processes with an allowlisted environment, as under MCP.
"""

from __future__ import annotations

from typing import Any

PLUGIN_NAME = "jake-tools"


def register(ctx: Any) -> None:
    """Hermes' plugin entry point."""
    from .plugin import register_plugin

    register_plugin(ctx)
