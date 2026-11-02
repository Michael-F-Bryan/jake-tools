"""The native Hermes plugin: Jake's operations and packaged skills.

Installed as the ``jake-tools`` entry point in the ``hermes_agent.plugins``
group, so Hermes discovers it in whatever Python environment its installer
(nixfiles) provides. It loads only when the profile lists ``jake-tools`` in
``plugins.enabled``. ``register(ctx)``:

- registers the shared :mod:`jake_tools.operations` for the capabilities
  named in ``plugins.entries.jake-tools.settings.capabilities`` (none when
  unset), as native tools of the same names as the MCP server's;
- registers the packaged skills those operations point to, from the same
  ``skills/`` sources the MCP server serves, as ``jake-tools:<skill>``.

It registers no terminal backend and does not touch Hermes' terminal: the
installer's own backend and hooks are unchanged. Jake's credential names are
therefore not on Hermes' subprocess scrub list (see the README, "Credentials
and terminal commands").

Handlers run inside Hermes. Credentials come from the owning profile through
``agent.secret_scope.get_secret`` on every call, are consumed by the
operation, and are never returned. Delegated Claude workers stay separate
processes with an allowlisted environment, as under MCP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.plugins import (  # pyright: ignore[reportMissingImports]
        PluginContext,
    )

PLUGIN_NAME = "jake-tools"


def register(ctx: PluginContext) -> None:
    """Hermes' plugin entry point."""
    from .plugin import register_plugin

    register_plugin(ctx)
