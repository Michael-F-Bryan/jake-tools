from __future__ import annotations

import click


@click.group("ai-watch")
def ai_watch() -> None:
    """Low-noise AI developments radar."""


def _register_stub(name: str) -> None:
    @click.command(name)
    def _command() -> None:
        raise click.ClickException(f"{name} is not implemented yet")

    ai_watch.add_command(_command)


for _command_name in (
    "collect",
    "fetch",
    "scout",
    "curate",
    "obsidian-sync",
    "digest",
    "deliver",
    "run",
    "audit",
):
    _register_stub(_command_name)
