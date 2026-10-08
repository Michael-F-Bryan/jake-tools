"""``python -m jake_tools.mcp``: serve (default) | doctor | install-skills | worker.

Stdout is the MCP transport, so every sub-command logs to stderr only.
Configuration comes from :func:`jake_tools.config.load_config` over the
process environment; each sub-command loads it itself, per the repository's
no-``ctx.obj`` rule.
"""

from __future__ import annotations

import logging
import sys

import click

from ..config import ConfigError, load_config
from .doctor import doctor
from .install_skills import install_skills
from .worker import worker


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def main(ctx: click.Context) -> None:
    """The jake-tools MCP server for Hermes."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(serve)


@main.command()
def serve() -> None:
    """Run the stdio MCP server (the default when no sub-command is given)."""
    from .server import build_server

    configure_logging()
    try:
        config = load_config()
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    app = build_server(config)
    app.run(transport="stdio")


def configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


main.add_command(doctor)
main.add_command(install_skills)
main.add_command(worker)

if __name__ == "__main__":
    main()
