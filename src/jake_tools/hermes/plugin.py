"""``register(ctx)``: what the jake-tools plugin installs into Hermes."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Config
from ..operations import Operation
from ..operations.catalogue import OPERATIONS
from .profile import profile_config
from .tools import ConfigFactory, make_handler, signature_metadata, tool_schema

if TYPE_CHECKING:
    # The pinned Hermes' own plugin API; present only inside a Hermes runtime.
    from hermes_cli.plugins import (  # pyright: ignore[reportMissingImports]
        PluginContext,
    )

log = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def register_plugin(ctx: PluginContext) -> None:
    # Registration needs only non-secret facts (signatures, which endpoints
    # are configured); the profile's secrets are read per call.
    register_operations(
        ctx,
        current_profile_config(with_secrets=False),
        current_profile_config,
    )


def register_operations(
    ctx: PluginContext,
    config: Config,
    config_factory: ConfigFactory,
) -> tuple[Operation, ...]:
    """Register the available operations and the skills they point to."""
    operations = available_operations(config)
    for operation in operations:
        metadata = signature_metadata(operation, config)
        ctx.register_tool(
            name=operation.name,
            toolset=f"jake_tools_{operation.capability}",
            schema=tool_schema(operation, metadata),
            handler=make_handler(operation, metadata, config_factory),
            description=operation.description,
        )
    for skill in skills_for(operations):
        ctx.register_skill(skill, SKILLS_DIR / skill / "SKILL.md")
    log.info("jake-tools: %d tools", len(operations))
    return operations


def available_operations(config: Config) -> tuple[Operation, ...]:
    return tuple(operation for operation in OPERATIONS if operation.available(config))


def skills_for(operations: Iterable[Operation]) -> tuple[str, ...]:
    """The packaged skills the registered tools point to, once each."""
    return tuple(dict.fromkeys(operation.skill for operation in operations))


def current_profile_config(*, with_secrets: bool = True) -> Config:
    """Jake's configuration for the profile Hermes is serving right now."""
    from agent.secret_scope import get_secret  # pyright: ignore[reportMissingImports]
    from hermes_constants import (  # pyright: ignore[reportMissingImports]
        get_hermes_home,
    )

    getter = get_secret if with_secrets else _no_secret
    return profile_config(getter, os.environ, hermes_home=Path(get_hermes_home()))


def _no_secret(name: str) -> None:
    return None
