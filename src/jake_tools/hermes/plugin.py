"""``register(ctx)``: what the jake-tools plugin installs into Hermes."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_args

from ..config import Config
from ..operations import Capability, Operation
from ..operations.catalogue import OPERATIONS
from .profile import profile_config
from .project_env import ACCEPT_DEGRADED_SETTING
from .tools import ConfigFactory, make_handler, signature_metadata, tool_schema

log = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
CAPABILITIES: tuple[str, ...] = get_args(Capability)


def register_plugin(ctx: Any) -> None:
    from .terminal import DirenvLocalProvider

    ctx.register_terminal_environment_provider(
        DirenvLocalProvider(
            accept_degraded=accepts_degraded_backend(
                ctx.get_config(ACCEPT_DEGRADED_SETTING)
            )
        )
    )
    # Registration needs only non-secret facts (signatures, which endpoints
    # are configured); the profile's secrets are read per call.
    register_operations(
        ctx,
        configured_capabilities(ctx.get_config("capabilities")),
        current_profile_config(with_secrets=False),
        current_profile_config,
    )


def register_operations(
    ctx: Any,
    capabilities: tuple[Capability, ...],
    config: Config,
    config_factory: ConfigFactory,
) -> tuple[Operation, ...]:
    """Register the granted operations and the skills they point to."""
    operations = enabled_operations(capabilities, config)
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
    log.info(
        "jake-tools: capabilities %s; %d tools",
        ", ".join(capabilities) or "none",
        len(operations),
    )
    return operations


def configured_capabilities(value: object) -> tuple[Capability, ...]:
    """``plugins.entries.jake-tools.settings.capabilities``, validated.

    Unset means none: an agent gets Jake's tools only when its configuration
    names them. A malformed value fails the plugin's load (Hermes reports it)
    rather than guessing.
    """
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("jake-tools settings.capabilities must be a list of names")
    unknown = sorted(set(value) - set(CAPABILITIES))
    if unknown:
        raise ValueError(
            f"jake-tools settings.capabilities has unknown {', '.join(unknown)}; "
            f"known: {', '.join(CAPABILITIES)}"
        )
    return tuple(c for c in CAPABILITIES if c in value)  # type: ignore[misc]


def accepts_degraded_backend(value: object) -> bool:
    """``settings.accept_degraded_direnv_local``, validated; unset is false."""
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(
            f"jake-tools settings.{ACCEPT_DEGRADED_SETTING} must be a boolean"
        )
    return value


def enabled_operations(
    capabilities: Iterable[Capability], config: Config
) -> tuple[Operation, ...]:
    granted = set(capabilities)
    return tuple(
        operation
        for operation in OPERATIONS
        if operation.capability in granted and operation.available(config)
    )


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
