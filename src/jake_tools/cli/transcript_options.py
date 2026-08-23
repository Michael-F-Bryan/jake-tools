"""Options objects and decorators for the ``transcript`` CLI group.

Transcription commands do not use ``AppContext``/``ctx.obj`` (memo:
`_working/transcription-workflow-interview/plans/memo-cli-options.md`,
E19). Instead, each dependency a command needs arrives as a typed options
object — a Pydantic model holding raw flag/env values plus one or more
dependency-constructor methods — injected by a decorator that stacks the
underlying ``click.option``s, pops their values out of the parsed kwargs,
builds the model, and forwards it via ``ctx.invoke``. This is the same
shape as ``lcd``'s ``DbOptions``/``db_options`` (``lcd/cli/options.py``):

    @some_options
    def command(some_options: SomeOptions): ...

A command handler constructs its real dependencies from the options objects
at the top of the function and delegates everything else to library code in
``transcription/``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import click
from pydantic import BaseModel

from ..transcription.audio import AudioTool, FfmpegAudioTool
from ..transcription.cache import RunCache
from ..transcription.obsidian import ObsidianCli, VaultClient

F = Callable[..., Any]


class ObsidianOptions(BaseModel):
    """Which Obsidian vault/CLI binary to resolve embeds through."""

    vault: str | None
    binary: str

    def vault_client(self) -> VaultClient:
        return ObsidianCli(vault=self.vault, binary=self.binary)


class AudioOptions(BaseModel):
    """Which ffmpeg/ffprobe binaries to shell out to."""

    ffmpeg: str
    ffprobe: str

    def audio_tool(self) -> AudioTool:
        return FfmpegAudioTool(ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)


class CacheOptions(BaseModel):
    """Where the run cache lives; ``root=None`` means the standard location."""

    root: Path | None

    def run_cache(self) -> RunCache:
        return RunCache(root=self.root)


def obsidian_options(func: F) -> F:
    @click.option(
        "--vault",
        envvar="OBSIDIAN_VAULT",
        default=None,
        help="Obsidian vault name to select (only needed with multiple vaults open).",
    )
    @click.option(
        "--obsidian-binary",
        "obsidian_binary",
        envvar="OBSIDIAN_BINARY",
        default="obsidian",
        show_default=True,
        help="Path to (or name of) the obsidian CLI binary.",
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        options = ObsidianOptions(
            vault=kwargs.pop("vault"),
            binary=kwargs.pop("obsidian_binary"),
        )
        return ctx.invoke(func, *args, obsidian_options=options, **kwargs)

    return cast(F, wrapper)


def audio_options(func: F) -> F:
    @click.option(
        "--ffmpeg",
        envvar="JAKE_TOOLS_FFMPEG",
        default="ffmpeg",
        show_default=True,
        help="Path to (or name of) the ffmpeg binary.",
    )
    @click.option(
        "--ffprobe",
        envvar="JAKE_TOOLS_FFPROBE",
        default="ffprobe",
        show_default=True,
        help="Path to (or name of) the ffprobe binary.",
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        options = AudioOptions(
            ffmpeg=kwargs.pop("ffmpeg"),
            ffprobe=kwargs.pop("ffprobe"),
        )
        return ctx.invoke(func, *args, audio_options=options, **kwargs)

    return cast(F, wrapper)


def cache_options(func: F) -> F:
    @click.option(
        "--cache-root",
        "cache_root",
        type=click.Path(path_type=Path, file_okay=False),
        default=None,
        help=(
            "Root directory for the run cache (mainly for tests/agents). "
            "Defaults to the standard per-user cache location."
        ),
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        options = CacheOptions(root=kwargs.pop("cache_root"))
        return ctx.invoke(func, *args, cache_options=options, **kwargs)

    return cast(F, wrapper)
