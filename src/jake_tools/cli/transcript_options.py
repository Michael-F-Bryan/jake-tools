"""Options objects and decorators for the ``transcript`` CLI group.

Transcription commands build their dependencies from injected options
objects rather than shared context state (memo:
`_working/transcription-workflow-interview/plans/memo-cli-options.md`,
E19). Each dependency a command needs arrives as a typed options
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

from ..transcription.asr import (
    DEFAULT_ASR_CHUNK_DURATION,
    DEFAULT_ASR_CHUNK_OVERLAP,
    DEFAULT_ASR_MODEL,
    DEFAULT_DIARISATION_MODEL,
    DiarisationDevice,
    LocalTranscriber,
    Transcriber,
)
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


class TranscriberOptions(BaseModel):
    """Which ASR/diarisation models to run, and the token that gates them.

    `hf_token` is read from `--hf-token` or the `HF_TOKEN` environment
    variable (`.env` is already loaded elsewhere in the package) — pyannote's
    pretrained diarisation pipeline is a gated HuggingFace model. It is
    never logged or echoed; :class:`~..transcription.asr.LocalTranscriber`
    only refers to it as `HF_TOKEN` in error messages.

    `memory_budget_bytes=None` means "let `LocalTranscriber` derive its own
    ~60%-of-RAM default at run time" — the CLI flag exists so that default
    is tunable without a code change if a real run shows it's wrong (it
    hasn't been empirically validated against a real crash trace; see
    task-004b's report), not because every caller needs to set it.
    """

    hf_token: str | None
    asr_model: str
    diarisation_model: str
    diarisation_device: DiarisationDevice
    num_speakers: int | None
    asr_chunk_duration: float
    asr_chunk_overlap: float
    memory_budget_bytes: int | None

    def transcriber(self, *, num_speakers: int | None = None) -> Transcriber:
        return LocalTranscriber(
            hf_token=self.hf_token,
            asr_model=self.asr_model,
            diarisation_model=self.diarisation_model,
            diarisation_device=self.diarisation_device,
            num_speakers=(self.num_speakers if num_speakers is None else num_speakers),
            asr_chunk_duration=self.asr_chunk_duration,
            asr_chunk_overlap=self.asr_chunk_overlap,
            memory_budget_bytes=self.memory_budget_bytes,
        )


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


def transcriber_options(func: F) -> F:
    @click.option(
        "--hf-token",
        "hf_token",
        envvar="HF_TOKEN",
        default=None,
        help=(
            "HuggingFace token for pyannote's gated diarisation model "
            "(or set HF_TOKEN). Not needed for a cached run."
        ),
    )
    @click.option(
        "--asr-model",
        "asr_model",
        default=DEFAULT_ASR_MODEL,
        show_default=True,
        help="parakeet-mlx model id (or local path) for ASR.",
    )
    @click.option(
        "--diarisation-model",
        "diarisation_model",
        default=DEFAULT_DIARISATION_MODEL,
        show_default=True,
        help="pyannote-audio pipeline id (or local path) for speaker diarisation.",
    )
    @click.option(
        "--diarisation-device",
        "diarisation_device",
        type=click.Choice(["auto", "cpu", "mps"], case_sensitive=True),
        default="auto",
        show_default=True,
        help="Diarisation device. auto uses MPS only when it is available.",
    )
    @click.option(
        "--num-speakers",
        "num_speakers",
        type=click.IntRange(min=1),
        default=None,
        help="Known speaker count; omitted means unconstrained diarisation.",
    )
    @click.option(
        "--asr-chunk-duration",
        "asr_chunk_duration",
        type=float,
        default=DEFAULT_ASR_CHUNK_DURATION,
        show_default=True,
        help=(
            "Chunk length, in seconds, for parakeet-mlx's own bounded-"
            "memory ASR chunking. Lower to reduce peak ASR memory further "
            "at the cost of more (overlapping) inference passes."
        ),
    )
    @click.option(
        "--asr-chunk-overlap",
        "asr_chunk_overlap",
        type=float,
        default=DEFAULT_ASR_CHUNK_OVERLAP,
        show_default=True,
        help="Overlap, in seconds, between consecutive ASR chunks.",
    )
    @click.option(
        "--memory-budget-bytes",
        "memory_budget_bytes",
        type=int,
        default=None,
        help=(
            "Byte ceiling for the memory watchdog and MLX's accelerator "
            "memory cap (default: ~60% of total physical RAM, computed at "
            "run time - see LocalTranscriber/memory_watchdog)."
        ),
    )
    @click.pass_context
    @functools.wraps(func)
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        options = TranscriberOptions(
            hf_token=kwargs.pop("hf_token"),
            asr_model=kwargs.pop("asr_model"),
            diarisation_model=kwargs.pop("diarisation_model"),
            diarisation_device=kwargs.pop("diarisation_device"),
            num_speakers=kwargs.pop("num_speakers"),
            asr_chunk_duration=kwargs.pop("asr_chunk_duration"),
            asr_chunk_overlap=kwargs.pop("asr_chunk_overlap"),
            memory_budget_bytes=kwargs.pop("memory_budget_bytes"),
        )
        return ctx.invoke(func, *args, transcriber_options=options, **kwargs)

    return cast(F, wrapper)
