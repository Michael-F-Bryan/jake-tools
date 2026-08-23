"""The Obsidian vault-client seam.

Downstream pipeline stages need two things from the vault: where it lives on
disk, and where a note's ``![[...]]`` embed actually resolves to. Both are
exposed through :class:`VaultClient` so tests can inject a fake instead of
shelling out to the real Obsidian.app CLI.

:class:`ObsidianCli` is the real, subprocess-backed implementation. Probing
the real binary (v1.13.7) showed two quirks worth quarantining behind this
one class:

- Every invocation prints an Electron loader line (``2026-08-23 07:39:31
  Loading updated app package ...``) and, on this out-of-date install, an
  "installer is out of date" warning line before the real payload. Both are
  stripped before the payload is parsed.
- The CLI exits ``0`` even when a lookup fails — a missing file prints
  ``Error: File "..." not found.`` on stdout with exit code 0. Success is
  therefore judged by the payload's shape, not the exit code alone.

``file file=<name>`` resolves an embed target's vault-relative path directly
(confirmed against the real vault), so that is tried first; a vault-relative
glob for ``**/<target>`` is the fallback for anything the CLI can't resolve.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

RunFn = Callable[..., subprocess.CompletedProcess[str]]

_ADVISORY_LINE_PREFIXES = ("Your Obsidian installer is out of date",)


class ObsidianCliError(RuntimeError):
    """Raised when the ``obsidian`` CLI fails and no fallback resolves the target."""


class VaultClient(Protocol):
    def vault_root(self) -> Path: ...

    def resolve_embed(self, target: str) -> Path:
        """Resolve an embed target (e.g. ``"Recording 20260811112359.m4a"``)
        to its absolute path on disk."""
        ...


class ObsidianCli:
    """Talks to the running Obsidian app via the ``obsidian`` binary."""

    def __init__(
        self,
        vault: str | None = None,
        binary: str = "obsidian",
        run: RunFn = subprocess.run,
    ) -> None:
        self._vault = vault
        self._binary = binary
        self._run = run
        self._vault_root: Path | None = None

    def vault_root(self) -> Path:
        if self._vault_root is None:
            payload = self._invoke(["vault", "info=path"])
            self._vault_root = Path(payload.strip())
        return self._vault_root

    def resolve_embed(self, target: str) -> Path:
        root = self.vault_root()
        relative = self._resolve_via_cli(target)
        if relative is not None:
            candidate = root / relative
            if candidate.exists():
                return candidate
        return self._resolve_via_glob(root, target)

    def _resolve_via_cli(self, target: str) -> str | None:
        try:
            payload = self._invoke(["file", f"file={target}"])
        except ObsidianCliError:
            return None
        fields = _parse_fields(payload)
        return fields.get("path")

    def _resolve_via_glob(self, root: Path, target: str) -> Path:
        matches = sorted(root.glob(f"**/{target}"))
        if len(matches) == 0:
            raise ObsidianCliError(
                f"no file resolves embed target {target!r} under {root}"
            )
        if len(matches) > 1:
            candidates = ", ".join(str(match) for match in matches)
            raise ObsidianCliError(
                f"embed target {target!r} is ambiguous, candidates: {candidates}"
            )
        return matches[0]

    def _invoke(self, args: list[str]) -> str:
        command = [self._binary]
        if self._vault is not None:
            command.append(f"vault={self._vault}")
        command.extend(args)
        try:
            result = self._run(command, capture_output=True, text=True, check=False)
        except OSError as exc:
            raise ObsidianCliError(f"failed to run {self._binary}: {exc}") from exc

        payload = _strip_cli_noise(result.stdout)
        if result.returncode != 0 or _looks_like_error(payload):
            detail = payload or f"{self._binary} exited with {result.returncode}"
            raise ObsidianCliError(detail)
        return payload


def _strip_cli_noise(stdout: str) -> str:
    """Drop the Electron loader line and known advisory lines from stdout."""
    kept: list[str] = []
    for line in stdout.splitlines():
        if _is_loader_line(line):
            continue
        if line.startswith(_ADVISORY_LINE_PREFIXES):
            continue
        kept.append(line)
    return "\n".join(kept).strip("\n")


def _is_loader_line(line: str) -> bool:
    # "2026-08-23 07:39:31 Loading updated app package ..."
    parts = line.split(" ", 2)
    if len(parts) < 3:
        return False
    date_part, time_part, _rest = parts
    return (
        len(date_part) == 10
        and date_part.count("-") == 2
        and len(time_part) == 8
        and time_part.count(":") == 2
    )


def _looks_like_error(payload: str) -> bool:
    return any(line.startswith("Error:") for line in payload.splitlines())


def _parse_fields(payload: str) -> dict[str, str]:
    """Parse the ``key\\tvalue`` lines the CLI prints for e.g. ``file``."""
    fields: dict[str, str] = {}
    for line in payload.splitlines():
        if "\t" not in line:
            continue
        key, _, value = line.partition("\t")
        fields[key] = value
    return fields
