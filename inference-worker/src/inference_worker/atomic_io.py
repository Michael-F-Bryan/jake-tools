"""Atomic text writes: temp file + fsync + os.replace (B2).

``Path.write_text`` truncates the destination before writing its
replacement content. If the process dies mid-write (crash, disk full, a
resource limit like RLIMIT_FSIZE), the destination is left torn — and if
that destination already held a previous good artefact, that one is gone
too. Every artefact this worker writes (response.json, asr.json,
diarisation.json) goes through this instead: write to a sibling temp
file, flush + fsync it to disk, then atomically rename it over the
destination. `os.replace` is atomic on POSIX and Windows, so a reader
never observes a partial file — either the old content or the fully
written new content, never a mix.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
