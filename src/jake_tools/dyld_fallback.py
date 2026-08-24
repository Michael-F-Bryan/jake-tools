"""Homebrew dyld fallback path: shared between the CLI entrypoint and tests.

`torchcodec` (pulled in transitively by `pyannote-audio`, plan 004's
diarisation dependency) dlopens Homebrew's ffmpeg shared libraries
(`libavutil.*.dylib` etc.) by a bare `@rpath` reference. On Apple Silicon,
`/opt/homebrew/lib` isn't on macOS's default dyld search path, so that
dlopen fails on this uv-managed Python even though the libraries are right
there — this was the exact failure that stopped plan 004's real-meeting
acceptance run from completing diarisation.

dyld only consults `DYLD_FALLBACK_LIBRARY_PATH` at process launch, so
setting it via `os.environ` after Python has already started (e.g. lazily,
right before the diarisation pipeline runs) has no effect — this has to
happen before anything that might eventually import `torch`/`torchcodec`
gets a chance to dlopen against it.

**Two call sites need it, independently**, because a bare `pytest`
invocation never goes through `jake_tools.__main__` at all:

- `src/jake_tools/__main__.py` calls `ensure_homebrew_dyld_fallback_path()`
  first, before even `dotenv`/the CLI package are imported — covers the
  `jake-tools`/`python -m jake_tools` entrypoints.
- `tests/conftest.py` calls it too, before pytest collects any test
  module — covers a direct `pytest ...` invocation, including the plan's
  own `uv run pytest -q -m live tests/test_transcription_asr.py`
  acceptance command, and any other direct import of the transcription
  library (a notebook, a script, an agent invoking the library API
  without going through the CLI). `conftest.py` is pytest's own earliest
  collection hook — it's imported before any test module in its
  directory — so it's the earliest point available to react, short of a
  wrapper script around `pytest` itself.

Calling it from two places never double-restarts a single process: each
call re-execs (via `sys.orig_argv`, the exact original command line) only
if the fallback path isn't already set, then returns immediately once it
is — so whichever call site runs first does the one restart, and the
second call site (if reached at all, post-restart) is a no-op.
"""

from __future__ import annotations

import os
import sys

_HOMEBREW_LIB = "/opt/homebrew/lib"


def ensure_homebrew_dyld_fallback_path() -> None:
    """Restart this process once with Homebrew's lib dir on the dyld search
    path, if it isn't already there. See the module docstring for why this
    exists and why it must be called from every entrypoint that might
    import `torch`/`torchcodec`, not just `jake_tools.__main__`.

    `sys.orig_argv` (the exact original command line, however this
    process was launched — a console script, `python -m jake_tools`,
    `pytest`, ...) makes the restart exact.
    """
    if sys.platform != "darwin" or not os.path.isdir(_HOMEBREW_LIB):
        return  # not Apple Silicon Homebrew: nothing to work around
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if _HOMEBREW_LIB in current.split(os.pathsep):
        return  # already set — either a previous restart, or the caller's shell
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
        f"{_HOMEBREW_LIB}{os.pathsep}{current}" if current else _HOMEBREW_LIB
    )
    os.execve(sys.orig_argv[0], sys.orig_argv, os.environ)
