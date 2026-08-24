import os
import sys

_HOMEBREW_LIB = "/opt/homebrew/lib"


def _ensure_homebrew_dyld_fallback_path() -> None:
    """Restart this process once with Homebrew's lib dir on the dyld search
    path, if it isn't already there.

    `torchcodec` (pulled in transitively by `pyannote-audio`, plan 004's
    diarisation dependency) dlopens Homebrew's ffmpeg shared libraries
    (`libavutil.*.dylib` etc.) by a bare `@rpath` reference. On Apple
    Silicon, `/opt/homebrew/lib` isn't on macOS's default dyld search path,
    so that dlopen fails on this uv-managed Python even though the
    libraries are right there — this was the exact failure that stopped
    plan 004's real-meeting acceptance run from completing diarisation.

    dyld only consults `DYLD_FALLBACK_LIBRARY_PATH` at process launch, so
    setting it via `os.environ` after Python has already started (e.g.
    lazily, right before the diarisation pipeline runs) has no effect —
    this has to happen before anything that might eventually dlopen
    against it gets a chance to, hence it runs first, in `__main__`, before
    even `dotenv`/the CLI package are imported. `sys.orig_argv` (the exact
    original command line, however this process was launched — a console
    script, `python -m jake_tools`, ...) makes the restart exact.
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


_ensure_homebrew_dyld_fallback_path()

import dotenv  # noqa: E402

# Must be executed before importing the CLI
dotenv.load_dotenv()

from .cli import main  # noqa: E402

if __name__ == "__main__":
    main()
