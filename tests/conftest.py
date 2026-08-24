"""Pytest bootstrap: must run before any test module imports jake_tools.

See `jake_tools.dyld_fallback`'s module docstring for the full story. In
short: a bare `pytest` invocation never goes through `jake_tools.__main__`,
so without this, a real `torchcodec.decoders.AudioDecoder` import in a
live test (e.g. the plan's own `uv run pytest -q -m live
tests/test_transcription_asr.py` acceptance command) would hit the exact
Homebrew-ffmpeg dlopen failure the `__main__.py` fix targets, on a clean
shell that hasn't already got `/opt/homebrew/lib` on
`DYLD_FALLBACK_LIBRARY_PATH`.

The check runs from a `pytest_configure` hook rather than at this module's
own top level. It has to be a hook, not a bare module-level call, for a
subtle but load-bearing reason: pytest's own stdout/stderr fd-level
capturing starts *before* conftest.py is even imported (inside the same
hook call that imports it), so by the time any of this module's code
runs, fd 1/2 already point at pytest's internal capture target rather
than the real terminal/pipe. `ensure_homebrew_dyld_fallback_path()`'s
`os.execve` replaces the whole process, and the replacement inherits
whatever fd 1/2 currently point at - so calling it at module level makes
the re-exec'd process inherit the *stale* capture target from the
process being replaced, and every line of output (including the final
"N passed" summary) is silently swallowed. Confirmed empirically: with
`DYLD_FALLBACK_LIBRARY_PATH` unset, a module-level call turned `uv run
pytest -q` into zero output and exit 0 (tests genuinely ran and passed -
`-s`, which disables fd capturing entirely, showed them - only the
*reporting* was lost). `pytest_configure` runs after conftest loading but
still well before any test module is collected, and hands us the `config`
object needed to reach the active `CaptureManager` and suspend it around
the exec, so the re-exec'd process inherits the *real* fd 1/2 instead.
"""

from __future__ import annotations

import pytest

from jake_tools.dyld_fallback import ensure_homebrew_dyld_fallback_path


def pytest_configure(config: pytest.Config) -> None:
    capman = config.pluginmanager.get_plugin("capturemanager")
    if capman is not None:
        capman.suspend_global_capture(in_=True)
    try:
        ensure_homebrew_dyld_fallback_path()
    finally:
        # Unreachable if the call above re-exec'd (the process is gone) -
        # only runs on the branch where no restart was needed, restoring
        # normal capturing for the rest of the session.
        if capman is not None:
            capman.resume_global_capture()
