"""Test-only import of the REAL ``inference_worker.models`` (M11).

``inference-worker`` is a standalone project, pinned to its own Python
version, never installed into jake-tools' own virtualenv (see
``bundle/worker_contract.py``'s module docstring for why production code
mirrors the wire schema instead of importing it live). Tests are
different: a fake worker built here should construct its response
through the *actual* frozen contract, so a schema change there breaks
this test suite immediately rather than silently drifting out of sync
with ``bundle/worker_contract.py``'s mirrored types. ``inference_worker.
models`` itself only imports ``pydantic`` and the stdlib at module level
(no ML dependency), so this sys.path addition is safe under jake-tools'
own Python 3.14 interpreter without installing anything.

Not a ``test_*.py`` module -- pytest does not collect it; it exists
purely to be imported by ones that do (matching ``fixtures_bundle.py``'s
own pattern).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_WORKER_SRC = _REPO_ROOT / "inference-worker" / "src"
if str(_INFERENCE_WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_WORKER_SRC))

import inference_worker.models as models  # noqa: E402  # pyright: ignore[reportMissingImports]

__all__ = ["models"]
