from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from .audit_models import RunManifest
from .records import load_model

_SINCE_UNIT_DAYS = {"d": 1, "w": 7}


class InvalidSinceError(ValueError):
    """Raised when a --since value isn't a supported "<N><unit>" duration."""

    def __init__(self, since: str) -> None:
        super().__init__(
            f"invalid --since value {since!r}: expected a number followed by "
            f"'d' or 'w' (e.g. 7d, 2w)"
        )
        self.since = since


def parse_since_days(since: str) -> int:
    """Parse a lookback window like "7d" or "2w" into a day count."""
    since = since.strip()
    if len(since) < 2:
        raise InvalidSinceError(since)
    multiplier = _SINCE_UNIT_DAYS.get(since[-1])
    if multiplier is None:
        raise InvalidSinceError(since)
    try:
        count = int(since[:-1])
    except ValueError as error:
        raise InvalidSinceError(since) from error
    return count * multiplier


def audit_cutoff(*, since: str, today: date) -> date:
    return today - timedelta(days=parse_since_days(since))


def find_runs(base_dir: Path, *, cutoff: date) -> list[RunManifest]:
    """Load the manifest of every ai-watch run on or after `cutoff`."""
    watch_root = base_dir / "ai-watch"
    if not watch_root.exists():
        return []
    manifests: list[RunManifest] = []
    for run_dir in sorted(watch_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "state":
            continue
        try:
            run_date = date.fromisoformat(run_dir.name)
        except ValueError:
            continue
        if run_date < cutoff:
            continue
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifests.append(load_model(manifest_path, RunManifest))
    return manifests
