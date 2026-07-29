from __future__ import annotations

import json
from pathlib import Path

from .audit_models import CalibrationCase
from .models import ExtractResult

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_CASES_PATH = _PACKAGE_DIR / "assets" / "calibration-cases.json"
# Calibration replay transcripts are large fixture content that lives with the
# test suite (this repo checkout) rather than shipping as installed package data.
CALIBRATION_FIXTURES_DIR = (
    _PACKAGE_DIR.parents[2] / "tests" / "fixtures" / "ai_watch" / "calibration"
)
MIN_CALIBRATION_CONTENT_CHARS = 500


class CalibrationCasesNotFoundError(FileNotFoundError):
    """Raised when the requested calibration cases file does not exist."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Calibration cases file not found: {path}. "
            "--calibration-only requires a real cases file; it cannot run "
            "silently against an empty case list."
        )
        self.path = path


def load_calibration_cases(
    fixtures_path: Path | None = None,
) -> list[CalibrationCase]:
    path = fixtures_path or DEFAULT_CALIBRATION_CASES_PATH
    if not path.exists():
        raise CalibrationCasesNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [CalibrationCase.model_validate(row) for row in payload]


def calibration_cases_by_url(
    cases: list[CalibrationCase],
) -> dict[str, CalibrationCase]:
    return {case.url.strip().rstrip("/").lower(): case for case in cases}


def calibration_fixture_path(case_id: str) -> Path:
    return CALIBRATION_FIXTURES_DIR / f"{case_id}.md"


def load_calibration_fixture(
    case: CalibrationCase,
) -> ExtractResult | None:
    path = calibration_fixture_path(case.id)
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8").strip()
    if len(content) < MIN_CALIBRATION_CONTENT_CHARS:
        return None
    return ExtractResult(
        url=case.url,
        title=case.title_hint,
        content=content,
    )


def resolve_calibration_extract(
    *,
    url: str,
    title: str,
    extracted: ExtractResult,
    calibration_cases: list[CalibrationCase],
) -> ExtractResult:
    case = calibration_cases_by_url(calibration_cases).get(
        url.strip().rstrip("/").lower()
    )
    if case is None:
        return extracted

    fixture = load_calibration_fixture(case)
    if fixture is None:
        return extracted

    content = extracted.content.strip()
    if extracted.status != "ok" or len(content) < MIN_CALIBRATION_CONTENT_CHARS:
        return fixture
    return extracted
