from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .coordinator import LaneRunResult
from .models import LaneName, LaneOutput, LaneSpec
from .paths import DailyReportPaths


class LaneValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    lane: LaneName
    status: str
    errors: list[str] = Field(default_factory=list)
    output: LaneOutput | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


_BANNED_PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("todo", re.compile(r"\btodo\b", re.I)),
    ("tbd", re.compile(r"\btbd\b", re.I)),
    ("lorem ipsum", re.compile(r"\blorem\s+ipsum\b", re.I)),
    (
        "placeholder",
        re.compile(
            r"\b(?:placeholder\s+(?:response|content|text|only)|as\s+a\s+placeholder)\b",
            re.I,
        ),
    ),
    ("insert details", re.compile(r"\binsert\s+details\b", re.I)),
    ("insert findings", re.compile(r"\binsert\s+findings\b", re.I)),
    ("coming soon", re.compile(r"\bcoming\s+soon\b", re.I)),
)

_NO_FINDINGS_RE = re.compile(
    r"\bno\s+(findings?|relevant findings?|items?|matches?|results?)\b", re.I
)
_TOOL_UNAVAILABLE_RE = re.compile(
    r"\b(tool|himalaya|session_search|file)\s+unavailable\b|\bunavailable\s+tool\b",
    re.I,
)
_INBOX_FORBIDDEN_RE = re.compile(
    r"\b(draft(?:ed|s|ing)?\s+(?:a\s+)?(?:reply|response|email)|"
    r"(?:reply|response|email)\s+draft(?:ed)?|"
    r"sent\s+(?:a\s+)?(?:reply|response|email)|"
    r"(?:email|thread|message)\s+(?:resolved|handled|actioned)|"
    r"resolved\s+(?:the\s+)?(?:email|thread|message))\b",
    re.I,
)
_EXACT_CHECK_RE = re.compile(r"\b(search(?:ed)?|check(?:ed)?|query|command)\s*:", re.I)


def validate_lane_artifact(
    spec: LaneSpec,
    paths: DailyReportPaths,
    *,
    session_manifest_path: Path | None = None,
) -> LaneValidationResult:
    errors: list[str] = []
    artefact_path = spec.artefact_path
    output = _load_lane_output(artefact_path, errors)
    if output is None:
        return LaneValidationResult(lane=spec.name, status="fail", errors=errors)

    errors.extend(_missing_required_headings(output.markdown, spec.required_sections))
    errors.extend(_missing_declared_evidence(output.evidence_paths, paths.root))
    errors.extend(_banned_placeholder_errors(output))
    errors.extend(_unsupported_no_findings_errors(output))
    errors.extend(_unsupported_tool_unavailable_errors(output))
    if spec.name is LaneName.INBOX_TRIAGE:
        errors.extend(_inbox_boundary_errors(output))
    errors.extend(
        _invented_session_id_errors(
            output.cited_session_ids,
            session_manifest_path or paths.evidence / "session-manifest.json",
        )
    )

    return LaneValidationResult(
        lane=spec.name,
        status="fail" if errors else "ok",
        errors=errors,
        output=output,
    )


def validate_lane_results(
    specs: list[LaneSpec],
    paths: DailyReportPaths,
) -> dict[LaneName, LaneValidationResult]:
    return {spec.name: validate_lane_artifact(spec, paths) for spec in specs}


def apply_validation_to_lane_result(
    lane_result: LaneRunResult,
    validation: LaneValidationResult,
) -> LaneRunResult:
    if validation.ok:
        return lane_result
    error = "; ".join(validation.errors) or "validation failed"
    return lane_result.model_copy(
        update={"status": "fail", "error": error},
    )


def _load_lane_output(path: Path, errors: list[str]) -> LaneOutput | None:
    if not path.exists():
        errors.append(f"artefact missing: {path}")
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"artefact unreadable: {type(error).__name__}: {error}")
        return None
    if not raw.strip():
        errors.append(f"artefact empty: {path}")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        errors.append(f"artefact is not valid JSON: {error}")
        return None
    try:
        return LaneOutput.model_validate(payload)
    except ValidationError as error:
        errors.append(f"artefact does not parse as LaneOutput: {error}")
        return None


def _missing_required_headings(
    markdown: str, required_sections: tuple[str, ...]
) -> list[str]:
    headings = {_normalise_heading(heading) for heading in _markdown_headings(markdown)}
    return [
        f"required heading missing: {section}"
        for section in required_sections
        if _normalise_heading(section) not in headings
    ]


def _markdown_headings(markdown: str) -> list[str]:
    headings: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            headings.append(match.group(1).strip())
    return headings


def _normalise_heading(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _missing_declared_evidence(evidence_paths: list[str], root: Path) -> list[str]:
    errors: list[str] = []
    for declared in evidence_paths:
        evidence_path = Path(declared)
        if not evidence_path.is_absolute():
            evidence_path = root / evidence_path
        if not evidence_path.exists():
            errors.append(f"declared evidence missing: {declared}")
    return errors


def _banned_placeholder_errors(output: LaneOutput) -> list[str]:
    text = _combined_text(output)
    return [
        f"banned placeholder phrase present: {phrase}"
        for phrase, pattern in _BANNED_PLACEHOLDER_PATTERNS
        if pattern.search(text)
    ]


def _unsupported_no_findings_errors(output: LaneOutput) -> list[str]:
    text = _combined_text(output)
    if not _NO_FINDINGS_RE.search(text):
        return []
    if _EXACT_CHECK_RE.search(text):
        return []
    return ["'no findings' claim lacks exact search/check citation"]


def _unsupported_tool_unavailable_errors(output: LaneOutput) -> list[str]:
    text = _combined_text(output)
    if not _TOOL_UNAVAILABLE_RE.search(text):
        return []
    if "preflight" in text.lower() or any(
        "preflight" in Path(path).name.lower() for path in output.evidence_paths
    ):
        return []
    return ["'tool unavailable' claim lacks preflight citation"]


def _inbox_boundary_errors(output: LaneOutput) -> list[str]:
    if _INBOX_FORBIDDEN_RE.search(_combined_text(output)):
        return ["inbox lane drafts replies or implies email resolution"]
    return []


def _invented_session_id_errors(
    cited_session_ids: list[str], manifest_path: Path
) -> list[str]:
    if not cited_session_ids:
        return []
    if not manifest_path.exists():
        return [f"session manifest missing for cited_session_id check: {manifest_path}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [
            f"session manifest unreadable for cited_session_id check: {type(error).__name__}: {error}"
        ]
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        return ["session manifest has no sessions list"]
    known_ids = {session.get("id") for session in sessions if isinstance(session, dict)}
    return [
        f"cited_session_id not present in session manifest: {session_id}"
        for session_id in cited_session_ids
        if session_id not in known_ids
    ]


def _combined_text(output: LaneOutput) -> str:
    return "\n".join(
        [
            output.markdown,
            *output.findings,
            *output.actions,
            *output.caveats,
        ]
    )
