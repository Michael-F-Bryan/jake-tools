from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from .models import (
    ChapterPlan,
    TranscriptArtifact,
    VerificationCheck,
    VerificationReport,
)

NoteVerificationProfile = Literal["default", "dumc", "source"]

_OPERATIONAL_CHATTER_RE = re.compile(
    r"^\s*Loading the transcript-polisher skill\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _build_report(
    checks: list[VerificationCheck], *, affected_paths: list[Path]
) -> VerificationReport:
    failed_gate_ids = [
        check.check_id for check in checks if check.status in {"fail", "blocked"}
    ]
    status: str = "pass"
    if failed_gate_ids:
        status = "fail"
    return VerificationReport(
        status=status,
        checks=checks,
        failed_gate_ids=failed_gate_ids,
        affected_artifact_paths=affected_paths,
        next_action="Fix failing checks and rerun verification."
        if failed_gate_ids
        else None,
    )


def verify_turns(
    before: TranscriptArtifact, after: TranscriptArtifact, *, affected_paths: list[Path]
) -> VerificationReport:
    checks: list[VerificationCheck] = []
    checks.append(
        VerificationCheck(
            check_id="turns.non-empty",
            status="pass" if after.turns else "fail",
            message="Transformed transcript still has at least one turn.",
        )
    )

    monotonic = all(
        turn.start <= turn.end
        and (index == 0 or after.turns[index - 1].start <= turn.start)
        for index, turn in enumerate(after.turns)
    )
    checks.append(
        VerificationCheck(
            check_id="turns.monotonic-order",
            status="pass" if monotonic else "fail",
            message="Turn timestamps are monotonic and internally valid.",
        )
    )

    coverage_preserved = bool(before.turns and after.turns)
    if before.turns and after.turns:
        coverage_preserved = (
            after.turns[0].start <= before.turns[0].start + 0.001
            and after.turns[-1].end + 0.001 >= before.turns[-1].end
        )
    checks.append(
        VerificationCheck(
            check_id="turns.coverage-preserved",
            status="pass" if coverage_preserved else "fail",
            message="Transcript time-span coverage is preserved after transform.",
        )
    )

    before_speakers = {turn.speaker for turn in before.turns}
    after_speakers = {turn.speaker for turn in after.turns}
    checks.append(
        VerificationCheck(
            check_id="turns.speakers-preserved",
            status="pass" if before_speakers.issubset(after_speakers) else "fail",
            message="All speakers from input transcript remain represented.",
        )
    )

    overlapping_duplicate = any(
        previous.speaker == current.speaker
        and previous.text.strip().casefold() == current.text.strip().casefold()
        and current.start < previous.end
        for previous, current in zip(after.turns, after.turns[1:], strict=False)
    )
    checks.append(
        VerificationCheck(
            check_id="turns.no-adjacent-duplicates",
            status="fail" if overlapping_duplicate else "pass",
            message="Transcript does not contain overlapping duplicate turns.",
        )
    )

    return _build_report(checks, affected_paths=affected_paths)


def verify_chapters(
    chapter_plan: ChapterPlan,
    *,
    transcript: TranscriptArtifact | None,
    affected_paths: list[Path],
) -> VerificationReport:
    chapters = chapter_plan.chapters
    checks: list[VerificationCheck] = []
    checks.append(
        VerificationCheck(
            check_id="chapters.non-empty",
            status="pass" if chapters else "fail",
            message="Chapter plan includes at least one chapter.",
        )
    )

    ordered = all(
        chapter.start <= chapter.end
        and (index == 0 or chapters[index - 1].start <= chapter.start)
        for index, chapter in enumerate(chapters)
    )
    checks.append(
        VerificationCheck(
            check_id="chapters.monotonic-order",
            status="pass" if ordered else "fail",
            message="Chapter boundaries are monotonic and valid.",
        )
    )

    non_overlapping = all(
        index == 0 or chapters[index - 1].end <= chapter.start
        for index, chapter in enumerate(chapters)
    )
    checks.append(
        VerificationCheck(
            check_id="chapters.non-overlapping",
            status="pass" if non_overlapping else "fail",
            message="Chapters do not overlap in time.",
        )
    )

    covers_transcript = True
    if transcript and transcript.turns and chapters:
        covers_transcript = (
            chapters[0].start <= transcript.turns[0].start + 0.001
            and chapters[-1].end + 0.001 >= transcript.turns[-1].end
        )
    checks.append(
        VerificationCheck(
            check_id="chapters.covers-transcript-span",
            status="pass" if covers_transcript else "fail",
            message="Chapter range covers the transcript span.",
        )
    )
    return _build_report(checks, affected_paths=affected_paths)


def verify_note(
    note_body: str,
    *,
    expected_chapter_count: int | None,
    affected_path: Path,
    profile: NoteVerificationProfile = "default",
) -> VerificationReport:
    if profile == "dumc":
        return _verify_dumc_note(
            note_body,
            expected_chapter_count=expected_chapter_count,
            affected_path=affected_path,
        )
    if profile == "source":
        return _verify_source_note(note_body, affected_path=affected_path)
    if profile != "default":
        raise ValueError(f"unknown note verification profile: {profile}")

    checks = [
        VerificationCheck(
            check_id="note.has-meeting-notes",
            status="pass" if "## Meeting Notes" in note_body else "fail",
            message="Note contains a ## Meeting Notes section.",
        ),
        VerificationCheck(
            check_id="note.has-chapters",
            status="pass" if "## Chapters" in note_body else "fail",
            message="Note contains a ## Chapters section.",
        ),
        VerificationCheck(
            check_id="note.has-transcript",
            status="pass" if "## Transcript" in note_body else "fail",
            message="Note contains a ## Transcript section.",
        ),
        VerificationCheck(
            check_id="note.chapter-heading-count",
            status="pass"
            if (
                expected_chapter_count is None
                or note_body.count("### ") == expected_chapter_count
            )
            else "fail",
            message="Chapter heading count matches expected chapter count.",
        ),
        VerificationCheck(
            check_id="note.no-operational-chatter",
            status="fail" if _OPERATIONAL_CHATTER_RE.search(note_body) else "pass",
            message="Note does not contain transcript-polisher operational chatter.",
        ),
    ]
    return _build_report(checks, affected_paths=[affected_path])


def _verify_source_note(
    note_body: str,
    *,
    affected_path: Path,
) -> VerificationReport:
    provenance_fields = (
        "title:",
        "source:",
        "channel:",
        "published:",
        "duration:",
        "video-id:",
        "subtitle-track:",
        "subtitle-kind:",
        "capture-method:",
    )
    provenance_ok = True
    for field in provenance_fields:
        match = re.search(
            rf"^{re.escape(field)}\s*(?P<value>.*)$", note_body, re.MULTILINE
        )
        if match is None or not match.group("value").strip().strip("\"'"):
            provenance_ok = False
            break
    caption_artifacts = re.search(
        r"(?:^WEBVTT\s*$|\s-->\s|<c(?:\s|>)|<\d{2}:\d{2}:\d{2}\.\d{3}>)",
        note_body,
        re.IGNORECASE | re.MULTILINE,
    )
    checks = [
        VerificationCheck(
            check_id="note.has-required-provenance",
            status="pass" if provenance_ok else "fail",
            message="Source note records the required video and caption provenance.",
        ),
        VerificationCheck(
            check_id="note.no-caption-artifacts",
            status="fail" if caption_artifacts else "pass",
            message="Source note does not contain raw caption artefacts.",
        ),
        VerificationCheck(
            check_id="note.no-operational-chatter",
            status="fail" if _OPERATIONAL_CHATTER_RE.search(note_body) else "pass",
            message="Source note does not contain operational chatter.",
        ),
    ]
    return _build_report(checks, affected_paths=[affected_path])


def _heading_index(note_body: str, heading: str) -> int | None:
    index = note_body.find(heading)
    return index if index >= 0 else None


def _verify_dumc_note(
    note_body: str,
    *,
    expected_chapter_count: int | None,
    affected_path: Path,
) -> VerificationReport:
    summary_index = _heading_index(note_body, "> [!summary]")
    discussion_index = _heading_index(note_body, "## Discussion Notes")
    checks = [
        VerificationCheck(
            check_id="note.has-summary-callout",
            status="pass"
            if summary_index is not None
            and discussion_index is not None
            and summary_index < discussion_index
            else "fail",
            message="DUM-C note has a summary callout before Discussion Notes.",
        ),
        VerificationCheck(
            check_id="note.has-discussion-notes",
            status="pass" if "## Discussion Notes" in note_body else "fail",
            message="DUM-C note contains a ## Discussion Notes section.",
        ),
        VerificationCheck(
            check_id="note.has-chapters",
            status="pass" if "## Chapters" in note_body else "fail",
            message="DUM-C note contains a ## Chapters section.",
        ),
        VerificationCheck(
            check_id="note.has-transcript",
            status="pass" if "## Transcript" in note_body else "fail",
            message="DUM-C note contains a ## Transcript section.",
        ),
        VerificationCheck(
            check_id="note.chapter-heading-count",
            status="pass"
            if (
                expected_chapter_count is None
                or note_body.count("### ") == expected_chapter_count
            )
            else "fail",
            message="Chapter heading count matches expected chapter count.",
        ),
        VerificationCheck(
            check_id="note.no-meeting-notes",
            status="fail" if "## Meeting Notes" in note_body else "pass",
            message="DUM-C note does not use the generic ## Meeting Notes heading.",
        ),
        VerificationCheck(
            check_id="note.no-notes-heading",
            status="fail"
            if re.search(r"^## Notes\s*$", note_body, re.MULTILINE)
            else "pass",
            message="DUM-C note does not include a generic ## Notes heading.",
        ),
        VerificationCheck(
            check_id="note.no-next-steps",
            status="fail" if "## Next Steps" in note_body else "pass",
            message="DUM-C note does not include an assigned next-steps section.",
        ),
        VerificationCheck(
            check_id="note.no-task-checkboxes",
            status="fail"
            if re.search(r"^\s*- \[[ xX]\]", note_body, re.MULTILINE)
            else "pass",
            message="DUM-C note does not contain task checkboxes.",
        ),
        VerificationCheck(
            check_id="note.no-action-labels",
            status="fail" if re.search(r"\bAction:\s*", note_body) else "pass",
            message="DUM-C note does not contain Action: labels.",
        ),
        VerificationCheck(
            check_id="note.no-raw-vtt-link",
            status="fail"
            if re.search(r"\.vtt\b", note_body, re.IGNORECASE)
            else "pass",
            message="DUM-C note does not link raw VTT by default.",
        ),
        VerificationCheck(
            check_id="note.no-operational-chatter",
            status="fail" if _OPERATIONAL_CHATTER_RE.search(note_body) else "pass",
            message="Note does not contain transcript-polisher operational chatter.",
        ),
    ]
    return _build_report(checks, affected_paths=[affected_path])
