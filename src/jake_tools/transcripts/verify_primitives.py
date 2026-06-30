from __future__ import annotations

import re
from pathlib import Path

from .models import (
    ChapterPlan,
    SourceArtifact,
    TranscriptArtifact,
    VerificationCheck,
    VerificationReport,
)

BOILERPLATE_CHECK_IDS: tuple[str, ...] = (
    "boilerplate.no-operational-chatter",
    "boilerplate.no-markdown-fences",
)
TURNS_CHECK_IDS: tuple[str, ...] = (
    "turns.non-empty",
    "turns.monotonic-order",
    "turns.coverage-preserved",
    "turns.speakers-preserved",
    "turns.no-adjacent-duplicates",
)
CHAPTERS_CHECK_IDS: tuple[str, ...] = (
    "chapters.non-empty",
    "chapters.monotonic-order",
    "chapters.non-overlapping",
    "chapters.covers-transcript-span",
)
NOTE_CHECK_IDS: tuple[str, ...] = (
    "note.has-meeting-notes",
    "note.has-chapters",
    "note.has-transcript",
    "note.chapter-heading-count",
    "note.no-operational-chatter",
)

_OPERATIONAL_CHATTER_RE = re.compile(
    r"^\s*Loading the transcript-polisher skill\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MARKDOWN_FENCE_RE = re.compile(r"^\s*```{3,}", re.MULTILINE)


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


def verify_boilerplate_text(text: str, *, affected_path: Path) -> VerificationReport:
    checks = [
        VerificationCheck(
            check_id="boilerplate.no-operational-chatter",
            status="fail" if _OPERATIONAL_CHATTER_RE.search(text) else "pass",
            message="Transcript output does not contain operational chatter.",
        ),
        VerificationCheck(
            check_id="boilerplate.no-markdown-fences",
            status="fail" if _MARKDOWN_FENCE_RE.search(text) else "pass",
            message="Transcript output does not include markdown fence markers.",
        ),
    ]
    return _build_report(checks, affected_paths=[affected_path])


def read_source_text_for_verification(source: SourceArtifact) -> tuple[Path, str]:
    source_text_path = source.raw_text_path or source.source_path
    if source_text_path is None:
        raise ValueError("SourceArtifact is missing raw_text_path and source_path.")
    return source_text_path, source_text_path.read_text(encoding="utf-8")


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

    adjacent_duplicate = any(
        previous.speaker == current.speaker
        and previous.text.strip().casefold() == current.text.strip().casefold()
        for previous, current in zip(after.turns, after.turns[1:], strict=False)
    )
    checks.append(
        VerificationCheck(
            check_id="turns.no-adjacent-duplicates",
            status="fail" if adjacent_duplicate else "pass",
            message="Transcript does not contain adjacent duplicate turns with the same speaker and text.",
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
) -> VerificationReport:
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
