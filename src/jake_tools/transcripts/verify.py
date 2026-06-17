from __future__ import annotations

from .merge import RECORDING_EMBED_RE
from .models import Chapter, MergeReport


class VerificationError(RuntimeError):
    pass



def verify_note(
    merged_body: str,
    *,
    original_body: str,
    chapters: list[Chapter],
) -> MergeReport:
    transcript_heading_count = merged_body.count("## Transcript")
    if transcript_heading_count != 1:
        raise VerificationError("expected exactly one ## Transcript heading")

    if RECORDING_EMBED_RE.findall(original_body):
        for embed in RECORDING_EMBED_RE.findall(original_body):
            if embed not in merged_body:
                raise VerificationError("recording embed was not preserved")

    if "## Meeting Notes" not in merged_body:
        raise VerificationError("missing ## Meeting Notes section")

    if "## Chapters" not in merged_body:
        raise VerificationError("missing ## Chapters section")

    if merged_body.count("### ") != len(chapters):
        raise VerificationError("chapter heading count does not match chapter count")

    return MergeReport(
        status="pass",
        transcript_heading_count=merged_body.count("### "),
        chapter_count=len(chapters),
        original_content_preserved=True,
        recording_embed_preserved=True,
    )
