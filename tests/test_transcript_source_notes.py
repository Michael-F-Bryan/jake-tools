from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from jake_tools.claude import Reply
from jake_tools.prompting import StructuredPrompt
from jake_tools.transcripts.models import (
    ChapterPlan,
    PlannedChapter,
    SourceArtifact,
    SourceNotePlan,
    SourceOverview,
    TranscriptArtifact,
    TranscriptTurn,
)
from jake_tools.transcripts.render_primitives import (
    _source_chapter_paragraphs,
    render_source_note_markdown,
)
from jake_tools.transcripts.stage_primitives import (
    StagePrimitiveError,
    run_source_note_plan_stage,
)
from jake_tools.transcripts.verify_primitives import verify_note


class FakeAgent:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.prompt: Any | None = None

    async def run_structured[TModel: BaseModel](
        self, prompt: StructuredPrompt[TModel]
    ) -> tuple[TModel, Reply]:
        self.prompt = prompt
        result = prompt.response_model.model_validate(self.response)
        return cast(TModel, result), Reply(text=json.dumps(self.response))


def _source(tmp_path: Path) -> SourceArtifact:
    return SourceArtifact(
        kind="youtube",
        source_path=tmp_path / "video-info.json",
        source_url="https://www.youtube.com/watch?v=video-123",
        message_id="youtube:video-123",
        title="MAVLink tools - Patrick Pereira, Blue Robotics",
        date=dt.date(2025, 11, 28),
        organisation="PX4 Autopilot",
        raw_text_path=tmp_path / "captions.en-orig.json3",
        metadata={
            "video_id": "video-123",
            "channel": "PX4 Autopilot",
            "duration_seconds": 620,
            "subtitle_track": "en-orig",
            "subtitle_kind": "automatic",
            "capture_method": "yt-dlp",
        },
    )


def _transcript() -> TranscriptArtifact:
    return TranscriptArtifact(
        turns=[
            TranscriptTurn(
                start=0.0,
                end=20.0,
                speaker="Speaker",
                text="MAVLink tools make vehicle integration easier.",
            ),
            TranscriptTurn(
                start=310.0,
                end=330.0,
                speaker="Speaker",
                text="The inspector exposes messages in a browser.",
            ),
        ]
    )


def _chapters() -> ChapterPlan:
    return ChapterPlan(
        boundary_source="deterministic",
        chapters=[
            PlannedChapter(
                start=0.0,
                end=300.0,
                title="Why MAVLink tooling matters",
                summary="The talk frames common integration and observability problems.",
            ),
            PlannedChapter(
                start=310.0,
                end=330.0,
                title="Browser-based inspection",
                summary="The inspector makes live messages visible without custom software.",
            ),
        ],
    )


async def test_source_note_plan_stage_returns_faithful_structured_notes(
    tmp_path: Path,
) -> None:
    agent = FakeAgent(
        {
            "overview": {
                "summary": "Patrick Pereira presents MAVLink tools for vehicle integration and browser-based inspection.",
                "key_points": [
                    "The tools reduce custom integration work.",
                    "Browser inspection exposes live MAVLink messages.",
                ],
            },
            "chapters": _chapters().model_dump(mode="json"),
        }
    )

    plan, _reply = await run_source_note_plan_stage(
        agent,
        _source(tmp_path),
        _transcript(),
        draft_plan=_chapters(),
        max_attempts=1,
    )

    assert isinstance(plan, SourceNotePlan)
    assert plan.overview.key_points == [
        "The tools reduce custom integration work.",
        "Browser inspection exposes live MAVLink messages.",
    ]
    assert plan.chapters == _chapters()
    assert agent.prompt is not None
    assert agent.prompt.source["title"].startswith("MAVLink tools")
    assert len(agent.prompt.turns) == 2


async def test_source_note_plan_rejects_changed_chapter_boundaries(
    tmp_path: Path,
) -> None:
    chapters = _chapters().model_dump(mode="json")
    chapters["chapters"][0]["start"] = 1.0
    agent = FakeAgent(
        {
            "overview": {"summary": "Summary.", "key_points": ["Point."]},
            "chapters": chapters,
        }
    )

    with pytest.raises(StagePrimitiveError, match="chapter boundaries"):
        await run_source_note_plan_stage(
            agent,
            _source(tmp_path),
            _transcript(),
            draft_plan=_chapters(),
            max_attempts=1,
        )


def test_source_chapter_paragraphs_do_not_split_mid_sentence() -> None:
    turns = [
        TranscriptTurn(
            start=0.0,
            end=2.0,
            speaker="Speaker",
            text="This sentence contains " + ("detail " * 30) + "and",
        ),
        TranscriptTurn(
            start=2.0,
            end=4.0,
            speaker="Speaker",
            text="continues across a caption turn. A second sentence follows.",
        ),
    ]

    rendered = _source_chapter_paragraphs(turns, target_chars=100)

    assert "and continues across a caption turn." in rendered
    assert "and\n\ncontinues" not in rendered


def test_render_source_note_includes_provenance_and_timestamp_links(
    tmp_path: Path,
) -> None:
    overview = SourceOverview(
        summary="A practical survey of MAVLink development tools.",
        key_points=["The browser inspector exposes live messages."],
    )

    note, sections = render_source_note_markdown(
        _source(tmp_path),
        overview,
        _transcript(),
        _chapters(),
    )

    assert sections == ["summary", "key-points", "chapters", "transcript"]
    assert 'source: "https://www.youtube.com/watch?v=video-123"' in note
    assert 'channel: "PX4 Autopilot"' in note
    assert "published: 2025-11-28" in note
    assert "duration: 620" in note
    assert 'video-id: "video-123"' in note
    assert 'subtitle-track: "en-orig"' in note
    assert 'subtitle-kind: "automatic"' in note
    assert "> [!summary]" in note
    assert "## Key points" in note
    assert "## Chapters" in note
    assert "## Transcript" in note
    assert note.count("https://www.youtube.com/watch?v=video-123&t=0s") == 2
    assert note.count("https://www.youtube.com/watch?v=video-123&t=310s") == 2
    assert "**Speaker**" not in note
    assert "The inspector exposes messages in a browser." in note


def test_verify_source_note_accepts_rendered_shape(tmp_path: Path) -> None:
    note, _sections = render_source_note_markdown(
        _source(tmp_path),
        SourceOverview(summary="Summary.", key_points=["Point."]),
        _transcript(),
        _chapters(),
    )

    report = verify_note(
        note,
        expected_chapter_count=2,
        affected_path=tmp_path / "note.md",
        profile="source",
    )

    assert report.status == "pass"
    assert report.failed_gate_ids == []


def test_verify_source_note_rejects_blank_provenance(tmp_path: Path) -> None:
    note, _sections = render_source_note_markdown(
        _source(tmp_path),
        SourceOverview(summary="Summary.", key_points=["Point."]),
        _transcript(),
        _chapters(),
    )
    note = note.replace('channel: "PX4 Autopilot"', 'channel: ""')

    report = verify_note(
        note,
        expected_chapter_count=2,
        affected_path=tmp_path / "note.md",
        profile="source",
    )

    assert "note.has-required-provenance" in report.failed_gate_ids


def test_verify_source_note_rejects_caption_artifacts_and_missing_source(
    tmp_path: Path,
) -> None:
    note = """---
title: Broken
---

> [!summary]
> Summary.

## Key points
- Point.

## Chapters
- 00:00 — Opening

## Transcript

### 00:00 — Opening
WEBVTT
00:00:01.000 --> 00:00:02.000
<c>raw caption</c>
"""

    report = verify_note(
        note,
        expected_chapter_count=1,
        affected_path=tmp_path / "note.md",
        profile="source",
    )

    assert "note.has-required-provenance" in report.failed_gate_ids
    assert "note.no-caption-artifacts" in report.failed_gate_ids
