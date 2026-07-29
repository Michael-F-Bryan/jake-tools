from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jake_tools.transcripts.models import (
    Chapter,
    ChapterPlan,
    SourceArtifact,
    TeamsMeetingResult,
    YoutubeCaptureMetadata,
    YoutubeSourceNoteResult,
)


def test_source_artifact_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        SourceArtifact(kind="bogus")  # type: ignore[arg-type]


def test_source_artifact_accepts_every_known_kind() -> None:
    for kind in ("obsidian-note", "youtube", "msgraph-teams"):
        assert SourceArtifact(kind=kind).kind == kind


def test_youtube_capture_metadata_requires_video_id_and_subtitle_fields() -> None:
    with pytest.raises(ValidationError):
        YoutubeCaptureMetadata(channel="PX4 Autopilot")  # type: ignore[call-arg]


def test_youtube_capture_metadata_round_trips_through_source_artifact_metadata() -> (
    None
):
    capture = YoutubeCaptureMetadata(
        video_id="video-123",
        channel="PX4 Autopilot",
        subtitle_track="en",
        subtitle_kind="manual",
    )
    source = SourceArtifact(kind="youtube", metadata=capture.model_dump(mode="json"))

    round_tripped = YoutubeCaptureMetadata.model_validate(source.metadata)

    assert round_tripped == capture


def test_chapter_plan_chapters_is_a_plain_list_of_chapter() -> None:
    plan = ChapterPlan(
        chapters=[Chapter(title="Kickoff", start=0, end=30, summary="Start")],
        boundary_source="llm",
    )

    assert isinstance(plan.chapters[0], Chapter)


def test_youtube_source_note_result_json_dump_keeps_stable_keys() -> None:
    result = YoutubeSourceNoteResult(
        title="A video",
        note=Path("/vault/note.md"),
        updated=True,
        out_dir=Path("/tmp/run"),
        rendered_note=Path("/tmp/run/source-note.md"),
        manifest=Path("/tmp/run/manifest.json"),
        verification_status="pass",
        subtitle_track="en",
        subtitle_kind="manual",
    )

    dumped = result.model_dump(mode="json")

    assert set(dumped) == {
        "title",
        "note",
        "updated",
        "out_dir",
        "rendered_note",
        "manifest",
        "verification_status",
        "subtitle_track",
        "subtitle_kind",
    }
    assert dumped["note"] == "/vault/note.md"


def test_teams_meeting_result_json_dump_keeps_stable_keys() -> None:
    result = TeamsMeetingResult(
        note=None,
        updated=False,
        out_dir=Path("/tmp/run"),
        rendered_note=Path("/tmp/run/rendered-note.md"),
        raw_vtt=Path("/tmp/run/transcript.vtt"),
        manifest=Path("/tmp/run/manifest.json"),
    )

    dumped = result.model_dump(mode="json")

    assert set(dumped) == {
        "note",
        "updated",
        "out_dir",
        "rendered_note",
        "raw_vtt",
        "manifest",
    }
    assert dumped["note"] is None
