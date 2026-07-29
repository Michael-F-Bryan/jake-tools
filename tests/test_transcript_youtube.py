from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from jake_tools.transcripts.models import SourceArtifact
from jake_tools.transcripts.parse import parse_youtube_json3
from jake_tools.transcripts.sources import (
    SourcePrimitiveError,
    source_from_youtube,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "transcript"


def _youtube_info(
    *, subtitles: dict | None = None, automatic: dict | None = None
) -> dict:
    return {
        "id": "video-123",
        "title": "MAVLink tools - Patrick Pereira, Blue Robotics",
        "channel": "PX4 Autopilot",
        "channel_id": "channel-123",
        "uploader": "PX4 Autopilot",
        "upload_date": "20251128",
        "duration": 1905,
        "webpage_url": "https://www.youtube.com/watch?v=video-123",
        "language": "en-US",
        "subtitles": subtitles or {},
        "automatic_captions": automatic or {},
    }


def _fake_ytdlp_runner(
    info: dict,
    *,
    caption_fixture: Path = FIXTURES_DIR / "youtube-sample.json3",
):
    calls: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "--dump-single-json" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(info), stderr=""
            )

        output_template = Path(command[command.index("--output") + 1])
        language = command[command.index("--sub-langs") + 1]
        caption_path = Path(
            str(output_template).replace("%(ext)s", f"{language}.json3")
        )
        caption_path.write_text(
            caption_fixture.read_text(encoding="utf-8"), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return calls, run


def test_source_from_youtube_prefers_manual_english_captions(tmp_path: Path) -> None:
    info = _youtube_info(
        subtitles={"en-GB": [{"ext": "json3"}]},
        automatic={"en-orig": [{"ext": "json3"}]},
    )
    calls, runner = _fake_ytdlp_runner(info)

    source = source_from_youtube(
        "https://www.youtube.com/watch?v=video-123",
        out_dir=tmp_path,
        run_command=runner,
    )

    assert source.kind == "youtube"
    assert source.message_id == "youtube:video-123"
    assert source.source_url == "https://www.youtube.com/watch?v=video-123"
    assert source.title == info["title"]
    assert source.date is not None and source.date.isoformat() == "2025-11-28"
    assert source.raw_text_path == tmp_path / "captions.en-GB.json3"
    assert source.metadata["subtitle_track"] == "en-GB"
    assert source.metadata["subtitle_kind"] == "manual"
    assert source.metadata["channel"] == "PX4 Autopilot"
    assert source.metadata["duration_seconds"] == 1905
    assert source.metadata["capture_method"] == "yt-dlp"
    assert (tmp_path / "video-info.json").exists()
    assert "--write-subs" in calls[1]
    assert "--write-auto-subs" not in calls[1]


def test_source_from_youtube_uses_original_english_auto_captions(
    tmp_path: Path,
) -> None:
    info = _youtube_info(
        automatic={
            "en": [{"ext": "json3"}],
            "en-orig": [{"ext": "json3"}],
        }
    )
    calls, runner = _fake_ytdlp_runner(info)

    source = source_from_youtube(
        "https://www.youtube.com/watch?v=video-123",
        out_dir=tmp_path,
        run_command=runner,
    )

    assert source.raw_text_path == tmp_path / "captions.en-orig.json3"
    assert source.metadata["subtitle_kind"] == "automatic"
    assert "--write-auto-subs" in calls[1]


def test_source_from_youtube_rejects_video_without_requested_captions(
    tmp_path: Path,
) -> None:
    _calls, runner = _fake_ytdlp_runner(_youtube_info())

    with pytest.raises(SourcePrimitiveError, match="No English captions"):
        source_from_youtube(
            "https://www.youtube.com/watch?v=video-123",
            out_dir=tmp_path,
            run_command=runner,
        )


def test_source_from_youtube_rejects_stale_caption_file(tmp_path: Path) -> None:
    info = _youtube_info(subtitles={"en": [{"ext": "json3"}]})
    caption_path = tmp_path / "captions.en.json3"
    caption_path.write_text("stale", encoding="utf-8")

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        stdout = json.dumps(info) if "--dump-single-json" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    with pytest.raises(SourcePrimitiveError, match="did not create JSON3 captions"):
        source_from_youtube(
            "https://www.youtube.com/watch?v=video-123",
            out_dir=tmp_path,
            run_command=runner,
        )

    assert not caption_path.exists()


def test_source_from_youtube_rejects_incomplete_provenance(tmp_path: Path) -> None:
    info = _youtube_info(subtitles={"en": [{"ext": "json3"}]})
    del info["upload_date"]
    calls, runner = _fake_ytdlp_runner(info)

    with pytest.raises(SourcePrimitiveError, match="publication date"):
        source_from_youtube(
            "https://www.youtube.com/watch?v=video-123",
            out_dir=tmp_path,
            run_command=runner,
        )

    assert len(calls) == 1


def test_parse_youtube_json3_preserves_timing_and_removes_caption_noise(
    tmp_path: Path,
) -> None:
    caption_path = tmp_path / "captions.en-orig.json3"
    caption_path.write_text(
        (FIXTURES_DIR / "youtube-sample.json3").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "kind": "youtube",
                "source_url": "https://www.youtube.com/watch?v=video-123",
                "raw_text_path": str(caption_path),
                "metadata": {"subtitle_track": "en-orig"},
            }
        ),
        encoding="utf-8",
    )

    source = source_from_json(source_path)
    artifact = parse_youtube_json3(source)

    assert [turn.text for turn in artifact.turns] == [
        "MAVLink tools solve integration problems.",
        "The protocol stays interoperable.",
        "The protocol stays interoperable.",
        "Blue Robotics uses the same messages.",
    ]
    assert artifact.turns[0].start == 1.2
    assert artifact.turns[0].end == 3.4
    assert artifact.turns[-1].start == 7.8
    assert artifact.source_refs[0].source_ref.endswith("events[1]")
    assert all("Music" not in turn.text for turn in artifact.turns)


def test_parse_youtube_json3_drops_only_overlapping_duplicate_events(
    tmp_path: Path,
) -> None:
    caption_path = tmp_path / "captions.en.json3"
    caption_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 1000,
                        "dDurationMs": 2000,
                        "segs": [{"utf8": "Repeated caption text."}],
                    },
                    {
                        "tStartMs": 2500,
                        "dDurationMs": 1500,
                        "segs": [{"utf8": "Repeated caption text."}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    source = SourceArtifact(kind="youtube", raw_text_path=caption_path)

    artifact = parse_youtube_json3(source)

    assert [turn.text for turn in artifact.turns] == ["Repeated caption text."]


def source_from_json(path: Path):
    from jake_tools.transcripts.models import SourceArtifact

    return SourceArtifact.model_validate_json(path.read_text(encoding="utf-8"))
