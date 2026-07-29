from __future__ import annotations

import json
from pathlib import Path

import pytest

from jake_tools.transcripts.parse import (
    ParsePrimitiveError,
    parse_scribe_transcript,
)


def test_parse_scribe_transcript_wraps_decode_errors_with_the_file_path(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "merged.json"
    transcript_path.write_text("not json", encoding="utf-8")

    with pytest.raises(ParsePrimitiveError, match=str(transcript_path)):
        parse_scribe_transcript(transcript_path)


def test_parse_scribe_transcript_rejects_zero_turns(tmp_path: Path) -> None:
    transcript_path = tmp_path / "merged.json"
    transcript_path.write_text(json.dumps({"segments": []}), encoding="utf-8")

    with pytest.raises(ParsePrimitiveError, match="No spoken transcript turns"):
        parse_scribe_transcript(transcript_path)


def test_parse_scribe_transcript_uses_shared_unknown_speaker_fallback(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "merged.json"
    transcript_path.write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "Hello"}]}),
        encoding="utf-8",
    )

    artifact = parse_scribe_transcript(transcript_path)

    assert artifact.turns[0].speaker == "Unknown speaker"
