import json
from datetime import datetime, timezone

from jake_tools.hermes import Hermes
from jake_tools.transcripts.coordinator import ObsidianRecordingCoordinator
from jake_tools.transcripts.models import RecordingRef, SourceNote, SpeakerIdentity, SpeakerMapping


def build_source(note, tmp_path) -> SourceNote:
    return SourceNote(
        path=note,
        body=note.read_text(encoding="utf-8"),
        recordings=[
            RecordingRef(
                raw_link="meeting.m4a",
                resolved_path=tmp_path / "meeting.m4a",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        ],
    )


def test_coordinator_transcript_mode_dry_run_preserves_note_file(tmp_path) -> None:
    note = tmp_path / "Meeting.md"
    note.write_text("# Meeting\n\n![[meeting.m4a]]\n", encoding="utf-8")

    def load_source(path):
        return build_source(note, tmp_path)

    def concatenate_audio(plan, concat_file):
        return None

    def fake_run_scribe(input_audio, output_json):
        output_json.write_text(
            json.dumps(
                {
                    "segments": [
                        {"start": 0, "end": 2, "speaker": "SPEAKER_01", "text": "Hello team"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return None

    def map_speakers(hermes, turns):
        return SpeakerMapping(
            mapping={
                "SPEAKER_01": SpeakerIdentity(name="Michael", confidence=0.9, reason="test")
            }
        )

    coordinator = ObsidianRecordingCoordinator(
        hermes=Hermes(),
        source_note=note,
        mode="transcript",
        dry_run=True,
        source_loader=load_source,
        concatenate_audio=concatenate_audio,
        transcribe_audio=fake_run_scribe,
        map_speakers=map_speakers,
    )

    result = coordinator.run()

    assert result.mode == "transcript"
    assert result.updated is False
    assert note.read_text(encoding="utf-8") == "# Meeting\n\n![[meeting.m4a]]\n"
