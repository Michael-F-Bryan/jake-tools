from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ..hermes import Hermes
from .audio import build_concat_plan, concatenate_recordings, run_scribe
from .merge import build_merge_report, merge_note, render_chaptered_transcript
from .models import ChaptersPayload, ConcatPlan, CoordinatorResult, MeetingMinutes, SourceNote, SpeakerMapping, TranscriptTurn
from .obsidian import load_source_note
from .paths import Paths
from .stages import run_chaptering_stage, run_minutes_stage, run_speaker_mapping_stage, run_transcript_polish_stage
from .transforms import merge_consecutive_turns, normalise_turns
from .verify import verify_note

SourceLoader = Callable[[Path], SourceNote]
ConcatenateRecordings = Callable[[ConcatPlan, Path], None]
RunScribe = Callable[[Path, Path], object]
MapSpeakers = Callable[[Hermes, SourceNote, list[TranscriptTurn]], SpeakerMapping]
PolishTranscript = Callable[[Hermes, SourceNote, list[TranscriptTurn], SpeakerMapping], list[TranscriptTurn]]
RunChaptering = Callable[[Hermes, list[TranscriptTurn]], ChaptersPayload]
RunMinutes = Callable[[Hermes, list[TranscriptTurn], ChaptersPayload | None], MeetingMinutes]


class ObsidianRecordingCoordinator:
    def __init__(
        self,
        *,
        hermes: Hermes,
        source_note: Path,
        dry_run: bool = False,
        source_loader: SourceLoader = load_source_note,
        concatenate_audio: ConcatenateRecordings = concatenate_recordings,
        transcribe_audio: RunScribe = run_scribe,
        map_speakers: MapSpeakers = run_speaker_mapping_stage,
        polish_transcript: PolishTranscript = run_transcript_polish_stage,
        build_chapters: RunChaptering = run_chaptering_stage,
        build_minutes: RunMinutes = run_minutes_stage,
    ) -> None:
        self.hermes = hermes
        self.source_note = source_note
        self.dry_run = dry_run
        self.source_loader = source_loader
        self.concatenate_audio = concatenate_audio
        self.transcribe_audio = transcribe_audio
        self.map_speakers = map_speakers
        self.polish_transcript = polish_transcript
        self.build_chapters = build_chapters
        self.build_minutes = build_minutes

    def run(self) -> CoordinatorResult:
        source = self.source_loader(self.source_note)
        if not source.recordings:
            raise ValueError(f"No recordings found in {self.source_note}")

        with Paths.temp() as paths:
            concat_plan = build_concat_plan(source.recordings, paths.merged)
            self.concatenate_audio(concat_plan, paths.root / "inputs.txt")
            self.transcribe_audio(paths.merged, paths.transcript)

            turns = merge_consecutive_turns(normalise_turns(self._load_turns(paths.transcript)))
            speaker_mapping = self.map_speakers(self.hermes, source, turns)
            paths.speaker_mapping.write_text(
                json.dumps(speaker_mapping.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            polished_turns = merge_consecutive_turns(
                normalise_turns(self.polish_transcript(self.hermes, source, turns, speaker_mapping))
            )
            chapters_payload = self.build_chapters(self.hermes, polished_turns)
            paths.chapters.write_text(
                json.dumps(chapters_payload.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            transcript_body = render_chaptered_transcript(
                polished_turns,
                chapters_payload.chapters,
                speaker_mapping,
            )
            minutes = self.build_minutes(self.hermes, polished_turns, chapters_payload)
            paths.minutes.write_text(
                json.dumps(minutes.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            updated_note = merge_note(
                source.body,
                transcript_body=transcript_body,
                chapters=chapters_payload.chapters,
                minutes=minutes,
            )
            merge_report = build_merge_report(
                source.body,
                updated_note,
                len(chapters_payload.chapters),
            )
            merge_report = verify_note(
                updated_note,
                original_body=source.body,
                chapters=chapters_payload.chapters,
            )
            paths.merge_report.write_text(
                json.dumps(merge_report.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            if not self.dry_run:
                self.source_note.write_text(updated_note, encoding="utf-8")

            return CoordinatorResult(
                note_path=self.source_note,
                updated=not self.dry_run,
                merged_audio=paths.merged,
                transcript_json=paths.transcript,
                chapters_json=paths.chapters,
                speaker_mapping_json=paths.speaker_mapping,
                merge_report=merge_report,
            )

    def _load_turns(self, transcript_json: Path) -> list[TranscriptTurn]:
        payload = json.loads(transcript_json.read_text(encoding="utf-8"))
        segments = payload.get("segments") if isinstance(payload, dict) else None
        if not isinstance(segments, list):
            raise ValueError("Transcript JSON does not contain a segments list")

        turns: list[TranscriptTurn] = []
        for index, segment in enumerate(segments, start=1):
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            speaker = (
                segment.get("speaker")
                or segment.get("speaker_label")
                or segment.get("speaker_name")
                or f"Speaker {index}"
            )
            turns.append(
                TranscriptTurn(
                    start=float(segment.get("start", 0.0) or 0.0),
                    end=float(segment.get("end", segment.get("start", 0.0)) or 0.0),
                    speaker=str(speaker),
                    text=text,
                )
            )
        return turns



def process_obsidian_recording(
    hermes: Hermes,
    obsidian_note: Path,
    *,
    dry_run: bool = False,
) -> CoordinatorResult:
    coordinator = ObsidianRecordingCoordinator(
        hermes=hermes,
        source_note=obsidian_note,
        dry_run=dry_run,
    )
    return coordinator.run()
