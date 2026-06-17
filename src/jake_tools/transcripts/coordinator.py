from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from ..hermes import Hermes
from .audio import build_concat_plan, concatenate_recordings, run_scribe
from .merge import build_merge_report, merge_note, render_chaptered_transcript, render_transcript
from .models import CoordinatorResult, MeetingMinutes, SpeakerMapping, TranscriptTurn
from .obsidian import load_source_note
from .paths import Paths
from .stages import run_chaptering_stage, run_minutes_stage, run_speaker_mapping_stage
from .verify import verify_note

Mode = Literal["transcript", "chaptered-transcript", "minutes"]


class ObsidianRecordingCoordinator:
    def __init__(
        self,
        *,
        hermes: Hermes,
        source_note: Path,
        mode: Mode = "minutes",
        dry_run: bool = False,
    ) -> None:
        self.hermes = hermes
        self.source_note = source_note
        self.mode: Mode = mode
        self.dry_run = dry_run

    def run(self) -> CoordinatorResult:
        source = load_source_note(self.source_note)
        if not source.recordings:
            raise ValueError(f"No recordings found in {self.source_note}")

        with Paths.temp() as paths:
            concat_plan = build_concat_plan(source.recordings, paths.merged)
            concatenate_recordings(concat_plan, paths.root / "inputs.txt")
            run_scribe(paths.merged, paths.transcript)

            turns = self._load_turns(paths.transcript)
            speaker_mapping = run_speaker_mapping_stage(self.hermes, turns)
            paths.speaker_mapping.write_text(
                json.dumps(speaker_mapping.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            chapters_payload = None
            minutes = None
            transcript_body = render_transcript(turns, speaker_mapping)

            if self.mode in {"chaptered-transcript", "minutes"}:
                chapters_payload = run_chaptering_stage(self.hermes, turns)
                paths.chapters.write_text(
                    json.dumps(chapters_payload.model_dump(mode="json"), indent=2),
                    encoding="utf-8",
                )
                transcript_body = render_chaptered_transcript(
                    turns,
                    chapters_payload.chapters,
                    speaker_mapping,
                )

            if self.mode == "minutes":
                minutes = run_minutes_stage(self.hermes, turns, chapters_payload)
                paths.minutes.write_text(
                    json.dumps(minutes.model_dump(mode="json"), indent=2),
                    encoding="utf-8",
                )

            updated_note = merge_note(
                source.body,
                mode=self.mode,
                transcript_body=transcript_body,
                chapters=chapters_payload.chapters if chapters_payload else None,
                minutes=minutes,
            )
            merge_report = build_merge_report(
                source.body,
                updated_note,
                len(chapters_payload.chapters) if chapters_payload else 0,
            )
            merge_report = verify_note(
                updated_note,
                original_body=source.body,
                mode=self.mode,
                chapters=chapters_payload.chapters if chapters_payload else [],
            )
            paths.merge_report.write_text(
                json.dumps(merge_report.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            if not self.dry_run:
                self.source_note.write_text(updated_note, encoding="utf-8")

            return CoordinatorResult(
                mode=self.mode,
                note_path=self.source_note,
                updated=not self.dry_run,
                merged_audio=paths.merged,
                transcript_json=paths.transcript,
                chapters_json=paths.chapters if chapters_payload else None,
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
    mode: Mode = "minutes",
    dry_run: bool = False,
) -> CoordinatorResult:
    coordinator = ObsidianRecordingCoordinator(
        hermes=hermes,
        source_note=obsidian_note,
        mode=mode,
        dry_run=dry_run,
    )
    return coordinator.run()
