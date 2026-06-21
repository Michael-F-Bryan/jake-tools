from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ..hermes import Hermes, HermesResult
from .audio import build_concat_plan, concatenate_recordings, run_scribe
from .merge import format_timestamp, merge_note, render_chaptered_transcript, speaker_name
from .models import (
    AIStageStats,
    AITotals,
    Chapter,
    ChapterSummary,
    ChaptersPayload,
    ConcatPlan,
    CoordinatorResult,
    MeetingMinutes,
    SourceNote,
    SpeakerMapping,
    SpeakerMessageCount,
    TranscriptTurn,
)
from .obsidian import load_source_note
from .paths import Paths
from .stages import (
    run_chaptering_stage,
    run_minutes_stage,
    run_speaker_mapping_stage,
    run_transcript_polish_stage,
)
from .transforms import merge_consecutive_turns, normalise_turns
from .verify import verify_note

SourceLoader = Callable[[Path], SourceNote]
ConcatenateRecordings = Callable[[ConcatPlan, Path], None]
RunScribe = Callable[[Path, Path], object]
MapSpeakers = Callable[
    [Hermes, SourceNote, list[TranscriptTurn]],
    tuple[SpeakerMapping, HermesResult | None],
]
PolishTranscript = Callable[
    [Hermes, SourceNote, list[TranscriptTurn], SpeakerMapping],
    tuple[list[TranscriptTurn], HermesResult | None],
]
RunChaptering = Callable[
    [Hermes, list[TranscriptTurn]],
    tuple[ChaptersPayload, HermesResult | None],
]
RunMinutes = Callable[
    [Hermes, list[TranscriptTurn], ChaptersPayload | None],
    tuple[MeetingMinutes, HermesResult | None],
]


def _build_ai_stage_stats(stage: str, result: HermesResult | None) -> AIStageStats | None:
    if result is None:
        return None

    return AIStageStats(
        stage=stage,
        model=result.model,
        provider=result.provider,
        api_calls=result.api_calls,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_write_tokens=result.cache_write_tokens,
        reasoning_tokens=result.reasoning_tokens,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
        repair_attempted=result.api_calls > 1,
    )


def _build_ai_totals(stage_stats: list[AIStageStats]) -> AITotals:
    totals = AITotals(stage_count=len(stage_stats))
    for stage in stage_stats:
        totals.api_calls += stage.api_calls
        totals.input_tokens += stage.input_tokens
        totals.output_tokens += stage.output_tokens
        totals.cache_read_tokens += stage.cache_read_tokens
        totals.cache_write_tokens += stage.cache_write_tokens
        totals.reasoning_tokens += stage.reasoning_tokens
        totals.prompt_tokens += stage.prompt_tokens
        totals.completion_tokens += stage.completion_tokens
        totals.total_tokens += stage.total_tokens
        totals.estimated_cost_usd += stage.estimated_cost_usd
    return totals


def _build_chapter_summaries(chapters: list[Chapter]) -> list[ChapterSummary]:
    return [
        ChapterSummary(
            title=chapter.title,
            start_timestamp=format_timestamp(chapter.start),
            end_timestamp=format_timestamp(chapter.end),
        )
        for chapter in chapters
    ]


def _build_speaker_message_counts(
    turns: list[TranscriptTurn],
    mapping: SpeakerMapping | None,
) -> list[SpeakerMessageCount]:
    counts: Counter[str] = Counter()
    for turn in turns:
        counts[speaker_name(turn.speaker, mapping)] += 1

    ordered_names: list[str] = []
    for turn in turns:
        name = speaker_name(turn.speaker, mapping)
        if name not in ordered_names:
            ordered_names.append(name)

    return [SpeakerMessageCount(speaker=name, messages=counts[name]) for name in ordered_names]


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

            turns = merge_consecutive_turns(
                normalise_turns(self._load_turns(paths.transcript))
            )

            ai_stage_stats: list[AIStageStats] = []

            speaker_mapping, speaker_mapping_result = self.map_speakers(
                self.hermes, source, turns
            )
            if stage_stats := _build_ai_stage_stats(
                "speaker_mapping", speaker_mapping_result
            ):
                ai_stage_stats.append(stage_stats)
            paths.speaker_mapping.write_text(
                json.dumps(speaker_mapping.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            polished_payload, transcript_polish_result = self.polish_transcript(
                self.hermes, source, turns, speaker_mapping
            )
            polished_turns = merge_consecutive_turns(
                normalise_turns(polished_payload)
            )
            if stage_stats := _build_ai_stage_stats(
                "transcript_polish", transcript_polish_result
            ):
                ai_stage_stats.append(stage_stats)

            chapters_payload, chaptering_result = self.build_chapters(
                self.hermes, polished_turns
            )
            if stage_stats := _build_ai_stage_stats("chaptering", chaptering_result):
                ai_stage_stats.append(stage_stats)
            paths.chapters.write_text(
                json.dumps(chapters_payload.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

            transcript_body = render_chaptered_transcript(
                polished_turns,
                chapters_payload.chapters,
                speaker_mapping,
            )
            minutes, meeting_minutes_result = self.build_minutes(
                self.hermes, polished_turns, chapters_payload
            )
            if stage_stats := _build_ai_stage_stats(
                "meeting_minutes", meeting_minutes_result
            ):
                ai_stage_stats.append(stage_stats)
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
            verify_note(
                updated_note,
                original_body=source.body,
                chapters=chapters_payload.chapters,
            )

            if not self.dry_run:
                self.source_note.write_text(updated_note, encoding="utf-8")

            return CoordinatorResult(
                note_path=self.source_note,
                updated=not self.dry_run,
                chapter_summaries=_build_chapter_summaries(chapters_payload.chapters),
                ai_stage_stats=ai_stage_stats,
                ai_totals=_build_ai_totals(ai_stage_stats),
                speaker_message_counts=_build_speaker_message_counts(polished_turns, speaker_mapping),
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
