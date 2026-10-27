"""Synthetic two-speaker recordings for exercising the real transcription service.

Built on demand with macOS ``say`` (two distinct voices) and ``afconvert``,
so nothing binary is committed. The conversation is scripted, which gives
the live test known words per speaker and a known speaker order across the
file boundary. Text-to-speech is clean, single-channel and unoverlapped: it
exercises the whole pipeline but is not evidence of accuracy on real
meetings.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16_000
GAP_SECONDS = 0.6

SPEAKER_A = "Samantha"
SPEAKER_B = "Daniel"

# (voice, text). File two opens with the voice that closed file one, so a
# speaker identity must survive the file boundary.
FILE_ONE: tuple[tuple[str, str], ...] = (
    (SPEAKER_A, "Good morning everyone. Let us start the weekly planning meeting."),
    (SPEAKER_B, "Thanks. The robot arm calibration finished yesterday evening."),
    (SPEAKER_A, "Excellent. Did the vision pipeline pass the regression suite?"),
    (SPEAKER_B, "Almost. Two camera tests still fail under low light conditions."),
    (SPEAKER_A, "Then let us schedule a review for Thursday afternoon."),
    (SPEAKER_B, "Agreed. I will prepare the failure report before then."),
)
FILE_TWO: tuple[tuple[str, str], ...] = (
    (SPEAKER_B, "One more thing. The battery supplier confirmed the delivery date."),
    (SPEAKER_A, "That is good news. Please update the procurement spreadsheet."),
    (SPEAKER_B, "Done already. Shall we close the meeting?"),
    (SPEAKER_A, "Yes. Thank you all, see you next week."),
)


@dataclass(frozen=True)
class Utterance:
    voice: str
    text: str
    file_index: int
    start_seconds: float
    end_seconds: float

    @property
    def words(self) -> list[str]:
        return [normalise(word) for word in self.text.split()]


@dataclass
class Conversation:
    directory: Path
    wav_files: list[Path]
    m4a_files: list[Path]
    utterances: list[Utterance]
    file_durations: list[float] = field(default_factory=list)

    @property
    def voices(self) -> list[str]:
        """Voices in order of first appearance across the whole recording."""
        seen: list[str] = []
        for utterance in self.utterances:
            if utterance.voice not in seen:
                seen.append(utterance.voice)
        return seen

    def words_for(self, voice: str) -> set[str]:
        return {w for u in self.utterances if u.voice == voice for w in u.words}


def normalise(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalnum())


def tools_available() -> str | None:
    """``None`` when fixtures can be built here; otherwise why not."""
    for tool in ("say", "afconvert"):
        if shutil.which(tool) is None:
            return f"`{tool}` is not available (macOS only)"
    probe = subprocess.run(
        ["say", "-v", "?"], capture_output=True, text=True, check=False
    )
    for voice in (SPEAKER_A, SPEAKER_B):
        if not any(line.startswith(voice + " ") for line in probe.stdout.splitlines()):
            return f"the `{voice}` voice is not installed"
    return None


def synthesise_conversation(directory: Path) -> Conversation:
    """Write ``meeting-part-1.wav``/``.m4a`` and ``meeting-part-2.wav``/``.m4a``."""
    directory.mkdir(parents=True, exist_ok=True)
    utterances: list[Utterance] = []
    wav_files: list[Path] = []
    m4a_files: list[Path] = []
    durations: list[float] = []
    offset = 0.0
    for file_index, script in enumerate((FILE_ONE, FILE_TWO)):
        frames = bytearray()
        file_start = offset
        for line_index, (voice, text) in enumerate(script):
            pcm = _speak(directory / f"u{file_index}-{line_index}.wav", voice, text)
            start = offset
            frames += pcm
            offset += len(pcm) / (2 * SAMPLE_RATE)
            utterances.append(Utterance(voice, text, file_index, start, offset))
            silence = struct.pack("<h", 0) * int(GAP_SECONDS * SAMPLE_RATE)
            frames += silence
            offset += GAP_SECONDS
        wav_path = directory / f"meeting-part-{file_index + 1}.wav"
        with wave.open(str(wav_path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(bytes(frames))
        m4a_path = wav_path.with_suffix(".m4a")
        subprocess.run(
            ["afconvert", "-f", "m4af", "-d", "aac", str(wav_path), str(m4a_path)],
            check=True,
            capture_output=True,
        )
        wav_files.append(wav_path)
        m4a_files.append(m4a_path)
        durations.append(offset - file_start)
    return Conversation(directory, wav_files, m4a_files, utterances, durations)


def _speak(path: Path, voice: str, text: str) -> bytes:
    subprocess.run(
        [
            "say",
            "-v",
            voice,
            "-o",
            str(path),
            "--file-format=WAVE",
            f"--data-format=LEI16@{SAMPLE_RATE}",
            text,
        ],
        check=True,
        capture_output=True,
    )
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == SAMPLE_RATE
        return handle.readframes(handle.getnframes())
