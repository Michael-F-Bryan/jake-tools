"""Prepare stage behaviour with a real ffmpeg subprocess (no mocking).

Source fixtures are deliberately NOT already 16 kHz mono s16le, so a
passing test proves ffmpeg actually normalised the audio rather than the
stage merely copying an already-conformant file.
"""

from __future__ import annotations

import os
import wave

from inference_worker.models import AudioPreparationConfig
from inference_worker.prepare import prepare_audio
from inference_worker.provenance import sha256_file


def _read_wav_format(path):
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()


def test_prepare_converts_stereo_44k_to_mono_16k(sine_wav_factory, tmp_path):
    source = sine_wav_factory(
        tmp_path / "source.wav", seconds=1.0, sample_rate=44100, channels=2
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = prepare_audio(source, out_dir, AudioPreparationConfig(), timeout_s=30.0)

    assert result.status == "completed"
    assert result.error is None
    assert result.output is not None
    rate, channels, width = _read_wav_format(result.output.path)
    assert (rate, channels, width) == (16000, 1, 2)  # s16le == 2-byte samples
    assert result.output.duration_ms == 1000
    assert len(result.config_hash) == 64


def test_prepare_output_hash_matches_written_file(sine_wav_factory, tmp_path):
    source = sine_wav_factory(tmp_path / "source.wav", seconds=0.5)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = prepare_audio(source, out_dir, AudioPreparationConfig(), timeout_s=30.0)

    assert result.output is not None
    assert result.output.sha256 == sha256_file(result.output.path)


def test_prepare_reports_ffmpeg_failure_on_corrupt_input(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not actually audio data")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = prepare_audio(corrupt, out_dir, AudioPreparationConfig(), timeout_s=30.0)

    assert result.status == "failed"
    assert result.output is None
    assert result.error is not None
    assert result.error.error_class == "ffmpeg-failed"
    assert result.error.retryable is False


def test_prepare_reports_timeout_on_a_real_stalled_ffmpeg(tmp_path):
    """A FIFO that nothing ever writes to makes ffmpeg block on read
    indefinitely — a real subprocess timeout, not a mocked one."""
    fifo_path = tmp_path / "stalled_input"
    os.mkfifo(fifo_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = prepare_audio(fifo_path, out_dir, AudioPreparationConfig(), timeout_s=0.5)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.error_class == "timeout"
    assert result.error.retryable is True
