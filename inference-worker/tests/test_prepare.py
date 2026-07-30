"""Prepare stage behaviour with a real ffmpeg subprocess (no mocking).

Source fixtures are deliberately NOT already 16 kHz mono s16le, so a
passing test proves ffmpeg actually normalised the audio rather than the
stage merely copying an already-conformant file.
"""

from __future__ import annotations

import os
import wave

from inference_worker.models import AudioPreparationConfig
from inference_worker.prepare import (
    ffmpeg_version,
    prepare_audio,
    prepare_stage_config_hash,
)
from inference_worker.provenance import config_hash, sha256_file

_SOURCE_SHA = "a" * 64  # tests here mostly don't care about the real value


def _read_wav_format(path):
    with wave.open(str(path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()


def test_prepare_converts_stereo_44k_to_mono_16k(sine_wav_factory, tmp_path):
    source = sine_wav_factory(
        tmp_path / "source.wav", seconds=1.0, sample_rate=44100, channels=2
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = prepare_audio(
        source, out_dir, AudioPreparationConfig(), _SOURCE_SHA, timeout_s=30.0
    )

    assert result.status == "completed"
    assert result.error is None
    assert result.output is not None
    rate, channels, width = _read_wav_format(result.output.path)
    assert (rate, channels, width) == (16000, 1, 2)  # s16le == 2-byte samples
    assert result.output.duration_ms == 1000
    assert len(result.config_hash) == 64


def test_prepare_result_echoes_the_source_sha256(sine_wav_factory, tmp_path):
    """M3: source_sha256 is threaded through unchanged — resume-by-hash
    needs the *source* hash, not just the prepared wav's."""
    source = sine_wav_factory(tmp_path / "source.wav", seconds=0.2)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    source_sha = sha256_file(source)

    result = prepare_audio(
        source, out_dir, AudioPreparationConfig(), source_sha, timeout_s=30.0
    )

    assert result.source_sha256 == source_sha


def test_prepare_output_hash_matches_written_file(sine_wav_factory, tmp_path):
    source = sine_wav_factory(tmp_path / "source.wav", seconds=0.5)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = prepare_audio(
        source, out_dir, AudioPreparationConfig(), _SOURCE_SHA, timeout_s=30.0
    )

    assert result.output is not None
    assert result.output.sha256 == sha256_file(result.output.path)


def test_prepare_reports_ffmpeg_failure_on_corrupt_input(tmp_path):
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not actually audio data")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = prepare_audio(
        corrupt, out_dir, AudioPreparationConfig(), _SOURCE_SHA, timeout_s=30.0
    )

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

    result = prepare_audio(
        fifo_path, out_dir, AudioPreparationConfig(), _SOURCE_SHA, timeout_s=0.5
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.error_class == "timeout"
    assert result.error.retryable is True


def test_prepare_handles_a_colon_in_the_source_filename(sine_wav_factory, tmp_path):
    """m1: ffmpeg's protocol detection treats a bare `scheme:...` argument
    as a URL. A source filename like "14:30 sync.wav" (a plausible
    meeting-title-derived name) must still work — proving the `file:`
    prefix + Path.resolve() actually neutralises it, not merely that a
    plain filename works."""
    source = sine_wav_factory(tmp_path / "14:30 sync.wav", seconds=0.2)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = prepare_audio(
        source, out_dir, AudioPreparationConfig(), _SOURCE_SHA, timeout_s=30.0
    )

    assert result.status == "completed", result.error
    assert result.output is not None


def test_ffmpeg_unavailable_is_retryable(sine_wav_factory, tmp_path, monkeypatch):
    """m4: a missing/unlaunchable ffmpeg binary is an environment problem
    the caller can plausibly fix (install ffmpeg, fix PATH) and retry —
    unlike ffmpeg rejecting the input itself, which isn't retryable."""
    source = sine_wav_factory(tmp_path / "source.wav", seconds=0.2)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path))  # a real directory with no ffmpeg in it

    result = prepare_audio(
        source, out_dir, AudioPreparationConfig(), _SOURCE_SHA, timeout_s=5.0
    )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.error_class == "ffmpeg-unavailable"
    assert result.error.retryable is True


def test_ffmpeg_version_resolves_a_real_version_string():
    version = ffmpeg_version()

    assert version != "unknown"
    assert "ffmpeg" in version.lower()


def test_prepare_stage_config_hash_folds_in_ffmpeg_version():
    """M2: two runs against different ffmpeg builds must hash
    differently — config_hash must not cover only the compile-time
    AudioPreparationConfig constants."""
    config = AudioPreparationConfig()

    hash_with_ffmpeg_version = prepare_stage_config_hash(config)
    hash_without_ffmpeg_version = config_hash(config.model_dump())

    assert hash_with_ffmpeg_version != hash_without_ffmpeg_version
