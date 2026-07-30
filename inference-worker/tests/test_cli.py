"""CLI contract-level behaviour: everything that must fail *before* any
stage runs, so these tests stay fast without needing real ASR/diarisation.

Exit code contract: 0 whenever a valid response.json was written, nonzero
only for contract-level failures (unreadable/invalid request, audio hash
mismatch, unwritable out-dir).
"""

from __future__ import annotations

import os

from inference_worker.__main__ import main


def test_run_rejects_missing_request_file(tmp_path):
    out_dir = tmp_path / "out"

    exit_code = main(
        ["run", str(tmp_path / "does-not-exist.json"), "--out-dir", str(out_dir)]
    )

    assert exit_code != 0
    assert not (out_dir / "response.json").exists()


def test_run_rejects_malformed_request_json(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text("{not valid json")
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(request_path), "--out-dir", str(out_dir)])

    assert exit_code != 0
    assert not (out_dir / "response.json").exists()


def test_run_rejects_audio_hash_mismatch(request_factory, sine_wav_factory, tmp_path):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(
        audio_path=wav, audio_sha256="0" * 64
    )  # deliberately wrong
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    out_dir = tmp_path / "out"

    exit_code = main(["run", str(request_path), "--out-dir", str(out_dir)])

    assert exit_code != 0
    assert not (out_dir / "response.json").exists()


def test_run_rejects_unwritable_out_dir(request_factory, sine_wav_factory, tmp_path):
    wav = sine_wav_factory(tmp_path / "source.wav")
    request = request_factory(audio_path=wav)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())

    read_only_parent = tmp_path / "locked"
    read_only_parent.mkdir()
    out_dir = read_only_parent / "out"
    os.chmod(read_only_parent, 0o500)
    try:
        exit_code = main(["run", str(request_path), "--out-dir", str(out_dir)])
    finally:
        os.chmod(read_only_parent, 0o700)  # restore so tmp_path cleanup can delete it

    assert exit_code != 0
    assert not out_dir.exists()
