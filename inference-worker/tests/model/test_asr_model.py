"""Real Parakeet ASR smoke tests. Heavyweight (model download/load) —
opt-in via `-m model`, excluded from the default `pytest -q` run.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from inference_worker.asr import CHUNK_DURATION_S, run_asr
from inference_worker.provenance import sha256_file

pytestmark = pytest.mark.model


def test_run_asr_completes_on_synthetic_audio(sine_wav_factory, tmp_path):
    # Real Parakeet ASR doesn't care about the source sample rate/channel
    # layout the way ffmpeg's prepare stage does, but the M11 pipeline
    # always hands it an already-prepared 16 kHz mono wav, so match that.
    wav = sine_wav_factory(
        tmp_path / "prepared.wav", seconds=3.0, sample_rate=16000, channels=1
    )
    wav_sha256 = sha256_file(wav)

    result = run_asr(wav, wav_sha256, timeout_s=180.0)

    assert result.status == "completed", result.error
    assert result.input_audio_sha256 == wav_sha256
    assert result.model_provenance is not None
    assert result.model_provenance.identity.name == "mlx-community/parakeet-tdt-0.6b-v2"
    assert result.model_provenance.package_versions["parakeet-mlx"]
    assert result.output is not None
    # Sine tones aren't speech: empty text/tokens is the expected,
    # non-crashing outcome — the point is the typed pipeline runs for real.
    assert isinstance(result.output.text, str)
    assert isinstance(result.output.tokens, list)


def _synthesize_long_speech(tmp_path: Path, *, minutes: float) -> Path:
    """Real (if repetitive) synthesized speech via macOS `say`, long
    enough to force multiple CHUNK_DURATION_S/OVERLAP_DURATION_S chunks.

    Sine-tone audio (used above) produces zero transcribable tokens —
    fine for a cheap smoke test, useless for exercising chunk-boundary
    merge behaviour, which needs actual words landing near a chunk seam.
    Text is generated on the fly (counting sentences), not read from any
    recording, vault note, or corpus fixture.
    """
    number_words = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]
    words_per_minute = 200  # `say`'s default rate is roughly this fast
    target_words = int(minutes * words_per_minute)

    sentences: list[str] = []
    word_count = 0
    n = 0
    while word_count < target_words:
        digits: list[str] = []
        remaining = n
        while True:
            digits.append(number_words[remaining % 10])
            remaining //= 10
            if remaining == 0:
                break
        sentence = "The count is now " + " ".join(reversed(digits)) + "."
        sentences.append(sentence)
        word_count += len(sentence.split())
        n += 1

    script_path = tmp_path / "script.txt"
    script_path.write_text(" ".join(sentences))

    aiff_path = tmp_path / "speech.aiff"
    subprocess.run(
        ["say", "-v", "Samantha", "-o", str(aiff_path), "-f", str(script_path)],
        check=True,
        timeout=120,
        capture_output=True,
    )

    wav_path = tmp_path / "speech_16k.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-i",
            str(aiff_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return wav_path


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


@pytest.mark.skipif(
    shutil.which("say") is None, reason="macOS `say` TTS not available on this machine"
)
def test_run_asr_completes_on_long_multi_chunk_audio(tmp_path):
    """Regression test for the field failure on real 28-minute meeting
    audio: whole-file transcription allocates a log-mel buffer
    proportional to duration and blew Metal's max buffer size
    (`[metal::malloc] Attempting to allocate 28611722496 bytes which is
    greater than the maximum allowed buffer size of 9534832640 bytes`)
    after a 10s synthetic smoke test had passed. Generates >5 minutes of
    real synthesized speech — long enough to force several
    CHUNK_DURATION_S=120s/OVERLAP_DURATION_S=15s chunk boundaries (at
    roughly the 120s, 225s, 330s marks) — and asserts completed status
    and (start_ms, end_ms)-ordered tokens (M6/M7: non-strict — duplicates
    are legal raw evidence, not asserted against here; see
    test_asr_merge_property.py for the property that any surviving
    duplicate must trace to a single chunk's own decode)."""
    wav = _synthesize_long_speech(tmp_path, minutes=6.0)
    duration_s = _wav_duration_s(wav)
    assert duration_s > 2 * CHUNK_DURATION_S, (
        f"fixture audio ({duration_s:.0f}s) must force at least 2 chunk boundaries"
    )
    wav_sha256 = sha256_file(wav)

    result = run_asr(wav, wav_sha256, timeout_s=280.0)

    assert result.status == "completed", result.error
    assert result.output is not None
    # Real speech, not tone/silence: this would be near-zero if chunking
    # somehow dropped whole chunks instead of merging them.
    assert len(result.output.tokens) > 100

    keys = [(t.start_ms, t.end_ms) for t in result.output.tokens]
    assert keys == sorted(keys), (
        "chunk-merged tokens must stay ordered (non-strictly) by (start_ms, end_ms)"
    )

    duplicate_count = len(keys) - len(
        {(t.text, t.start_ms, t.end_ms) for t in result.output.tokens}
    )
    zero_length_count = sum(1 for t in result.output.tokens if t.start_ms == t.end_ms)
    print(
        f"\n[chunk_boundaries_ms={result.output.chunk_boundaries_ms}] tokens={len(keys)} zero_length={zero_length_count} exact_duplicates={duplicate_count}"
    )

    last_token_end_s = result.output.tokens[-1].end_ms / 1000
    assert last_token_end_s > 2 * CHUNK_DURATION_S, (
        "transcription must cover audio past the first chunk"
    )
