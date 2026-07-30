"""B1/N2: pyannote-audio 4.0.7 enables OpenTelemetry metrics by default and
sends recording duration + speaker constraints + a session UUID to
https://otel.pyannote.ai/v1/traces on every pipeline call, with the OTel
logger forced to CRITICAL so a blocked export is invisible. Importing
inference_worker.diarise must force PYANNOTE_METRICS_ENABLED=false before
pyannote.audio is ever imported (it's only ever imported lazily, inside
run_diarisation), and it must win even over an *inherited* ambient
PYANNOTE_METRICS_ENABLED=true (N2: `os.environ.setdefault` let a
pre-existing "true" in the environment reinstate the telemetry beacon —
verified, the DNS target reappeared — so this is now a hard assignment,
not a default).

Runs in a real, isolated subprocess (not this test process) so it proves
the ordering from a clean environment, and so it can't be fooled by some
earlier test in the same process having already imported pyannote.audio
and set the env var itself.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_diarise_disables_pyannote_telemetry_before_any_pyannote_import(
    monkeypatch,
):
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import inference_worker.diarise\n"
            "import os\n"
            "assert 'pyannote.audio' not in __import__('sys').modules, "
            "'pyannote.audio must not be imported eagerly by inference_worker.diarise'\n"
            "assert os.environ.get('PYANNOTE_METRICS_ENABLED') == 'false', os.environ.get('PYANNOTE_METRICS_ENABLED')\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_importing_diarise_overrides_a_pre_existing_true_value(monkeypatch):
    """N2: os.environ.setdefault would let an inherited shell/CI variable
    reinstate the telemetry beacon — given the corpus privacy rules this
    worker operates under, the worker's own choice must win regardless of
    what the ambient environment it was launched from already set.
    Opt-in-via-environment is deliberately not honoured."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os\n"
            "os.environ['PYANNOTE_METRICS_ENABLED'] = 'true'\n"
            "import inference_worker.diarise\n"
            "assert os.environ['PYANNOTE_METRICS_ENABLED'] == 'false', os.environ['PYANNOTE_METRICS_ENABLED']\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_pyannote_telemetry_stays_disabled_once_pyannote_audio_is_actually_imported(
    monkeypatch,
):
    """The real regression: pyannote.audio.telemetry.metrics only sets
    PYANNOTE_METRICS_ENABLED from its own config.yaml default (true) if
    the var isn't ALREADY set when *that* module gets imported. This
    proves our setdefault still wins even once the real import happens,
    not just that the env var looks right before pyannote loads."""
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import inference_worker.diarise\n"
            "from pyannote.audio.telemetry.metrics import is_metrics_enabled\n"
            "assert is_metrics_enabled() is False, 'pyannote telemetry is enabled'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
