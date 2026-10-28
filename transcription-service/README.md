# transcription-service

A long-running native macOS HTTP service that turns uploaded recordings into
one structured transcript: speech recognition plus speaker diarisation, fully
local, with the models loaded once and kept warm. The Python MCP tool
`transcribe` (`src/jake_tools/mcp/tools/transcribe.py`) is its client; the
design brief is `_working/transcription-service/README.md`.

## Engine and models

| Component | Choice | Pin |
| --- | --- | --- |
| Framework | [FluidAudio](https://github.com/FluidInference/FluidAudio) (Core ML) | `0.17.7`, `traits: []` |
| Speech recognition | Parakeet TDT 0.6B v3 (multilingual, 25 European languages) | `FluidInference/parakeet-tdt-0.6b-v3-coreml` at commit `7dd20fe6…` (`ModelPins.asrRevision`) |
| Diarisation | pyannote community-1 offline pipeline (powerset segmentation, WeSpeaker, VBx) | `FluidInference/speaker-diarization-coreml` at FluidAudio's own pinned commit `df2625ac…` |
| HTTP | Hummingbird `2.28.0`, MultipartKit `5.0.0-beta.2` (streaming parser) | exact |

Minimum macOS 15 on Apple silicon (FluidAudio reports a Core ML BNNS crash in
the offline diarizer on macOS 14). Built with Apple's toolchain (Xcode or the
Command Line Tools, Swift 6.2+), not nixpkgs' Swift.

## Running

```bash
cd transcription-service
swift build -c release

# Once, online: download and compile the pinned models.
.build/release/transcription-service provision [--models-dir DIR]

# Then, offline:
echo -n 'some-long-random-token' > /path/to/token
.build/release/transcription-service serve --token-file /path/to/token \
    --host 127.0.0.1 --port 8787 [--models-dir DIR]
```

`serve` binds immediately and reports `starting` on `/health` while it loads
and warms the models; it reports `ready` only after a warm-up inference has
succeeded, and exits non-zero if loading fails (so a supervisor notices). It
never downloads: a missing model is a load failure that `provision` fixes.

Flags (all have defaults): `--host 127.0.0.1`, `--port 8787`, `--token-file`
(or `TRANSCRIPTION_API_TOKEN`), `--models-dir` (or `TRANSCRIPTION_MODELS_DIR`,
default FluidAudio's cache `~/Library/Application Support/FluidAudio/Models`),
`--max-upload-bytes 2147483648`, `--max-files 64`, `--max-duration-seconds
10800`, `--max-queued 4`, `--work-dir` (default
`$TMPDIR/jake-transcription-service`), `--tls-certificate` +
`--tls-private-key` (PEM; enables HTTPS, TLS 1.2+), `--log-level info`.

`--work-dir` must be new, empty, or one this service created: it marks the
directory with `.jake-transcription-service`, and at startup removes only the
UUID-named request directories a crash left behind. A non-empty directory
without the marker is refused (exit 64), never cleaned. Staging holds at most
`--max-queued` + 1 requests' uploads, so size it as that many times
`--max-upload-bytes`.

`provision` replaces a model directory whose `.fluidaudio-revision` marker
does not match the pin (as after a pin bump); `serve` refuses such a
directory with a message naming both revisions.

Memory: decoded audio is held in memory as 16 kHz float32, about 230 MB per
hour, plus the models and the diarizer's working set. Measured on an M4 Max
(macOS 27, release build, real meetings, 2026-10-28): 230–340 MB physical
footprint idle with warm models, peaking at about 1.3 GB for 6 minutes,
1.65 GB for 33 minutes and 3.4 GB for 98 minutes. Size
`--max-duration-seconds` to the host.

Measured on the same machine with synthesised two-voice speech (not real
meetings): launch to ready 12.5 s (11.5 s of it loading Parakeet), a 41 s
two-file recording in 1.0 s, a 30 minute recording in 36 s (ASR 5 s,
diarisation 29 s, about 50x real time), and a mid-inference disconnect freed
the slot within a second. HTTPS via `--tls-*` was checked with a self-signed
certificate.

## HTTP contract

### `GET /health` (no authentication)

```json
{"status": "ready", "ready": true, "version": "0.1.0",
 "engine": {"name": "FluidAudio", "version": "0.17.7",
            "asr_model": "FluidInference/parakeet-tdt-0.6b-v3-coreml@7dd20fe6…",
            "diarization_model": "FluidInference/speaker-diarization-coreml@df2625ac…"},
 "detail": null,
 "queue": {"running": 0, "queued": 0, "max_queued": 4}}
```

`200` when `ready`, `503` while `starting` or after `failed` (then `detail`
says why). Intended for supervision and `jake-tools-mcp doctor`.

### `POST /v1/transcriptions`

- `Authorization: Bearer <token>` (constant-time comparison).
- `Content-Type: multipart/form-data; boundary=…`. Every part is named `file`
  and carries a `filename`; **part order is recording order**. Parts are
  streamed to a per-request temporary directory (mode 0700) and deleted when
  the request ends, however it ends. Anything `AVAssetReader` decodes is
  accepted (WAV, AIFF, CAF, M4A/MP4 with AAC, ALAC or Opus, MP3, FLAC);
  files are decoded to 16 kHz mono and concatenated with no gap. WebM and
  Ogg are refused with a message saying so: AVFoundation cannot demux them.
- A file must decode completely. A decode error part-way (damaged packets),
  a decoded length more than 0.5 s short of the container's declared
  duration, or a WAV/AIFF/CAF header declaring more audio than the file
  holds is `invalid_input` naming the file, never a shorter transcript.
  Declared durations are summed against `--max-duration-seconds` before
  anything is decoded.
- Response `200` with the complete transcript, or an error (below). There is
  no partial result, no job handle, no polling: the connection stays open for
  the duration. Expect minutes for long recordings.

Transcript (`application/json`, keys in snake_case):

```json
{
  "schema_version": 1,
  "engine": {"name": "…", "version": "…", "asr_model": "…", "diarization_model": "…"},
  "timestamp_unit": "seconds",
  "timestamp_origin": "first sample of the concatenated recording (files joined in request order)",
  "timing_resolution_seconds": 0.08,
  "duration_seconds": 41.13,
  "files": [
    {"index": 0, "name": "meeting-part-1.m4a", "start_seconds": 0.0, "end_seconds": 25.59, "duration_seconds": 25.59},
    {"index": 1, "name": "meeting-part-2.m4a", "start_seconds": 25.59, "end_seconds": 41.13, "duration_seconds": 15.54}
  ],
  "speakers": [
    {"id": "speaker_1", "speaking_seconds": 18.2, "segment_count": 5},
    {"id": "speaker_2", "speaking_seconds": 17.9, "segment_count": 5}
  ],
  "segments": [
    {"speaker": "speaker_1", "start_seconds": 0.08, "end_seconds": 3.6,
     "text": "Good morning everyone.",
     "words": [{"text": "Good", "start_seconds": 0.08, "end_seconds": 0.32,
                "confidence": 0.97, "speaker": "speaker_1", "overlap": false}]}
  ],
  "text": "Good morning everyone. …",
  "unassigned_word_count": 0,
  "limitations": ["…"],
  "processing": {"decode_seconds": 0.1, "asr_seconds": 1.2, "diarization_seconds": 2.3, "total_seconds": 3.7}
}
```

Semantics:

- Timestamps are seconds (three decimals) from the first sample of the
  concatenated recording. Subtract a file's `start_seconds` to map back into
  that file. Word timings come from the recogniser's 80 ms encoder frames;
  `timing_resolution_seconds` says so and nothing finer is claimed.
- `speakers` are anonymous, request-wide identities named by first
  appearance. A speaker keeps their label across file boundaries within a
  request; labels mean nothing across requests.
- Each word is attributed to the diarised turn overlapping it most. If turns
  of several speakers overlap it, `overlap` is `true`. A word no turn covers
  snaps to the nearest turn within 0.5 s, else `speaker` is `null` and it
  counts in `unassigned_word_count`. `segments` are runs of consecutive words
  with one speaker, also split at pauses over 2 s.
- `text` is the recogniser's full output; segment text is rebuilt from timed
  words and can differ in spacing/punctuation. `limitations` repeats the
  engine caveats on every result.

### Errors

Body `{"error": {"code": "…", "message": "…"}}`. Messages name files by their
uploaded name and limits by their value, never audio or transcript content.

| Status | `code` | Meaning |
| --- | --- | --- |
| 401 | `unauthorized` | Missing or wrong bearer token. |
| 400 | `invalid_input` | Not multipart, a part not named `file`, an empty part, no parts, a file that is not decodable audio (including WebM/Ogg), or one that is damaged or truncated. |
| 413 | `payload_too_large` | Over `--max-upload-bytes`, `--max-files` or `--max-duration-seconds` (also from `Content-Length` before reading the body). |
| 503 | `not_ready` | Models still loading (`Retry-After: 15`) or failed to load. |
| 429 | `overloaded` | `--max-queued` + 1 requests already admitted (`Retry-After: 30`). Decided before the body is read; nothing was uploaded or started. |
| 500 | `inference_failed` | The engine failed on an accepted recording; details are in the service log only. |
| 500 | `internal_error` | Anything else. |

## Concurrency, cancellation, failure

- One inference job at a time (`JobGate`), first come first served, at most
  `--max-queued` waiting; concurrent connections never mean concurrent model
  execution. A request is admitted (or refused with `429`) before its body is
  read, and counts against the bound from the start of its upload, so at most
  `--max-queued` + 1 requests are uploading, waiting or running. Uploads are
  received before a request takes its turn. A short request waits for the
  whole of a long one ahead of it.
- A client that disconnects cancels its request: a queued job leaves the queue
  without running; a running job stops at its next cancellation check
  (between decoded audio buffers, per ASR chunk/decode step, and in the
  diarizer's PLDA scoring), not instantly. Its temporary directory is removed
  either way.
- `SIGTERM` drains: the listener closes at once (health checks and new
  requests get connection refused, not `503`), the running job finishes, and
  the process exits 0. A 98 minute recording runs about 70 s on an M4 Max, so
  a supervisor's exit timeout (launchd's `ExitTimeOut`, default 20 s) shorter
  than the longest job kills it part-way. Set it deliberately.
- Nothing survives a restart. In-flight requests fail with the connection, and
  the service reloads and re-warms the models before reporting `ready`.
- The service never retries inference on its own, and neither does the Python
  client: after a timeout the original request may still be running.

## Deployment boundary

The service needs the models directory (read), its work directory (write) and
the listening socket. It does not need the repository, agent credentials,
`$HOME` beyond the default models path, or the container runtime. Bind it to
the interface the MCP server's network can reach (host loopback is not
reachable from a container by default) and terminate TLS, either with the
built-in `--tls-*` flags or a proxy, so the bearer token is never plaintext on
a shared network. Host installation, the launchd service, token provisioning
and firewall rules belong to `nixfiles` and are a separate step.

## Tests

```bash
swift test          # unit tests (reconciliation, admission, multipart) and the
                    # real server with a stub engine: mechanics only
```

The real engine is exercised by `tests/test_transcription_live.py` in the
repository root (`uv run pytest --slow -k transcription_live`), which builds
nothing: it expects this package built and the models provisioned.
