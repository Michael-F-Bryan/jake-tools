---
name: transcription
description: Transcribe and diarise local audio files with the `transcribe` tool; what the structured transcript promises about timestamps, file boundaries and anonymous speakers, and how to handle each error.
---

# Transcription

`transcribe(files)` uploads audio files the MCP server can read to the native
transcription service and returns one complete, structured transcript. All
inference is local to that host; nothing goes to a cloud provider. The tool is
only present when the server has `transcription.api_base_url` configured; if
you cannot see it, there is no transcription capability on this server.

## Calling it

- `files`: a non-empty list of paths, **as the MCP server sees them**, in
  recording order. The server opens and uploads them; it never asks the host
  to resolve your paths, and you never need to copy or encode audio.
- Several files are treated as **one continuous recording**: file 2 starts
  where file 1 ends. Pass separate meetings as separate calls.
- Anything AVFoundation decodes is accepted: WAV, AIFF, CAF, M4A/MP4 (AAC,
  ALAC or Opus), MP3, FLAC. **WebM and Ogg are refused** (`invalid_argument`
  saying so); they need converting to M4A, WAV or FLAC first. Limits
  (defaults): 64 files, 2 GiB upload, 3 hours of audio per call.
- The service runs **one job at a time** with a bounded queue. A long
  recording, or one behind another job, can take minutes; the call blocks until
  the transcript is ready or `transcription.timeout_seconds` passes.

## What comes back

`inputs` lists what was uploaded (your path, the byte size sent, index).
`transcript`:

| Field | Meaning |
| --- | --- |
| `timestamp_unit`, `timestamp_origin` | Seconds from the first sample of the concatenated recording. |
| `timing_resolution_seconds` | `0.08`: word timings are quantised to the engine's 80 ms frames. Do not read finer precision into them. |
| `duration_seconds` | Total decoded audio. |
| `files[]` | `index`, `name`, `start_seconds`, `end_seconds`, `duration_seconds` on that timeline. Subtract a file's `start_seconds` to get a position inside the original file. A word's edges can spread a few tenths of a second into the silence around it, so a word right at a boundary can start inside the previous file. |
| `speakers[]` | `id` (`speaker_1`, `speaker_2`, … by first appearance), total `speaking_seconds` from the diariser, `segment_count`. |
| `segments[]` | Runs of consecutive words with one `speaker`, split at speaker changes and pauses over 2 s. Each has `start_seconds`, `end_seconds`, `text` and `words[]`. |
| `words[]` | `text`, `start_seconds`, `end_seconds`, `confidence` (0–1), `speaker` (may be `null`), `overlap`. |
| `text` | The recogniser's full transcript. Segment text is rebuilt from timed words and may differ slightly in spacing or punctuation. |
| `unassigned_word_count` | Words no speaker turn covered or was within 0.5 s of; their `speaker` is `null`. |
| `limitations[]` | Engine limitations that apply to every result; read them once. |
| `processing` | Decode, ASR, diarisation and total seconds on the service. |
| `engine` | Engine and pinned model identities, for the record. |

What this does and does not promise:

- **Speakers are anonymous and request-scoped.** `speaker_1` is whoever spoke
  first in this request. It is not a person, and it is unrelated to
  `speaker_1` in any other call. Identify people from context only, and say so.
- **Speaker identity is continuous across file boundaries** within a call: a
  speaker in file 1 keeps their label in file 2.
- **Overlap is attributed, not resolved.** When two speakers' turns cover a
  word, it goes to the larger overlap and is flagged `overlap: true`. Treat such
  words, and anything with `speaker: null`, as uncertain attribution.
- **Results are complete or absent.** A request that cannot be fully
  transcribed fails. A file that is damaged or truncated (decoding stops
  early, or yields less audio than its container or header declares) is
  refused as `invalid_argument` naming the file, not transcribed in part.
- **Recognition can drop speech.** On real meetings the recogniser
  occasionally omits a whole sentence with no marker in the result. Do not
  treat an absent statement as proof it was not said.
- Quality has been checked against existing transcripts of real meetings
  (about 85–89% word agreement, about 93% speaker agreement) but never
  against hand-checked ground truth. Domain terms and acronyms are often
  misheard, and numbers come back as words.

## Errors

Every failure is a result with `isError: true` and a stable `code`:

| `code` | `detail.reason` | What to do |
| --- | --- | --- |
| `invalid_argument` | (file problems) `detail.file` names the path | The server cannot read that file, it is empty, or it is not decodable audio. Fix the path or the file. |
| `invalid_argument` | `invalid_input` | The service refused a file: unsupported container (WebM, Ogg), not audio, or damaged/truncated. The message names the file. Convert or replace it; retrying the same file gives the same answer. |
| `invalid_argument` | `payload_too_large` | Too many files, too many bytes, or too much audio for one call. Split the recording. |
| `missing_credentials` | | `TRANSCRIPTION_API_TOKEN` is unset on the server. Report it; nothing you pass fixes it. |
| `access_denied` | `unauthorized` | The token was rejected. Report it. |
| `capacity_exceeded` | `overloaded` | The queue is full. Wait at least `detail.retry_after_seconds`, then try again. |
| `upstream_error` | `not_ready` | The service is starting or its models failed to load. Wait for `retry_after_seconds` (starting) or report it (failed). |
| `upstream_error` | `unavailable` | The service is unreachable from the MCP server. Report it; do not loop. |
| `upstream_error` | `tls_failed` | The service answered but TLS failed, usually a certificate the MCP server does not trust. Report it as a configuration problem; do not loop. |
| `upstream_error` | `timeout` (`retry_safe: false`) | No answer within the configured time. **The service may still be processing the request.** Do not resubmit automatically; check with the user, and if you must retry, expect to wait for the earlier job too. |
| `upstream_error` | `inference_failed`, `bad_response` | The engine failed on this recording, or answered off-contract. Report it with the message. |

Never retry uncertain failures in a loop: each attempt queues more work on a
single-job service.
