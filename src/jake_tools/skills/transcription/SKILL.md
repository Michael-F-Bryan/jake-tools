---
name: transcription
description: Transcribe and diarise local audio files with the `transcribe` tool, and turn a saved result into a clean-verbatim review draft with `polish_transcript`; what each promises about timestamps, file boundaries, anonymous speakers and wording, what polishing sends to Claude, and how to handle each error.
---

# Transcription

`transcribe(files)` uploads audio files that jake-tools can read (the MCP server's, or for the
native Hermes tool Hermes' own, filesystem) to the native
transcription service and returns one complete, structured transcript. All
`transcribe` inference is local to that host; nothing goes to a cloud provider.
The tool is only present when jake-tools has `transcription.api_base_url`
configured; if you cannot see it, there is no transcription capability here.
`polish_transcript` (below) is different: it sends transcript text to Claude.

## Calling it

- `files`: a non-empty list of paths, **as jake-tools sees them**, in
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
- **Short excerpts can merge distinct speakers.** In live comparisons, a
  30-second conversation excerpt and a one-minute group-meeting excerpt each
  returned one speaker, while the matching passages within five-minute inputs
  had two and three speaker labels respectively. Identical decoded-audio
  controls supported a context-length effect. Prefer transcribing/diarising
  the full meeting (or a longer contiguous excerpt), then slicing the resulting
  timed transcript for polishing or review. Do not treat one speaker detected
  in a short clip as proof that only one person spoke, or invent missing speaker
  boundaries in text. These observations establish neither a universal minimum
  duration nor correct attribution in longer results.
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

## Polishing a saved transcript

`polish_transcript(transcript_path, output_directory, glossary?, start_seconds?,
end_seconds?, max_budget_usd?)` reads a saved `transcribe` result (the whole
JSON, or just its `transcript`) and writes a **new** directory: a readable
draft for a person to review, not a verified record.

- **Cloud boundary.** The transcript's words, timings, labels and any glossary
  go to Anthropic's Claude models through jake-tools' local Claude CLI
  (tool-less calls, no settings files or MCP servers). The audio never does.
  Do not polish a conversation that must not leave the host. It belongs to the
  `claude` capability, so a transcription-only grant does not include it.
- **Paths** are absolute, as jake-tools sees them. The source file is read
  once and never written (its SHA-256 is re-checked). `output_directory` must
  not exist (`invalid_argument` otherwise) and its parent must.
- **What changes.** Code removes standalone um/uh/erm. A conservative edit pass
  may also propose these, and code accepts only what it can check against the
  source words: a stutter (one identical adjacent copy always stays; never a
  repeated number, which may be a year or digits, nor a lone repeated
  negation or intensifier); a restart that the speaker then says again word
  for word, or a cut-off word they then finish ("pr produced"); a set-off
  "you know" / "I mean"; and punctuation or a first letter's case on one word
  (never a question mark, a name's capital or an all-caps word). Nothing may
  cross or empty a speaker turn, and stutters and restarts in unassigned or
  overlapping speech stay. There is no free replacement, no ordinary-word
  "repair" and no speaker change. Labels and timings are the source's (`null`
  shows as "Unassigned speaker"). Much of the readability comes from layout and
  filler removal; many restarts and false starts stay as spoken.
- **Glossary** (optional): `"TERM"` or `"TERM: what it is"` strings. Each
  occurrence is decided separately from its context and shown as `[TERM?]`,
  a tentative candidate, with the heard words listed. It is never a
  find-and-replace. Do not invent terms: pass only ones you know are used.
- **Review.** A separate detection-only pass lists material problems
  (`possible_misrecognition`, `unclear_meaning`, `fragment`,
  `attribution_doubt`) as `[R#]` markers and a list with timestamps and
  source words. It cannot supply wording. When it questions an edit, or flags
  a passage containing edits, those edits are undone and the source words
  shown (code's own um/uh/erm removal stays). It has no audio and is not exhaustive. An
  unflagged passage is **not** verified, and the draft cannot recover speech
  the recogniser dropped or speakers diarisation merged.
- **Known risk.** A meaningful repeat can look like a stutter when the
  recogniser drops the sentence boundary between them ("it's done. Done
  properly." heard as "it's done done properly"), so one copy may be removed,
  and the review may not notice. `ledger.json` keeps every removed word.
- **Files.** `transcript.md` (the draft), `ledger.json` (every source word by
  stable ID with its status and display, every edit with its source words, and
  every refused decision), `review.json`, `metadata.json` (policy, models,
  limits, per-chunk outcomes, cost; written last) and `diagnostics/` (each
  call's exact prompt and parsed reply or error).
- **Limits.** Work is split into chunks of about 900 words at speaker turns,
  two calls each, at most 16 chunks per call (about 1.5–2 hours of speech).
  For longer recordings, polish consecutive windows with
  `start_seconds`/`end_seconds`. Each call has its own cost and time cap. The
  whole call stays within `max_budget_usd` (default 5, at most 25) and finishes
  within about 270 s. A chunk whose pass failed, timed out or did not fit the
  budget or time is named in `chunks[]`, marked in the draft, and makes
  `status: "partial"`. A partial draft is still source-faithful (fillers only
  there) but unpolished or unreviewed where marked. Do not resubmit to the same
  `output_directory`; pick a new one.

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
| `invalid_argument` | (`polish_transcript`) `detail.argument` names it | A relative path, an existing `output_directory`, a file that is not a `transcribe` result, an empty window, a window too long for one call, or a bad glossary/budget. Fix the argument. |
| `upstream_error` | (`polish_transcript`) `detail.integration: claude` | Not one chunk could be edited (Claude unavailable, rate-limited, or out of budget). `detail.output_directory` holds the diagnostics; use a new directory for any retry. |

Never retry uncertain failures in a loop: each attempt queues more work on a
single-job service.
