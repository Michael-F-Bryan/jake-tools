# Jake's Tools

Internal tools used by Jake.

## Installation

```bash
uv tool install -e .
```

## Development

Install dependencies and the git hooks once:

```bash
uv sync
uv run pre-commit install
```

Every commit then runs `ruff check --fix`, `ruff format`, `pyright`, `pytest`,
and a `uv lock` consistency check (plus standard file-hygiene hooks). Run them
all manually at any time with:

```bash
uv run pre-commit run --all-files
```

## Jira to Clockify sync

```bash
jake-tools clockify jira-sync --dry-run
jake-tools clockify jira-sync --dry-run --json
jake-tools clockify jira-sync --apply
jake-tools clockify jira-sync --issue SF-304 --dry-run --json
jake-tools clockify jira-sync --issue SF-304 --apply --json
```

The command reconciles Jira project `SF` with the active Clockify projects for
the `Sunfish Robotics` client. These defaults can be overridden with
`--jira-project` and `--clockify-client`.

Dry-run is the default. Without `--issue`, the command reconciles active Jira
work assigned to `currentUser()`. Repeat `--issue KEY` to reconcile exact work
items regardless of assignee. The plan can create projects and tasks, rename
records, reactivate non-Done Jira tasks, and mark tasks done when Jira's status
category is Done. It ignores archived Clockify projects and never deletes
projects, tasks, or time entries. Ambiguous active duplicates are reported as
conflicts and block `--apply`.

JSON output identifies the selection scope and inventory counts. Applied writes
are verified by fetching the changed Clockify object again; the command fails
if the re-read does not match the planned name, status, or Jira project note.

The command requires:

- Jira REST credentials via `JIRA_BASE_URL`, `JIRA_EMAIL`, and
  `JIRA_API_TOKEN`, or their corresponding `jira-sync` options
- `CLOCKIFY_API_KEY`, or `jira-sync`'s own `--api-key` option

For unattended runs, inject these values through the scheduler's secret
environment or resolve `op://` references with `op run`. Do not persist the API
tokens or depend on an interactive 1Password unlock in cron.

## Transcription

```bash
jake-tools transcribe "2 Areas/Sunfish/2026-08-11 Team Sync.md"
jake-tools transcribe "...md" --assign "SPEAKER_03=Nikki Staltari"
jake-tools transcribe "...md" --assign "SPEAKER_04=Unknown" --finalise
```

The command runs the full meeting-transcription pipeline over one Obsidian
prep note in a single call: merges and transcribes the note's audio embeds
(or adapts a pre-diarised Gemini/Teams transcript embed when there's no
audio), resolves diarisation clusters to attendee names, chapterises the
raw transcript, polishes each chapter's dialogue, reviews the minutes with a
fresh adversarial pass, and exports an exact product candidate. It stops with
`status: "review_required"` and exit code 4; it never mutates the canonical
note until the human accepts that exact candidate.

Review and apply the candidate explicitly:

```bash
jake-tools transcript review product export "...md" --run-id RUN_ID
jake-tools transcript review product decide "...md" --run-id RUN_ID \
  --review-id REVIEW_ID --decision accept --reviewer "Michael Bryan"
jake-tools transcript integrate "...md" --run-id RUN_ID
```

`transcript integrate` is fail-closed: it checks the full source, artefact,
manifest, policy, note-base, renderer, and candidate-byte hashes before any
note, baseline, or deletion-fingerprint write. `## Chapters` and `## Transcript`
remain pipeline-owned; the summary callout and `## Discussion Notes` preserve
human edits through the existing three-way merge.

If speaker resolution can't confidently name every voice, the command prints
`{"status": "needs_input", "requests": [...]}` and exits with code 3 rather
than guess. Each request carries a couple of short audio snippets and the
surrounding resolved dialogue, for a human to identify (relayed over
Discord, or asked directly). Re-run with `--assign "SPEAKER_03=Name"` to
answer one cluster (repeatable), or `--assign "SPEAKER_04=Unknown"
--finalise` to give up on whatever's left and proceed anyway. ASR/diarisation
(and the pre-diarised-transcript adapter) check the run cache first, so that
`--assign` resume never redoes that expensive, HF-gated work; chapterising,
polishing, and generating minutes currently re-run their LLM calls if you
invoke the command again on an already-complete run (a known follow-up).

The individual pipeline stages are also available as `transcript` plumbing
sub-commands, each reading and writing the same run cache the porcelain
uses — useful for debugging one stage or re-running part of a pipeline by
hand:

- `transcript merge-audio` — merge a note's audio embeds into one recording
- `transcript asr` — local ASR + diarisation into the raw transcript
- `transcript adapt` — adapt a pre-diarised text transcript instead
- `transcript speakers` — resolve diarisation clusters to attendee names
- `transcript chapterise` — split the raw transcript into topic-based chapters
- `transcript polish` — clean up each chapter's dialogue, then check it adversarially
- `transcript minutes` — generate the meeting summary and Discussion Notes
- `transcript review product export` — export the exact candidate and checks
- `transcript review product decide` — record accept/reject for that candidate
- `transcript integrate` — apply only the accepted exact candidate

The command requires:

- `ffmpeg`/`ffprobe` on `PATH` (`brew install ffmpeg`) to merge and cut audio
- the Obsidian CLI (`obsidian`) on `PATH`, with the vault open, to resolve
  audio and transcript embeds
- `HF_TOKEN` for pyannote's gated diarisation model — only needed for a
  fresh ASR run over new audio; a cached run, or a run over a pre-diarised
  transcript source, needs neither ffmpeg nor a token

A coordinating agent without direct access to Michael's judgement can ask
Jake (his Hermes agent) for supporting context — which prep note to run
against, how to answer a `needs_input` request — via
`hermes chat --quiet --query '...'` (`--resume <session_id>` to continue a
session).
