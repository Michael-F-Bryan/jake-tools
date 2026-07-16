# Jake's Tools

Internal tools used by Jake.

## Installation

```bash
uv tool add -e .
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

## Daily report

```bash
jake-tools daily-report --date YYYY-MM-DD
jake-tools daily-report --date YYYY-MM-DD --json
```

The daily-report coordinator is deterministic and read-only outside its own work
directory:

- all six lanes always run: session hindsight, memory candidates, skill review,
  failure patterns, transcripts and DUM-C, and inbox triage
- inbox triage uses Himalaya envelope exports only; it does not read message
  bodies, draft replies, send mail, move mail, or delete mail
- the workflow does not mutate external systems: no memory, skills, email, cron,
  Obsidian, SharePoint, or git writes
- artefacts are written under `_working/daily-report-YYYY-MM-DD/`
- `--json` emits a machine-readable summary for automation
- `summary.json` includes token and estimated cost totals

## Transcribing Obsidian recordings

```bash
jake-tools transcribe obsidian-recording NOTE.md
jake-tools transcribe obsidian-recording --dry-run --json NOTE.md
```

The Obsidian recording pipeline always writes the same shape:

- `## Meeting Notes` with high-level dot points
- `## Chapters` with timestamps
- `## Transcript` with polished transcript text grouped by chapter

This command expects local `ffmpeg` and `scribe` executables to be available.

## Creating source notes from YouTube

```bash
jake-tools transcript youtube \
  'https://www.youtube.com/watch?v=VIDEO_ID' \
  --out-dir _working/youtube/VIDEO_ID

jake-tools transcript youtube \
  'https://www.youtube.com/watch?v=VIDEO_ID' \
  --out-dir _working/youtube/VIDEO_ID \
  --vault-note "$HOME/Documents/Vault/3 Resources/VIDEO_TITLE.md"
```

The command prefers authored captions in the requested language, then automatic
captions. It keeps selected public metadata, raw captions, raw and polished
transcripts, the rendered note, and a concise manifest under `--out-dir`. Vault
writes occur only after verification passes and only when `--vault-note` is
supplied without `--dry-run`.

Source notes contain provenance frontmatter, a summary callout, key points,
chapter summaries, clickable YouTube timestamps, and the polished transcript.
Videos without suitable captions currently fail explicitly rather than silently
starting a local audio transcription.

## Jira to Clockify sync

```bash
jake-tools clockify jira-sync --dry-run
jake-tools clockify jira-sync --dry-run --json
jake-tools clockify jira-sync --apply
```

The command reconciles Jira project `SF` with the active Clockify projects for
the `Sunfish Robotics` client. These defaults can be overridden with
`--jira-project` and `--clockify-client`.

Dry-run is the default. The plan can create projects and tasks, rename records,
reactivate tasks assigned to the current Jira user in an active status, and
mark tasks done when Jira's status category is Done. It ignores archived
Clockify projects and never deletes projects, tasks, or time entries. Ambiguous
active duplicates are reported as conflicts and block `--apply`.

The command requires:

- an authenticated `acli` session for Jira
- `CLOCKIFY_API_KEY`, or the Clockify group-level `--api-key` option

For unattended runs, inject `CLOCKIFY_API_KEY` through the scheduler's secret
environment. Do not depend on an interactive 1Password unlock in cron.
