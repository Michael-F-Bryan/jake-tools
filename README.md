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
jake-tools transcript recipe obsidian-recording NOTE.md --show-plan
```

The Obsidian recording pipeline always writes the same shape:

- `## Meeting Notes` with high-level dot points
- `## Chapters` with timestamps
- `## Transcript` with polished transcript text grouped by chapter

This command expects local `ffmpeg` and `scribe` executables to be available.
The `transcript recipe` variant exposes the same workflow over primitives with
`--show-plan`, `--workdir`, and optional `--manifest` output.

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

- an authenticated `acli` session for Jira
- `CLOCKIFY_API_KEY`, or the Clockify group-level `--api-key` option

For unattended runs, inject `CLOCKIFY_API_KEY` through the scheduler's secret
environment. Do not depend on an interactive 1Password unlock in cron.
