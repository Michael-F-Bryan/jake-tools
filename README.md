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
