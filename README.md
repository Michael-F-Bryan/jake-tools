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

## Choosing a model

Commands that call an LLM take `--model` (a Claude model ID) and `--effort`.
Calls go through the Claude Agent SDK, which drives the local `claude` CLI, so
the same credentials and quota apply as when you run Claude Code by hand.

```bash
jake-tools transcript polish --model claude-opus-4-8 --effort high TRANSCRIPT.txt
```

## Transcribing Obsidian recordings

```bash
jake-tools transcript obsidian-recording NOTE.md
jake-tools transcript obsidian-recording --dry-run --json NOTE.md
jake-tools transcript obsidian-recording --work-dir _working/obsidian/NOTE NOTE.md
```

The Obsidian recording pipeline always writes the same shape:

- `## Meeting Notes` with high-level dot points
- `## Chapters` with timestamps
- `## Transcript` with polished transcript text grouped by chapter

This command expects local `ffmpeg` and `scribe` executables to be available.
It rewrites `NOTE.md` in place only after note verification passes; `--dry-run`
runs the full pipeline (transcription, speaker mapping, polish, chaptering,
minutes) without that final write. Without `--work-dir`, intermediate
artefacts (audio, transcripts, chapters, a manifest) live in a temporary
directory that is gone once the command returns; pass `--work-dir` to keep
them for inspection, matching the youtube/teams-meeting `--out-dir` convention.

`jake-tools transcribe ...` still works as a hidden, deprecated alias for
`jake-tools transcript obsidian-recording` / `jake-tools transcript polish` —
switch existing scripts to `transcript` when convenient.

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
- `CLOCKIFY_API_KEY`, or the Clockify group-level `--api-key` option

For unattended runs, inject these values through the scheduler's secret
environment or resolve `op://` references with `op run`. Do not persist the API
tokens or depend on an interactive 1Password unlock in cron.
