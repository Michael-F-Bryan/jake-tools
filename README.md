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
