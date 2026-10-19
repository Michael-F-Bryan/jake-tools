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

## Clockify MCP tools

The stdio server (`jake-tools-mcp`) also provides:

- `clockify_time_entries(start, end, workspace_id?, timezone="UTC", limit=100,
  cursor?)`: authenticated-user entries overlapping timezone-aware `[start,end)`.
  Original timestamps and null ends/running status are preserved, with project/task
  IDs and resolved names. Results sort by start then ID; cursors are query-bound
  and reject changed selected history. Running entries use a fixed `observed_at`
  across continuation pages.
- `clockify_workspace_catalogue(workspace_id?)`: all active/archived clients and
  projects, and ACTIVE/DONE tasks with an explicit `completed` flag. IDs remain
  available for creating entries.
- `clockify_create_time_entry(start, end, description, billable, workspace_id?,
  project_id?, task_id?)`: creates a completed entry, requires aware timestamps
  and `end > start`, and validates task/project ownership. Makes one POST, then
  GET-verifies the ID, user/workspace, interval, description, associations and
  billable flag. It never starts a timer or retries a write.

These tools require only `CLOCKIFY_API_KEY` (not Jira credentials), plus a mandatory
workspace allowlist in MCP configuration:

```toml
[clockify]
workspaces = ["your-workspace-id"]
```

Alternatively set `JAKE_TOOLS_CLOCKIFY_WORKSPACES` to comma-separated IDs. Omitted
workspace defaults to the first allowlisted ID; an empty allowlist or an outside
workspace is `access_denied`. IDs accept only letters, digits, `_` and `-`.

History scans **all** the authenticated user's entries without date filters on
**every** call, then filters overlaps locally. This includes arbitrarily old
straddling/running entries, but costs requests/memory proportional to all history.
Coverage reports upstream paths, filters, page counts and record counts. Missing
names fail the request rather than returning an apparently complete result.
Each upstream list is bounded at 1,000 pages of 5,000 records; repeated IDs,
oversized pages or reaching the bound are errors, never successful truncation.
Upstream pagination is not an atomic snapshot: concurrent changes can still affect
coverage. No live-account overlap or API compatibility validation has been done.

Create errors report `post_uncertain`, `verification_unavailable`, or
`verification_failed`, with `created_entry_id` when known and `retry_safe=false`.
A failed POST (even HTTP 500) may have committed. Inspect history/the known ID
before deciding whether to try again; **never automatically retry**.

## Claude delegation MCP tools

The stdio server also runs bounded Claude Code tasks in the background, each in
its own process and run directory (`$XDG_STATE_HOME/jake-tools/runs/<task_id>/`),
so they survive server restarts:

- `claude_start` starts a task with an explicit tool grant and hard limits. It
  also takes an idempotent `request_id`, an opt-in `allow_questions`, and
  `resume_task_id` to continue a finished task's conversation as a new task.
- `claude_poll` waits at most 25 s for change and returns state plus
  cursor-paged normalised events (no private thinking). `claude_transcript`
  returns larger pages.
- `claude_steer` adds an idempotent message to the running conversation and
  reports delivery honestly (`queued`, `delivered`, `in_conversation`,
  `not_delivered`).
- `claude_update` answers the worker's questions.
- `claude_status` returns a snapshot; `claude_cancel` stops the whole
  process tree.

At most two tasks run at once. The packaged
[`claude-delegation` skill](src/jake_tools/skills/claude-delegation/SKILL.md)
documents the contract, including retention, the local trust model and the
upgrade note: restart every server sharing a runs directory together. Native
MCP Tasks (2026-07-28) responses are not offered. The projection is
implemented in `claude_runs/tasks_protocol.py` for when a host negotiates the
extension.

## Codex usage alerts

```bash
jake-tools codex-usage-alert
```

The command reads the existing OpenAI Codex OAuth access token from Hermes'
`auth.json`, fetches the five-hour and weekly usage windows, and prints only
when a window first crosses 20%, 10%, or 5% remaining. It stores only the last
observed threshold and reset time under `~/.hermes/cron/state/`; OAuth tokens
remain owned and refreshed by Hermes.

A silent successful run means no new threshold was crossed. Authentication or
API failures are reported on stderr with a non-zero exit status so a script-only
scheduler can deliver them as alerts.
