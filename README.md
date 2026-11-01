# Jake's Tools

Internal tools used by Jake.

Tool-owned operational guidance ships with this package: see
[packaged skills, migration and resource delivery](src/jake_tools/skills/README.md).
List MCP resources to discover the focused `claude-delegation`, `clockify-time`
and `clockify-jira-sync` guidance by description, then read `skill://<name>`.
These resources replace the retired shared `jake-tools` and `clockify-api`
skills; no profile skill installation is needed. Supporting references are
separate resources and are not inlined into routine skill reads.

## Installation

```bash
uv tool install -e .
```

## Native Hermes plugin

The same operations the MCP server serves are also a native Hermes plugin,
`jake-tools` (entry point `jake_tools.hermes` in `hermes_agent.plugins`).
Other clients keep using the CLI and MCP server unchanged. Installed into
Hermes' own Python environment (nixfiles does this with the flake's
`lib.mkHermesPlugin`), it loads where a profile lists `jake-tools` in
`plugins.enabled`, and registers:

- **Tools**, for the capabilities in
  `plugins.entries.jake-tools.settings.capabilities` (`claude`, `clockify`,
  `transcription`; unset means none). They have the MCP tools' names,
  schemas, results and error codes; failures are
  `{"error", "code", "message", "detail"}`. `transcribe` also needs
  `transcription.api_base_url`. `ping` stays MCP-only.
- **Skills**, `jake-tools:<skill>` (via `skill_view`), from the same packaged
  sources, for the granted tools only.
- **The `direnv_local` terminal backend**, inert unless
  `terminal.backend: direnv_local` selects it.

Credentials are read per call from the owning profile with Hermes'
`agent.secret_scope.get_secret` (its `.env` and secret sources), consumed by
the operation and never returned. The launching process's environment is not
used for Jake's secrets, so one profile never gets another's. Delegated Claude
workers remain separate processes, spawned on Hermes' interpreter with only
`PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`, `TERM`, `SHELL`, `TMPDIR`, `XDG_*`,
`NIX_LD`, `NIX_LD_LIBRARY_PATH` and `PYTHONPATH` from Hermes' environment (what
the MCP server's worker saw), never Hermes' own credentials. Runs share
`$XDG_STATE_HOME/jake-tools/runs` with the MCP server. Transcription stays a
native host service reached over HTTP.

### The `direnv_local` backend

Hermes' local execution, plus each terminal command's project environment:
after Hermes restores its shell snapshot and enters the command's directory,
the backend sources the installer's loader (`JAKE_TOOLS_PROJECT_ENV_SCRIPT`,
which runs direnv with nix-direnv) in the same shell, then runs the command.
The tool arguments Hermes previews, approves and records are the command as
written. Directory changes and exports persist; the next command outside the
project unloads it. A present but unapproved or failing `.envrc` makes the
command fail (exit 1) with direnv's reason and does not run it. Because the
loader runs in the starting directory, leave a broken project with
`workdir` set elsewhere (e.g. `cd /` with `workdir: /`). Only the agent's
terminal commands (foreground and background) load projects; Hermes'
internal polls, kills and file reads do not. Jake's credential variables are
unset in every script the backend runs and listed in `strip_env_keys` for
Hermes' other spawns. Approval guards behave as for `local`.

Pinned Hermes (`5b32cfc`) decides some behaviour by the literal backend name
`local`, so under any plugin backend: background commands run through the
process registry's sandbox path (output polled from a log), `pty=true` is not
honoured, background processes take no stdin and are not recovered after a
gateway restart, a foreground command cannot yield to the background, and
`execute_code` uses its remote path rather than the session kernel. Under
`local` itself, Hermes does not strip plugin-declared credentials from
foreground commands; keep Jake's credentials out of the process environment
there.

Tests: `tests/test_hermes_plugin.py` (hermetic) runs by default.
`tests/test_hermes_runtime.py` is the acceptance test inside a real
installation: run `uv run pytest --slow tests/test_hermes_runtime.py` with
`JAKE_TOOLS_HERMES` (a built Hermes launcher including this package),
`JAKE_TOOLS_PROJECT_ENV_SCRIPT` and `JAKE_TOOLS_TEST_NIXPKGS` (a nixpkgs flake
reference); without them it reports itself blocked.

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

Dry-run is the default. Without `--issue`, the command reconciles planned or
underway Jira work assigned to `currentUser()`: an explicit status allowlist of
To Do, In Progress, Blocked, and In Review. To Do is SF’s Planned This Week
column; Backlog, TRIAGE, and Investigate are not automatically created. Existing
Clockify tasks deferred to the backlog stay intact and are still renamed as
needed; deferral is not completion. Repeat `--issue KEY` to reconcile exact work
items regardless of assignee or status. The plan can create projects and tasks,
rename records, reactivate eligible tasks, and mark tasks done when Jira’s status
category is Done (including Done and Won’t Do). It ignores archived Clockify
projects and never deletes projects, tasks, or time entries. Ambiguous active
duplicates are reported as conflicts and block `--apply`.

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

- `claude_start` starts a task with hard limits and two explicit lists:
  - `tools`, the Claude Code built-ins the session exposes. Any name is
    accepted, and the runtime decides what exists: a missing tool fails the
    task as `tools_unavailable`.
  - `allowed_tools`, optional permission rules such as
    `Bash(uv run pytest *)` that approve calls. Omitted, every exposed tool
    is approved.

  It also takes an idempotent `request_id`, an opt-in `allow_questions`,
  and `resume_task_id` to continue a finished task's conversation as a new
  task. [docs/claude-tools.md](docs/claude-tools.md) records the evidence
  behind this, and why there is no artifact-publishing tool.
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
upgrade note: restart every server sharing a runs directory together.

`claude_start` is also available as a native MCP Tasks (2026-07-28) task, for a
client that declares `io.modelcontextprotocol/tasks` in the request's own
capabilities: it then gets a `CreateTaskResult` and polls with `tasks/get`,
answers questions with `tasks/update` and cancels with `tasks/cancel`. Every
other client, including Hermes on its `initialize` connection, gets the
ordinary tools unchanged. Both are views of the same task.

## Transcription MCP tool

When `transcription.api_base_url` (or `TRANSCRIPTION_API_BASE_URL`) is set, the
stdio server also exposes `transcribe(files)`: it uploads audio files the MCP
server can read, in recording order, to the native transcription service in
[`transcription-service/`](transcription-service/README.md) and returns one
structured transcript (speaker segments with word timings in seconds from the
start of the concatenated recording, per-file spans, anonymous request-wide
speakers, limitations and processing times). Without an endpoint the tool is
absent. `TRANSCRIPTION_API_TOKEN` is the bearer token the service expects;
`transcription.timeout_seconds` (default 1800) bounds one call. The service
runs one job at a time with a bounded queue; a timeout is reported with
`retry_safe=false` and never retried, because the service may still be working
on the request. `jake-tools-mcp doctor` checks reachability, readiness and the
token. Guidance for agents is the `skill://transcription` resource.

The service itself is a Swift package built with Apple's toolchain
(`swift build`, then `transcription-service provision` once online, then
`transcription-service serve --token-file …`). `nix develop` provides the
remaining development tools (`uv`, `swift-format`, `sox`).

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
