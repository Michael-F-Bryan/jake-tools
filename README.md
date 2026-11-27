# Jake's Tools

Internal tools used by Jake.

Tool-owned operational guidance ships with this package: see
[packaged skills, migration and resource delivery](src/jake_tools/skills/README.md).
List MCP resources to discover the focused `claude-delegation`, `clockify-time`
and `clockify-jira-sync` guidance by description, then read `skill://<name>`.
These resources replace the retired shared `jake-tools` and `clockify-api`
skills; no profile skill installation is needed. Supporting references are
separate resources and are not inlined into routine skill reads.

## Issue preparation prompts

The MCP server exposes three prompts, each with one required `issue_url` argument:

- `issue_discovery`: establish the problem, existing behaviour and remaining uncertainties.
- `issue_planning`: propose the architecture, implementation sequence and evidence needed.
- `issue_maintainer_review`: judge whether the change is worth owning, then challenge its maintainability and incidental complexity.

Use them in that order through your client's MCP prompt picker, or via
`prompts/get` (for example, `{"name":"issue_discovery","arguments":{"issue_url":"https://github.com/owner/repo/issues/42"}}`).
Each asks the receiving agent to post its findings to the issue and verify the
comment. Retrieving a prompt only returns text: it does not launch an agent,
access the issue or post a comment. Hermes retrieves them through its prompt utility tools for this server.

## Installation

```bash
uv tool install -e .
```

## The agent interface: one MCP server

`jake-tools-mcp` (also `python -m jake_tools.mcp`) is a portable stdio MCP
server and the single agent-facing interface, for Hermes and every other MCP
client. The `jake-tools` CLI stays the operator interface over the same shared
operations. There is no Hermes plugin. The server provides:

- **Tools**, every operation, always listed, with stable
  `{code, message, detail?}` error results. No setting hides a tool: a missing
  prerequisite (for example `TRANSCRIPTION_API_BASE_URL` or
  `TRANSCRIPTION_API_TOKEN` for `transcribe`, or a Claude Code session that
  does not expose `Artifact` for `artifact_*`) is an error that names what to
  fix when the tool is called. `ping` is a connectivity check.
- **Guidance resources**: each guide is `skill://<name>` and its supporting
  references are `skill://<name>/references/<file>.md`, served from the
  installed package so they always match the running code. Each tool
  description names its guide.
- **Prompts**: `issue_discovery`, `issue_planning` and `issue_maintainer_review`
  (above).

Hermes consumes the server through its built-in MCP client (`mcp_servers`),
which prefixes tools and exposes per-server utility tools for listing and
reading resources and for listing and getting prompts. Nothing is imported
into Hermes.

Credentials and non-secret settings come only from the server's flags and its
own process environment (no `config.toml`, no `.env`; see
[Configuration](#configuration)), configured per agent environment: variables
such as `CLOCKIFY_API_KEY`, `JIRA_*`, `TRANSCRIPTION_API_TOKEN`,
`MICROSOFT_REFRESH_TOKEN` and their `_FILE` forms. The server never reads
another profile's secrets, delegated Claude workers never receive them, and a
missing credential fails only the call that needs it. Changing configuration or
credentials needs a reconnect (a fresh server process).

Package, `mcp_servers` configuration, credentials, refresh and cutover from the
old native plugin are in [docs/hermes-mcp.md](docs/hermes-mcp.md). The real-Hermes
acceptance test is `tests/test_hermes_mcp.py` (`--slow`, with `JAKE_TOOLS_HERMES`
set to a Hermes launcher; without it the test is blocked, not passed).

## Configuration

jake-tools is configured by **CLI flags and process environment variables,
and nothing else**: flags win over the environment, which wins over the
built-in default. There is no `config.toml` (not in `$XDG_CONFIG_HOME`, not in
`$XDG_CONFIG_DIRS`, not anywhere) and no `.env` loading: a `.env` or
`config.toml` in the working directory, its ancestors or `$HOME` has no effect
on the `jake-tools` CLI or on `jake-tools-mcp`. Run state still lives under
`$XDG_STATE_HOME/jake-tools` and caches under `$XDG_CACHE_HOME/jake-tools`.
Secrets are environment-only: `NAME`, or `NAME_FILE` naming a file whose
contents are the value (the direct variable wins); they have no flag and
`doctor` only ever prints `set`/`unset`.

> **Breaking change.** `config.toml` support was removed, with no fallback and
> no automatic migration, and `jake-tools` no longer loads `.env` files. An
> existing `config.toml` is silently ignored, so move each value to the
> variable in the table (or pass the flag) before upgrading, and have your
> process manager export the environment (`jake_tools.mcp doctor` prints
> every setting and its source). Previously the five `claude.model`, `.effort`,
> `.max_turns`, `.timeout_seconds` and `.max_budget_usd` settings were
> file-only; they now have the `JAKE_TOOLS_CLAUDE_*` variables below.

| Setting (as `doctor` prints it) | Environment variable | `serve` / `doctor` flag |
| --- | --- | --- |
| `claude.runs_dir` | `JAKE_TOOLS_RUNS_DIR` | `--runs-dir` |
| `session_store.path` | `JAKE_TOOLS_SESSION_STORE` | `--session-store` |
| `clockify.jira_project` | `JAKE_TOOLS_JIRA_PROJECT` | `--jira-project` |
| `clockify.client` | `JAKE_TOOLS_CLOCKIFY_CLIENT` | `--clockify-client` |
| `clockify.api_base_url` | `CLOCKIFY_API_BASE_URL` | `--clockify-api-base-url` |
| `clockify.workspaces` | `JAKE_TOOLS_CLOCKIFY_WORKSPACES` (comma-separated IDs) | `--clockify-workspace` (repeat) |
| `claude.model` | `JAKE_TOOLS_CLAUDE_MODEL` | `--claude-model` |
| `claude.effort` | `JAKE_TOOLS_CLAUDE_EFFORT` | `--claude-effort` |
| `claude.max_turns` | `JAKE_TOOLS_CLAUDE_MAX_TURNS` | `--claude-max-turns` |
| `claude.timeout_seconds` | `JAKE_TOOLS_CLAUDE_TIMEOUT_SECONDS` | `--claude-timeout-seconds` |
| `claude.max_budget_usd` | `JAKE_TOOLS_CLAUDE_MAX_BUDGET_USD` | `--claude-max-budget-usd` |
| `claude.max_live_tasks` | `JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS` | `--claude-max-live-tasks` |
| `claude.account_context` | `JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT` | `--claude-account-context` |
| `hindsight.api_url` | `JAKE_TOOLS_HINDSIGHT_API_URL` | `--hindsight-api-url` |
| `hindsight.bank_id` | `JAKE_TOOLS_HINDSIGHT_BANK_ID` | `--hindsight-bank-id` |
| `hindsight.access` | `JAKE_TOOLS_HINDSIGHT_ACCESS` (`read` or `readwrite`, default) | `--hindsight-access` |
| `transcription.api_base_url` | `TRANSCRIPTION_API_BASE_URL` | `--transcription-api-base-url` |
| `transcription.timeout_seconds` | `JAKE_TOOLS_TRANSCRIPTION_TIMEOUT_SECONDS` | `--transcription-timeout-seconds` |
| `microsoft.tenant_id` | `JAKE_TOOLS_MICROSOFT_TENANT_ID` | `--microsoft-tenant-id` |
| `microsoft.client_id` | `JAKE_TOOLS_MICROSOFT_CLIENT_ID` | `--microsoft-client-id` |
| `microsoft.account` | `JAKE_TOOLS_MICROSOFT_ACCOUNT` | `--microsoft-account` |
| `microsoft.account_object_id` | `JAKE_TOOLS_MICROSOFT_ACCOUNT_OBJECT_ID` | `--microsoft-account-object-id` |
| `microsoft.authority_url` | `JAKE_TOOLS_MICROSOFT_AUTHORITY_URL` | (none; test stand-in) |
| `newsletters.graph_root` | `JAKE_TOOLS_NEWSLETTERS_GRAPH_ROOT` | (none; test stand-in) |
| `newsletters.sharepoint_root` | `JAKE_TOOLS_NEWSLETTERS_SHAREPOINT_ROOT` | (none; test stand-in) |
| `files.graph_root` | `JAKE_TOOLS_FILES_GRAPH_ROOT` | (none; test stand-in) |
| `files.sharepoint_hosts` | `JAKE_TOOLS_FILES_SHAREPOINT_HOSTS` (comma-separated) | `--files-sharepoint-host` (repeat) |

Secret environment variables (each also as `NAME_FILE`): `CLOCKIFY_API_KEY`,
`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `TRANSCRIPTION_API_TOKEN`,
`MICROSOFT_REFRESH_TOKEN`, `MICROSOFT_CLIENT_SECRET`. Other inputs: `HOME`,
`XDG_STATE_HOME`, `XDG_CACHE_HOME` and `HERMES_HOME` (the default session
store is `$HERMES_HOME/state.db`).

The flags belong to `jake-tools-mcp serve` and `jake-tools-mcp doctor` (the
bare `jake-tools-mcp` takes none: name `serve`). The `jake-tools` CLI keeps its
own existing flags, each with its environment variable: `clockify`'s
`--api-key`/`--api-base-url` (`CLOCKIFY_API_KEY`, `CLOCKIFY_API_BASE_URL`),
`--jira-*` (`JIRA_*`), and `clockify jira-sync`'s `--jira-project` and
`--clockify-client` (`JAKE_TOOLS_JIRA_PROJECT`, `JAKE_TOOLS_CLOCKIFY_CLIENT`).
`jake-tools files` reads the Microsoft settings from the environment only.

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
items regardless of assignee or status; a missing task for a Done issue is
created and then completed, both shown in the plan. The plan can create projects and tasks,
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
- `clockify_workspace_catalogue(workspace_id?, keys?, client_ids?, project_ids?)`:
  without `keys`, all active/archived clients and projects, and ACTIVE/DONE tasks
  with an explicit `completed` flag (large). With `keys` (1..50 Jira keys), an
  exact work-item lookup (a task name must start with the key; a project owns only the first Jira key in its note): one result per key, `resolved` (one task or project
  with its project/client and archived/completed flags), `ambiguous` (all
  candidates, none preferred), `not_found` or `incomplete` (part of the scope
  could not be read). Never a similar-name match. `client_ids`/`project_ids`
  declare the scope; `coverage` reports it, unread projects and the number of
  upstream requests. Scope, not `keys`, is what reduces upstream requests.
- `clockify_create_time_entry(start, end, description, billable, workspace_id?,
  project_id?, task_id?)`: creates a completed entry, requires aware timestamps
  and `end > start`, and validates task/project ownership. Makes one POST, then
  GET-verifies the ID, user/workspace, interval, description, associations and
  billable flag. It never starts a timer or retries a write.

These tools require only `CLOCKIFY_API_KEY` (not Jira credentials), plus a mandatory
workspace allowlist in the MCP server's environment:

```sh
JAKE_TOOLS_CLOCKIFY_WORKSPACES=60176d9aa84b61134aaef479   # comma-separated IDs
```

Or pass `--clockify-workspace ID` (repeatable) to `serve`. Omitted
workspace defaults to the first allowlisted ID; an empty allowlist or an outside
workspace is `access_denied`. IDs accept only letters, digits, `_` and `-`.

History scans **all** the authenticated user's entries without date filters on
**every** call, then filters overlaps locally. This includes arbitrarily old
straddling/running entries, but costs requests/memory proportional to all history.
Coverage reports upstream paths, filters, page counts and record counts. Missing
names fail the request rather than returning an apparently complete result.
Each upstream list is bounded at 1,000 pages of 5,000 records; repeated IDs,
oversized pages or reaching the bound are errors, never successful truncation.
Such an `upstream_error` carries `detail.complete: false` and `detail.failure`
(`reason`, `resource`, `page`, `page_size`, `page_records`, `records_seen`,
`duplicate_count`): counts only, never record IDs, entry content or credentials.
There is no automatic restart or deduplication: an offset scan that repeated IDs
may also have skipped records unseen; a caller's retry of the whole call is only as
trustworthy as any clean, non-snapshot scan. Whether a bounded restart is defensible is
unresolved (jake-tools#13).
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
  behind this, and the evidence for the experimental artifact tools.
- `claude_poll` waits (up to 120 s) until the task needs its caller (a
  question, the end, a new denial or warning, undeliverable steering) and
  returns a compact snapshot with an observation token: progress, pending
  questions, the changes since the caller's token, and the bounded result.
  It never pages history: `claude_transcript` does (no private thinking).
- `claude_steer` adds an idempotent message to the running conversation and
  reports delivery honestly (`queued`, `delivered`, `in_conversation`,
  `not_delivered`).
- `claude_update` answers the worker's questions.
- `claude_status` returns the detail (configuration as requested,
  normalised and exposed; usage; denials; messages) or the full result text
  in chunks; `claude_cancel` stops the whole process tree and says whether
  termination was confirmed (never a rollback).

How many tasks run at once is configurable: `JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS`
or `--claude-max-live-tasks` (a positive integer;
default 2; `doctor` prints it). A start while that many are live is refused
with `capacity_exceeded` (`detail.live`, `detail.limit`; there is no queue).
The live count is shared by every process using one runs directory but each
process applies its *own* limit, so differently configured processes do not
share a ceiling: give them all the same value. Lowering the limit below the
live count cancels nothing; starts are refused until fewer are live. More
local capacity does not add provider quota. The packaged
[`claude-delegation` skill](src/jake_tools/skills/claude-delegation/SKILL.md)
documents the contract, including retention, the local trust model and the
upgrade note: restart every server sharing a runs directory together.

`claude_start` is also available as a native MCP Tasks (2026-07-28) task, for a
client that declares `io.modelcontextprotocol/tasks` in the request's own
capabilities: it then gets a `CreateTaskResult` and polls with `tasks/get`,
answers questions with `tasks/update` and cancels with `tasks/cancel`. Every
other client, including Hermes on its `initialize` connection, gets the
ordinary tools unchanged. Both are views of the same task.

### Shared Hindsight memory for workers

With `hindsight.api_url` and `hindsight.bank_id` both set (neither has a
default, and there is no fallback bank), every delegated worker gets the
Hindsight server and bank of the agent that started it, as one explicit HTTP
MCP server at `<api_url>/mcp/<bank_id>/` (Hindsight 0.10.1's single-bank route:
no tool can name another bank). Approved: `recall`, `reflect`, and `retain`
unless `hindsight.access=read`. The route's administrative and deleting tools
are listed but refused. This is not the Hindsight plugin or hooks, ambient MCP
inheritance or a settings file; nothing is retained automatically; no prompt is
added; the `max_turns` handoff stays tool-free. The URL is HTTPS or loopback
HTTP, and carries no credential (the pod's server is unauthenticated and
loopback-only). `doctor` lists the route's tools without calling any and fails
only when just one of the two settings is set. A worker with `Bash` can still
reach the server directly; the bank limit is of the MCP route, not a sandbox.

## Newsletter MCP tools (Microsoft)

`newsletter_list`, `newsletter_get`, `newsletter_create` and `newsletter_update`
read and change the CSU Weekly Newsletter SharePoint list, including
attachment-only updates, as **one explicitly configured Microsoft account**
through a delegated, refresh-aware grant (tenant, application and account are
configuration; a grant for any other account is refused, and there is no
fallback to `az login`). Lists are one bounded page with explicit
coverage (`complete`, `next_cursor`). Every write reads the entry back and
downloads each attachment to compare bytes; a failure after the entry exists
returns its ID and an applied/failed/uncertain/not-attempted effect per step,
is never retried, and is recovered with `newsletter_update`, never a second
create. Secrets are environment-only (`MICROSOFT_REFRESH_TOKEN[_FILE]`,
optional `MICROSOFT_CLIENT_SECRET[_FILE]`). Provisioning, reauthorisation and
the identity each call uses: [`docs/microsoft-auth.md`](docs/microsoft-auth.md);
agent guidance: `skill://newsletters`. The tests run the real server and clients
against a local fake of the token endpoint, Graph and SharePoint; they do not
establish live Microsoft compatibility. The `jake-tools newsletter` command
is separate and still uses the caller's own `az login`.

## SharePoint and Teams file MCP tools

`sharepoint_resolve`, `sharepoint_get`, `sharepoint_list`, `sharepoint_download`,
`sharepoint_pdf` and `sharepoint_upload` move files between the host and a
SharePoint/Teams library as the same configured Microsoft account as the newsletter
tools (the application additionally needs the delegated `Files.ReadWrite.All`
permission). A SharePoint link on an allowed host (`files.sharepoint_hosts`) or an
explicit team and channel resolves to a drive item; every item carries its
drive/item IDs, URL and `etag`/`ctag` version, and a folder listing is one page
with explicit coverage. A download is saved only if the version did not change
while reading it; PDF rendering is Microsoft's own, with unsupported source types
refused up front. A new file chooses `fail` or `rename` on a name clash and never
overwrites; it is read back and compared byte for byte (`verified: true`), and a
write that may or may not have happened is `write_uncertain` and is never retried.
**Replacing an existing file is deliberately not offered**: Microsoft documents
`if-match` only when an upload session is created, not at the commit, so a
replacement could silently overwrite an edit made in between (issue #11 keeps
that criterion open). Guidance: `skill://sharepoint-files`; the same operations are
available to an operator as `jake-tools files ...`. The tests run the real server,
clients and token provider against a local fake of the token endpoint and Graph;
they do not establish live Microsoft compatibility.

## Transcription MCP tool

The stdio server always exposes `transcribe(files)`; until
`TRANSCRIPTION_API_BASE_URL` (or `--transcription-api-base-url`) and
`TRANSCRIPTION_API_TOKEN` are set it answers `missing_credentials` naming the
variable. Once configured it uploads audio files the MCP
server can read, in recording order, to the native transcription service in
[`transcription-service/`](transcription-service/README.md) and returns one
structured transcript (speaker segments with word timings in seconds from the
start of the concatenated recording, per-file spans, anonymous request-wide
speakers, limitations and processing times). `TRANSCRIPTION_API_TOKEN` is the bearer token the service expects;
`transcription.timeout_seconds` (default 1800) bounds one call. The service
runs one job at a time with a bounded queue; a timeout is reported with
`retry_safe=false` and never retried, because the service may still be working
on the request. `jake-tools-mcp doctor` checks reachability, readiness and the
token. Guidance for agents is the `skill://transcription` resource.

The service itself is a Swift package built with nixpkgs' Swift and Apple SDK
(`nix build .#transcription-service`); its pinned models are another output
(`nix build .#transcription-models`), passed read-only as `serve --models-dir`.
`nix develop` provides the remaining development tools (`uv`, `swift-format`,
`sox`).

### Polishing a saved transcript

`polish_transcript(transcript_path, output_directory, glossary?, start_seconds?,
end_seconds?, max_budget_usd?)` turns a saved `transcribe` result into a
clean-verbatim Markdown draft in a new directory, with a word ledger, review
items, metadata and the exact prompts and replies. It sends the transcript's
**text** (never audio) to Claude through the local CLI. That is why it is a
`claude` toolset and not `transcription`, and why it needs no
transcription endpoint. Per chunk of about 900 words, a conservative edit pass
proposes filler, stutter and restart deletions, punctuation and optional
glossary candidates, and code validates each against the source words. A
detection-only review then flags places to check. Neither can rewrite
wording or speakers. Limits: 16 chunks per call, `max_budget_usd` (default 5),
about 270 s. Code is in `src/jake_tools/polishing/`; contract and caveats are
in `skill://transcription`; `tests/test_polish_transcript_live.py` (`--slow`)
is the real-model acceptance.

### Publishing an HTML file as a private Claude Artifact (experimental)

`artifact_publish(file_path)` publishes an existing `.html`/`.htm` file as a
**new** private artifact on claude.ai through the signed-in Claude account and
verifies it in a separate session; `artifact_read(url, file_path?,
expected_version_id?, output_path?)` reads one back without changing anything
(`jake-tools artifact publish|read` do the same from a shell; `read` exits
nonzero when `--file` or `--expected-version-id` is given and does not hold).
The file's contents are sent to Anthropic.

It rests on an **undocumented** Claude Code mechanism: the SDK hides the
`Artifact` tool unless `CLAUDE_CODE_ARTIFACT` is set, which Jake does for these
sessions only. It may change without notice, but there is no switch: the tools are
always listed, and if Claude Code does not expose `Artifact` (signed out, policy,
`enableArtifact: false`, or the route changed) the call fails with
`upstream_error`/`tool_not_exposed` before anything is published. Each session is guarded before every
tool call by a `PreToolUse` hook that judges the raw model input, so it can only
read the one file and publish it once (or read the one URL); a prompt is not what
keeps it there. `verified` means the hosted file,
with Claude Code's fixed document shell removed, equals the local file and the
readback reported `private`. Sharing, listing and deleting are not supported.

`artifact_update(url, file_path, expected_version_id)` (`jake-tools artifact
update URL FILE --expected-version-id V`) publishes a file as the **next
version of an existing private artifact you own**, keeping its URL and id, under
`artifact_update` (it overwrites live content). A separate read
first checks the artifact is yours, private and at the version you expect; a
stale version fails with `plan_stale` and writes nothing. Afterwards another
read checks the same artifact, unchanged visibility, the new version and the
byte-identical file. The write session may publish only after its own read of
the target was observed as yours, private and at the expected version. It is
**not** an atomic compare-and-swap (native read and write are separate calls):
an edit landing between that read and the publish is not detectable, and Claude
Code's own stale-copy refusal (untested here) is the only protection. Only private artifacts are updated; no sharing/ACL, no general CRUD. Evidence and caveats: `docs/claude-tools.md`; usage:
`skill://artifact-publishing`; code: `src/jake_tools/artifacts/`;
`tests/test_artifacts_live.py` (`--slow`, `live`) needs an existing target for
readback and refuses to create an artifact unless explicitly told to.

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
