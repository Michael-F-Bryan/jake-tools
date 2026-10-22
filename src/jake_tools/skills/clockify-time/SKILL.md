---
name: clockify-time
description: Read Clockify history and create verified time entries.
version: 0.1.0
author: Michael F. Bryan (Michael-F-Bryan), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
---

# Clockify time

Use the `jake-tools` MCP server to read time history, select existing projects/tasks,
and create completed time entries. No Jira credentials or CLI commands are needed.
Hermes exposes the tools with its `mcp_<server>_` prefix; names below are bare names.

## Discover the current interface

Use the host's tool search/list and schema-description facility to find these
bare tool names (prefixes depend on the host). Inspect the live input schema
before calling; do not assume CLI flags are MCP arguments. Read
`skill://clockify-time` for guidance matching the running server when a local
copy may be stale. Tools use the server's configured credentials, not the agent's
shell environment; report missing credentials/allowlisting rather than probing
local secret stores or switching accounts.

| Tool | Effect |
| --- | --- |
| `clockify_time_entries` | Read-only authenticated-user overlap history; page to completion. |
| `clockify_workspace_catalogue` | Read-only clients, projects and tasks; no Jira access needed. |
| `clockify_create_time_entry` | Real write: one completed entry, no preview/apply parameter. |

Unsupported entry edits/deletes, timer changes, project/client cleanup and
exact-ID bulk migration are not MCP tools. Only when explicitly requested, read
[direct API workflows](references/direct-api-workflows.md), also available as
`skill://clockify-time/references/direct-api-workflows.md`. That supporting
reference requires separate authorised credential access and stops if absent;
it is not a separate skill or a routine fallback for a tool error.

## When to use

- Review logged work over an interval, including running entries.
- Find existing project/task IDs, including archived or completed work.
- Log completed work with the user's chosen description and billable setting.
- Do not use to start/stop timers or to edit/delete entries.

## Prerequisites

The server environment needs `CLOCKIFY_API_KEY`. Configure `clockify.workspaces`
in its config.toml or comma-separated `JAKE_TOOLS_CLOCKIFY_WORKSPACES`. The
allowlist is mandatory; an omitted workspace means its first ID, not the user's
active workspace. IDs must contain only letters, digits, `_` and `-`.

## Procedure

1. Read `clockify_time_entries` with aware ISO `start` and `end`, e.g.
   `2026-10-14T00:00:00+08:00` to `2026-10-15T00:00:00+08:00`.
   Selection is half-open overlap, not containment. Optional `timezone` is an
   IANA display zone (default UTC), `limit` is 1..500 (default 100).
2. Consume `entries`, checking each nested `entry` for original `time_interval`,
   IDs, description and billable state. Outer fields include `running`,
   `project_name`, `task_name`, `display_start`, and `display_end`. A running
   entry keeps a null end; do not invent one or count it as completed work.
3. Follow `next_cursor` using exactly the same workspace/range/timezone/limit.
   `count` is this page; `total` is the complete selected count. `observed_at`
   fixes running-entry evaluation across pages. If selected history changes,
   restart without the cursor instead of combining incompatible pages.
4. Read `clockify_workspace_catalogue` before choosing project/task IDs. Clients
   and projects include `archived`; tasks include `status` and `completed` plus
   `project_id`. Do not guess IDs or silently substitute a similarly named task.
5. For a user-approved completed log, call `clockify_create_time_entry` with aware
   `start`, `end > start`, `description`, explicit `billable`, and optional
   `project_id`/`task_id`. A task requires its owning project. This is a real
   write: confirm ambiguous dates, associations or billable intent first.
6. Accept success only with `verified=true`: the tool POSTs once and re-reads
   all relevant fields. On error inspect `detail.outcome`, `created_entry_id`
   and `retry_safe`. Never automatically retry; even `post_uncertain` can mean
   the entry was created. Check history or the known entry in Clockify first.

## Pitfalls

History scans ALL authenticated-user entries without date filtering on every
call, including arbitrarily old straddling entries. It can be expensive for a
large account. Coverage reports list paths, filters, pages, records and strategy.
Names must resolve or the call fails; no silent name gaps. Each upstream list is
bounded at 1,000 pages of 5,000 records and errors rather than truncating. Pagination
is not an atomic snapshot; concurrent writes can affect coverage despite cursor
change detection. No live-account API compatibility check has been performed.

An empty/outside allowlist yields `access_denied`; missing API key yields
`missing_credentials`; invalid timestamps, IDs or cursors yield `invalid_argument`.
Remote failures yield `upstream_error`. Create outcomes are `post_uncertain`,
`verification_unavailable` (known ID, GET unavailable), or `verification_failed`
(known ID, re-read differs). No automatic retries are safe.

## Verification

Finish a history review only after `next_cursor` is null and the enumerated count
matches `total`. Preserve original boundaries (never clip returned entries).
Finish a write only when the returned completed `entry` matches the approved log
and `verified` is true; otherwise report the uncertainty and known ID.
