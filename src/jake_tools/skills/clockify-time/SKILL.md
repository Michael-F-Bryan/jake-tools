---
name: clockify-time
description: Read Clockify history and create verified time entries.
version: 0.1.0
author: Michael F. Bryan (Michael-F-Bryan), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
---

# Clockify time

Use jake-tools to read time history, select existing projects/tasks, and create
completed time entries. No Jira credentials or CLI commands are needed. The tools
are served by the `jake-tools` MCP server; names below are bare names (an MCP
client may prefix them, as Hermes does with `mcp__<server>__`).

## Discover the current interface

Use the host's tool search/list and schema-description facility to find these
bare tool names (prefixes depend on the host). Inspect the live input schema
before calling; do not assume CLI flags are tool arguments. Read this guide from
the running server as the MCP resource `skill://clockify-time`, through your
client's MCP resource reader (in Hermes: the server's `read_resource` utility tool, `mcp__<server>__read_resource`, with `uri`)
when a local copy may be stale. Tools use jake-tools' configured credentials, not the agent's
shell environment; report missing credentials/allowlisting rather than probing
local secret stores or switching accounts.

| Tool | Effect |
| --- | --- |
| `clockify_time_entries` | Read-only authenticated-user overlap history; page to completion. |
| `clockify_workspace_catalogue` | Read-only. With `keys`: exact work-item lookup (compact, per-key outcome). Without: the full clients/projects/tasks catalogue. No Jira access needed. |
| `clockify_create_time_entry` | Real write: one completed entry, no preview/apply parameter. |

Unsupported entry edits/deletes, timer changes, project/client cleanup and
exact-ID bulk migration are not jake-tools tools. Only when explicitly requested, read
[direct API workflows](references/direct-api-workflows.md), also available as
the MCP resource `skill://clockify-time/references/direct-api-workflows.md`.
That supporting
reference requires separate authorised credential access and stops if absent;
it is not a separate skill or a routine fallback for a tool error.

## When to use

- Review logged work over an interval, including running entries.
- Find existing project/task IDs, including archived or completed work.
- Log completed work with the user's chosen description and billable setting.
- Do not use to start/stop timers or to edit/delete entries.

## Prerequisites

jake-tools needs `CLOCKIFY_API_KEY` (the MCP server's environment). Configure `clockify.workspaces`
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
4. Find project/task IDs for known Jira keys with `clockify_workspace_catalogue`
   and `keys` (1..50, e.g. `["SF-304", "SF-310"]`). Do not fetch the full
   catalogue for a handful of keys. Each distinct key gets one result, in
   request order:
   - `resolved`: exactly one candidate. A `task` candidate is a task whose
     name starts with the key at its first character, no leading whitespace
     (`SF-304 Summary`), with its `project` and `client`; a `project` candidate
     is a project whose note's FIRST Jira key is the key (`Jira: SF-304`; a
     second key in the same note owns nothing) and has no task. Use `project.id` and `task.id` for
     `clockify_create_time_entry`.
   - `ambiguous`: several candidates, none preferred. Show them (IDs, names,
     client, archived/completed) and ask which; never take the first. At most
     10 are listed: `candidate_count` is the total and `truncated` says some
     were left out. This bounds the answer; it does not guarantee it fits a
     host's inline result limit.
   - `not_found`: absent from the declared scope, definitively.
   - `incomplete`: part of the scope could not be read (`coverage.unread`
     names each project and why: a read failure, a pagination fault, or a task
     whose status changed between the ACTIVE and DONE reads, which is never
     resolved by picking one. After 20 failed projects reading stops and the
     rest are `coverage.unattempted`). Absence and uniqueness are unproven: any
     `candidates` are evidence only. Retry, or report that the key is unknown.
   Matching is exact, never fuzzy: `SF-10` does not match `SF-101`, and a key
   mentioned mid-name does not claim the task. Work that does not follow the
   Jira naming convention (a hand-made task) is not found by key; use the full
   catalogue for it. Archived projects/clients and `DONE` tasks are included and
   flagged (`archived`, `completed`/`status`): a resolved key can still be
   archived or completed, so check the flags before logging time.
   Check `coverage` before trusting `not_found`: `scope.workspace_wide` is
   false when `client_ids`/`project_ids` narrowed it, and then absence means
   only absent from that scope. Call without `keys` for the full catalogue
   (all clients, archived/active projects, `ACTIVE`/`DONE` tasks with
   `completed`); it is large and may be persisted to a file by the host.
   Never guess IDs or silently substitute a similarly named task.
5. For a user-approved completed log, call `clockify_create_time_entry` with aware
   `start`, `end > start`, `description`, explicit `billable`, and optional
   `project_id`/`task_id`. A task requires its owning project. This is a real
   write: confirm ambiguous dates, associations or billable intent first.
6. Accept success only with `verified=true`: the tool POSTs once and re-reads
   all relevant fields. On error inspect `detail.outcome`, `created_entry_id`
   and `retry_safe`. Never automatically retry; even `post_uncertain` can mean
   the entry was created. Check history or the known entry in Clockify first.

## Reusing a catalogue

A persisted full catalogue (a result the host spilled to a file) is reusable
without another call while all of these hold: same workspace; its `coverage`
was complete; and nothing has created, renamed, archived or completed projects
or tasks since it was read (for example a `clockify_jira_sync` apply, or the
user saying so). Read the file selectively (search it for the keys; do not load
all of it) rather than refetching only because the first result spilled. If a
needed key is absent from it, or any condition fails, look the keys up with
`keys` instead of re-fetching everything.

## Cost: response size is not upstream work

`keys` makes the answer small; it does not by itself make the lookup cheap.
Clients and projects are always listed (archived and not, each paginated), and
tasks are read per project, so an
unscoped lookup makes the same upstream requests as the full catalogue
(`coverage.upstream_requests` counts them). Pass `project_ids` and/or
`client_ids` (IDs from an earlier result; both must exist in the workspace, and
they intersect) when the keys are known to live there: only in-scope projects'
tasks are read, which cuts requests. Fixture measurement, 60 filler projects:
full catalogue 130 requests, unscoped lookup 130 requests with a ~25x smaller
result, two-project lookup 8 requests. There is no cache: each call reads
upstream afresh, and a live-account measurement has not been made.

## Pitfalls

History scans ALL authenticated-user entries without date filtering on every
call, including arbitrarily old straddling entries. It can be expensive for a
large account. Coverage reports list paths, filters, pages, records and strategy.
Names must resolve or the call fails; no silent name gaps. Each upstream list is
bounded at 1,000 pages of the requested 5,000 records (a requested page-size, not
a documented provider maximum) and errors rather than truncating. A scan ends on
Clockify's documented `Last-Page` header (`true` ends, `false` continues, even after
a short page). Only when a response has no `Last-Page` header is the first short
page assumed to be the last; each `coverage.reads[]` entry's `completion` says
which basis applied (`last_page_header` or `short_page_assumed_last`). An invalid
`Last-Page` value, or an empty page with `Last-Page: false`, fails as incomplete.
Whether live endpoints always send the header has not been verified. A scan
that repeats record IDs fails as `upstream_error` with `detail.complete: false` and
`detail.failure` (reason, page, records seen, duplicate count; no IDs or content).
That history is NOT covered: never build a timesheet from it, and never treat a
retry as proof beyond its own clean result. There is no automatic restart. Pagination
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
