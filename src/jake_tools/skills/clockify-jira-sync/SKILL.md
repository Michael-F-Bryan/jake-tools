---
name: clockify-jira-sync
description: Reconcile Jira work items into Clockify projects and tasks with the jake-tools `clockify_jira_sync` tool. Use when asked to sync, tidy, or check Clockify against Jira, or when Clockify tasks are missing, misnamed, closed too early, or still open after the Jira issue is done.
---

# Jira → Clockify sync

One tool, `clockify_jira_sync`, from the jake-tools MCP server (an MCP client may prefix the name, as
Hermes does: `mcp__<server>__clockify_jira_sync`). This skill uses the bare
name. It needs Clockify and Jira credentials in the MCP server's environment,
so it only works where Jira is configured.

It creates projects and tasks, renames them, reactivates tasks, marks tasks
done, and relocates a task (with its time entries) to the project Jira places
it under. It never deletes anything, and it has no other Clockify or Jira
write surface.

Jira `Project / Phase` issues become Clockify projects. Ordinary issues and
subtasks become sibling Clockify tasks under their ancestor `Project / Phase`,
regardless of intermediate parents. Ancestors are loaded for placement, not
selected for task creation. Missing ancestors, chains with no phase, and cycles
are conflicts; the sync never guesses a project. A task found under a
different project than its phase is relocated (below).

## Automatic discovery versus explicit issues

| `issues` | Selects |
| --- | --- |
| omitted | The authenticated user's assigned work in To Do, In Progress, In Review or Blocked (scope `assigned-active`) |
| non-empty | Exactly those issues, whatever their assignee or Jira status (scope `issues`) |

Backlog, TRIAGE, and Investigate are excluded from automatic creation. Existing
tasks deferred to the backlog stay intact and can still be renamed. Only Jira
status category Done (including Done and Won’t Do) completes a task, and nothing
reopens one that Jira says is finished.

Explicit selection overrides those discovery filters and nothing else. Each
requested issue still needs a resolvable `Project / Phase` ancestor and passes
the duplicate checks, so a problem is a conflict naming that
key (an ambiguous phase project also blocks each requested issue under it,
with a reason of its own). A key Jira cannot return fails the call as
`upstream_error` naming it. Ancestors fetched for placement are not selected:
their phase project is checked, but their own Clockify tasks are never created,
renamed or completed. What an explicit issue gets:

- **No Clockify task yet.** Backlog, TRIAGE, Investigate, To Do or underway: one
  `CREATE_TASK`, active. Done or Won’t Do: `CREATE_TASK` **then**
  `MARK_TASK_DONE` for the same key, because Clockify creates tasks active.
  The second action has `task_id: null` in the preview (the task does not exist
  yet) and the digest covers it, so the plan reviewed is the completed final state.
- **A task already exists.** No duplicate: the normal name, reactivate and
  complete reconciliation applies, and a task already in the right state has no
  actions at all. If it is under the wrong project, it is relocated instead.

## Relocation (`RELOCATE_TASK`)

Jira is authoritative for placement. A selected issue whose one Clockify task
sits under another project gets `RELOCATE_TASK`, ordered after any
`CREATE_PROJECT` for its phase. Clockify's API cannot move a task between
projects (the task update has no `projectId`), so apply runs a recoverable
replacement and never deletes anything:

1. It creates a staging task `[relocating KEY from SOURCE_ID #BASELINE] KEY
   summary` in the destination. `BASELINE` is a hash of the source task's
   fields and of every entry minus its placement, recorded before any entry
   moves. It copies assignees, user groups, estimate, budget,
   billable and task hourly/cost rate, each set from now on and never
   retroactively, then re-reads it and compares every task field.
2. It re-points each time entry at it. The full update echoes start, end,
   description, billable, tags, custom fields and type. The entry is re-read
   before the write (its fingerprint must match the preview) and compared
   whole afterwards.
3. It re-scans the source for entries and retires it as
   `[relocated KEY to NEW_ID] …`, DONE.
4. It promotes the staging task to `KEY summary` with Jira's status, last.

The action's `relocation` object gives `source_project_id`, `source_task_id`,
`desired_status`, `time_entry_ids`, and, when resuming an interrupted
relocation, `staged_task_id`, `source_retired` and `moved_entry_ids`. The
digest binds a fingerprint of each whole entry and task record, so any edit
after the preview is `plan_stale`.

A fresh preview resumes only an exact relation: one staging task in the
destination naming the key's source task, or naming a source already retired
into it. The records must also still match the staging task's baseline, so an
entry whose move changed a field, even one whose write answered with an error,
is never silently accepted on retry. An interrupted relocation is resumed (or
reported) by the default scope even after its issue leaves discovery, and
Jira's Done status is applied on promotion. These are conflicts, and they
block the whole apply:

- records that no longer match the baseline. The conflict names the entries
  already on the staging task. After checking them in Clockify, a person may
  accept their current values by removing ` #BASELINE` from the staging task's
  name; the next apply records a new baseline before any other write;

- any other duplicate shape: two tasks claiming the key, two staging tasks, a
  staging task elsewhere or naming another source, or a staging task next to a
  task already in place;
- a running timer, a locked (approved) entry, or a HOLIDAY/TIME_OFF entry on
  the source.

Limits to state plainly:

- Entry `hourlyRate`/`costRate` are read-only: no time-entry endpoint writes
  them, and whether a move recomputes them is undocumented. A rate or any other
  field that reads back changed fails the apply, naming the entry and its
  original values. Nothing is rolled back, and the next preview reports it as
  a baseline conflict until a person decides.
- The entry API exposes no invoiced or approval-request state, so neither can
  be verified.
- Entries are found per member (`/users?status=ALL`, then each member's
  entries filtered by project and task). Members Clockify's default
  account-status filter omits, and concurrent writes during the offset-paginated
  scans, are not covered. The re-scan before retiring is evidence, not an
  atomic guarantee. Assume nobody else edits those entries during an apply.

`inventory.active_issues` counts planned or underway issues: all discovered
ones, or the requested ones whose Jira status category is not Done. A requested
Done issue is still planned without being counted, so `active_issues: 0` does
not mean nothing was selected; read `actions` and `conflicts`.

## Discovery, configuration and scope

Find `clockify_jira_sync` through the host's tool search/list, then inspect its
live schema: `issues?: list[str]`, `apply: bool = false`,
`plan_digest?: str`. Read this guide from the running server as the MCP resource
`skill://clockify-jira-sync`, through your client's MCP resource reader
(in Hermes: the server's `read_resource` utility tool, `mcp__<server>__read_resource`, with `uri`), for its version. Do not pass CLI flags or assume a local checkout is needed. Credentials
are resolved by jake-tools (`CLOCKIFY_API_KEY`, `JIRA_BASE_URL`, `JIRA_EMAIL`,
`JIRA_API_TOKEN`, including configured secret-file variants); do not retrieve
local vault credentials to make the call. Missing configuration is a blocker.

Unlike the history/catalogue/create tools, this tool has no `workspace_id`
argument and does not use their workspace allowlist. It resolves the Clockify
user's active workspace (and fails if none), then the exact configured client
(default `Sunfish Robotics`) and Jira project (default `SF`). Confirm that server
scope matches the request; never promise an arbitrary workspace selection.
The configured client selects the owner of **new** projects; it is not an
inventory filter. The sync inventories all active projects in that workspace
and their active/done tasks. Default selection also checks Jira keys already
referenced by those projects/tasks for drift and completion; it is not limited
to creating the authenticated user's currently assigned work.

Naming is deterministic: project = phase summary only, task = Jira key plus
summary, project note = `Jira: KEY`. The installed CLI's read-only
`jake-tools clockify jira-name --help` describes the formatter; it is not an MCP
tool. Sync does not repair client associations, merge duplicates, migrate time
entries, or edit timers. Those unsupported workflows are documented separately
in `clockify-time`'s `references/direct-api-workflows.md`, with separate approval
and credential requirements.

## The loop: preview, read, apply with the digest

1. **Preview.** Call with no arguments to cover planned or underway Jira work
   (To Do, In Progress, Blocked, In Review) assigned to the authenticated user,
   or `issues: ["SF-304", ...]` for exact keys
   regardless of assignee or status (including finished work you did not own,
   for historical timesheets). Preview is read-only. `apply` defaults to false.
2. **Read the report** before doing anything else (fields below). If
   `actions` is empty and `conflicts` is empty, there is nothing to do; say
   so and stop.
3. **Apply** by calling again with the *same* `issues` (or none), plus
   `apply: true` and `plan_digest` copied from the preview. Only apply when
   the person asked for changes, or has seen the planned actions and agreed.
4. **Check the apply report**: every action should have `applied: true` and
   `verified: true`, and `failure` should be `null`.

Without a `plan_digest`, apply refuses with `invalid_argument` and writes
nothing.

## Reading a report

```json
{
  "mode": "preview",
  "scope": {"kind": "assigned-active", "jira_project": "SF", "issue_keys": []},
  "inventory": {"active_issues": 1, "jira_issues": 3, "projects": 1, "tasks": 2},
  "actions": [
    {"kind": "REACTIVATE_TASK", "jira_key": "SF-427",
     "current_name": "SF-427 Evaluate PX4 external control methods",
     "desired_name": "SF-427 Evaluate PX4 external control methods",
     "project_key": "SF-131", "project_id": "p-131", "task_id": "t-427",
     "jira_status": "", "message": "", "applied": false, "verified": false},
    {"kind": "MARK_TASK_DONE", "jira_key": "SF-304", "jira_status": "Done",
     "task_id": "t-304", "applied": false, "verified": false}
  ],
  "conflicts": [],
  "plan_digest": "17842a21…",
  "applied": false,
  "failure": null
}
```

- `actions[].kind` is one of `CREATE_PROJECT`, `RENAME_PROJECT`,
  `RELOCATE_TASK`, `CREATE_TASK`, `REACTIVATE_TASK`, `RENAME_TASK`,
  `MARK_TASK_DONE`. `relocation` is null except on `RELOCATE_TASK`. Renames
  show `current_name` → `desired_name`; creates have an empty
  `current_name`. A `CREATE_TASK` and a `MARK_TASK_DONE` for the same key
  (with a null `task_id`) are one finished issue's task being created, then
  completed later in the plan; actions are ordered by kind, so with several keys
  they need not be adjacent. After an
  apply, a created record's `task_id` (or `project_id`) is the verified one.
- `conflicts` lists what the plan refuses to touch, each with a `jira_key` and
  a `message` (e.g. `"multiple active Clockify tasks reference SF-304"`).
  **Any conflict blocks the whole apply.**
- `applied` at the top level means "an apply was attempted", not that
  everything was written. Trust the per-action flags and `failure`.
- An empty `jira_status` or `message` just means there's nothing extra to say.

When summarising for a person, list the actions by Jira key and kind, call
out renames with both names, and show conflicts on their own.

## Failures

Errors come back with `isError: true` and `{code, message, detail}`:

| `code` | Meaning | What to do |
| --- | --- | --- |
| `plan_stale` | Jira or Clockify changed since the preview (`detail.expected`/`actual` digests). Nothing was written. | Preview again, show the new plan, then apply with the new digest. Don't loop on apply. |
| `invalid_argument` with `detail.conflicts` | The plan has conflicts. Nothing was written. | Report the conflicting keys and the preview's conflict messages. Fixing them is a human edit in Clockify or Jira. |
| `invalid_argument` with `detail.argument` | A bad argument: a missing `plan_digest`, or a malformed key in `issues`. | Fix the call. |
| `missing_credentials` | `detail.variables` names the unset environment variables. | Tell the person which variables are missing. This instance cannot sync until they are configured. Don't retry. |
| `upstream_error` | Clockify or Jira refused or was unreachable (`detail.integration`). The message gives the operation and the HTTP status or error class. | Report it. Retry once later at most. A 401/403 is a credentials problem, not a transient one. |

**Partial failure is not an error result.** An apply that fails partway
returns a normal report with `failure` set:

```json
"failure": {"jira_key": "SF-304", "kind": "MARK_TASK_DONE",
            "message": "Clockify error: PUT /workspaces/ws-1/projects/p-131/tasks/t-304 returned HTTP 500"}
```

Actions before the failure have `applied: true, verified: true`; they were
written and read back. The action named in `failure`, and everything after
it, show `applied: false`. The failing action **may still have been partly
written** (the write landed, then the read-back failed). So after a partial
failure:

1. Tell the person what was applied and what failed.
2. Preview again. The new plan shows what is still outstanding, including
   whether the failed action actually landed. If a finished issue's task was
   created and its completion failed, the created `CREATE_TASK` shows
   `applied: true` with its `task_id`, `failure.kind` is `MARK_TASK_DONE`, and the
   next preview plans only `MARK_TASK_DONE` on that task. If the creation's own
   read-back failed, `failure.message` names the Clockify task ID to inspect.
   The fresh plan is evidence, not permission to repeat an uncertain POST: inspect
   the exact changed records (and known IDs) first. If a create may have landed
   but identity cannot be established, stop and report uncertainty; do not create
   a duplicate. A transport failure during apply can also leave writes behind.
   A failed `RELOCATE_TASK` also sets `failure.recovery`: what it verified
   (staging task ID, entries already moved, whether the source is retired)
   and whether its last write is uncertain. The next preview reads those
   records and resumes through the staging marker; it never creates a second
   staging task for the same source.
3. Apply a new plan only after resolving uncertainty, if the failure looks
   transient and the person wants it finished. Never blindly retry an apply.

## Don't

- Don't apply without first showing or summarising the preview, unless the
  person explicitly asked for the sync to be applied.
- Don't reuse a digest across different `issues` scopes, or after a
  `plan_stale`.
- Don't try to resolve conflicts by applying a narrower `issues` list that
  skips them unless the person asks for that.
