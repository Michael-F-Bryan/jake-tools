# Optional: unsupported Clockify direct API workflows

Load only for explicitly requested work that the bundled MCP tools do **not**
implement: project/client cleanup, duplicate-project migration, exact-ID entry
updates, or timer mutations. This is a manual fallback reference, not an MCP tool
or an instruction to build new automation. Prefer `clockify_time_entries` and
`clockify_workspace_catalogue` for supported reads, `clockify_create_time_entry`
for one completed log, and `clockify-jira-sync` for supported reconciliation.

## Separate authorisation and credentials

The MCP tools use server-configured credentials; they do not export those
credentials to the calling agent. Direct API work needs separately authorised
credential access in the environment executing the request. Do not assume a local
secret manager, vault item, checkout, or server environment is available. **Stop
and report the blocker if authorised credential access is unavailable.** Never
extract credentials from the server, print them, put them in shell arguments,
or persist them in files, receipts or logs.

Keep the key in process memory and send `X-Api-Key`, `Accept: application/json`,
and (for mutations) `Content-Type: application/json`. Confirm the authorised API
base, user and workspace once per task and reuse exact IDs. Direct API access does
not inherit or enforce MCP workspace allowlisting: stay within the explicitly
approved workspace/user scope. A broader credential is not broader permission.

Consult the current [Clockify API documentation](https://docs.clockify.me/) for
only the needed endpoint, request schema and pagination rules. No large OpenAPI
file or credential-handling script is bundled or loaded by default. Read full
objects before updates; build the endpoint's required payload explicitly,
preserving unrelated fields rather than sending an unfiltered response object.

## Preview, apply, read back

1. Define exact scope: workspace/user IDs, explicit timezone-aware date boundaries,
   exact object IDs and deterministic selection rules. Inventory every page,
   including archived projects and completed tasks where relevant. Stop on partial
   inventory, ambiguous mappings, access gaps or concurrent changes.
2. Produce a dry-run receipt: selected IDs, current values, proposed values,
   preserved fields, conflicts and counts. Obtain approval for this precise change
   set; broad sync permission does not authorise cleanup or deletion.
3. Re-read each object before writing. Stop if its approved before-state changed.
   Apply the smallest mutation, with no fuzzy bulk edits and no blind retries.
4. Fetch each changed object again and compare the approved fields. For bulk work,
   re-run the original selection query and prove the old state no longer matches.
   Report exact IDs/names, before → after, counts, skipped conflicts and uncertainty.

A failed POST, including HTTP 500 or timeout, may already have committed. Record
any returned ID, inspect that exact object and the original selection before a
retry. Reuse an existing exact match only when identity is established. If the
outcome cannot be established, stop; do not duplicate a create. A failed update
also needs readback before retry. Never delete time entries unless the user
explicitly requests those exact deletions.

## Project/client association cleanup

Clockify “company” maps to project `clientId`; tasks inherit it from their project.
Resolve the target client and list both active and archived projects. Select by
exact criteria (for example an approved Jira-key naming pattern and empty
`clientId`), not semantic similarity. Preview IDs and client changes. For each
approved project, preserve name, colour, billable, public, archived, note and
other endpoint-required fields while changing the client. Re-list and prove no
selected project retains the old/empty client. If tasks matter, verify they belong
to those correctly associated projects. This is not an action of Jira sync.

## Jira-backed naming

Project names use the Jira parent/phase summary without an `SF-NNN` prefix. Keep
the Jira key in the project note (`Jira: SF-131`) or retain an existing Jira URL.
Task names keep the key prefix (`SF-353 Vehicle Control Logic - Preliminary
Architecture`). Prefer the read-only formatter when the installed CLI provides it:

```bash
jake-tools clockify jira-name --help
jake-tools clockify jira-name SF-353 "Vehicle Control Logic - Preliminary Architecture" --json
```

No checkout or credentials are required for this formatter. For actual supported
renames, use `clockify-jira-sync` preview/apply rather than direct API writes.

## Merging duplicate Jira-backed projects

Clockify has no project-merge endpoint. Keep the approved canonical project and
archive the duplicate only after moving its data; do not delete it.

1. Resolve both project IDs, all in-scope users, tasks and time entries. A scan of
   only the authenticated user's history cannot prove a project is empty. Stop if
   any user or page is inaccessible.
2. Prefer the unsuffixed summary-only project as canonical, but obtain approval
   for that exact ID and preserve its Jira note.
3. Copy duplicate tasks, reusing exact-name matches only when unambiguous. Preserve
   task attributes required by the request and record old → new task IDs. Re-read
   created tasks; recover uncertain creates before proceeding.
4. Preview every entry ID and mapped task, then move approved duplicate-project
   entries to the canonical project while preserving start, end (including null
   for a running timer), billable, description, tags and other required fields.
   Do not silently stop timers or move entries outside the approved scope.
5. Attach previously taskless canonical entries only when their description
   exactly matches a known task summary and the change is explicitly approved.
6. After proving no entries remain on the duplicate, archive it with the required
   full project-update payload. If any migration failed, leave it unarchived.
7. Verify the duplicate is archived and has no remaining entries, copied tasks
   exist, mappings match, and no active duplicate naming remains.

Make reruns idempotent: reuse established exact-name task identities, skip already
migrated entries and tolerate an already archived duplicate. Keep the ID map and
verification receipt (never credentials); do not infer progress from names alone.

## Recent work, hygiene and exact-ID updates

Use explicit periods: today/yesterday/this week resolved in the user's timezone,
or concrete aware timestamps. Read-only checks can flag running timers, missing
project/task, blank descriptions, suspiciously short entries and exact Jira keys.
Do not count running timers as completed work or invent their end times.

For updates, enumerate exact entry IDs and before/after fields first. Correlate
with Jira using, in order: an exact key in the description; an exact key from
another trusted task source; an already-established exact description mapping;
or explicit human instruction. If evidence is weaker, report candidates without
mutation. Entry edits, deletes and timer start/stop are **not** implemented MCP
tools. Use this separately authorised fallback only when needed and verify each
changed entry plus the complete original selection afterwards.
