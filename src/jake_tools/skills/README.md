# Packaged skills: knowledge ownership

Tool-specific operational guidance belongs beside its implementation in this
repository, not in a second broad skill in the shared Michael skills repository.
Code and live tool schemas are authoritative for supported behaviour; these
skills explain how to use it safely. Enforced invariants belong in code and
judgement belongs here.

| Owner | Guidance |
| --- | --- |
| `claude-delegation` | Discover, start, observe, steer, answer, cancel and resume bounded Claude tasks; grants, limits and result verification. |
| `clockify-time` | Read authenticated-user history/catalogue and create one verified completed entry; uncertain-write recovery. |
| `clockify-jira-sync` | Jira placement/naming, selection scope, read-only preview, digest-bound apply, conflicts and partial failure. |
| `clockify-time/references/direct-api-workflows.md` | Separately authorised manual fallback for unsupported cleanup, duplicate migration and exact-ID entry edits. Not implemented MCP tools. |

## Migration from broad shared skills

The shared `jake-tools` and `clockify-api` skills are retired rather than replaced
by a hub or another general-purpose skill. Consume these focused, tool-owned
skills as MCP resources for supported workflows. The useful unsupported Clockify
procedures from the former `clockify-api/references/workflows.md` are retained as
a supporting reference under `clockify-time`, with host-specific checkout/vault
assumptions removed. It is progressive disclosure within an existing skill,
**not** an optional skill tier. No OpenAPI dump or credential-handling script is
needed for this guidance; consult current endpoint documentation only for the
requested unsupported operation.

Old CLI-only routing is not a reason to bypass an available MCP tool. Other CLI
commands remain discoverable through `jake-tools --help` and subcommand help;
this migration does not create new tools or duplicate unrelated specialist
skills. Direct API fallback requires separate authorised credential access; MCP
server credentials are not available to callers merely because a tool works.

Retired shared/profile copies can still shadow current guidance. Remove stale
`jake-tools`/`clockify-api` copies only in the intended skills directory, with
authorisation. Do not modify another profile or runtime configuration as part
of a documentation migration.

## Resource-only delivery

Each `src/jake_tools/skills/<name>/` contains a `SKILL.md` and, when needed,
`references/`. Package builds include these data files; no profile installer or
installer replacement is provided. Consume them through the connected MCP server:

1. Use `resources/list` to discover `skill://<name>`. Each main resource's
   description is the skill's YAML frontmatter `description`, not generic
   packaging text. Resource listing provides metadata, not skill/reference bodies.
2. Use `resources/read` with the chosen URI to get the complete `SKILL.md` matching
   the running server. Inspect live tool schemas before calls; do not assume CLI
   flags are MCP arguments.
3. Read a linked supporting reference only when the task needs it. Nested
   Markdown files are independently listed and readable at
   `skill://<name>/references/<path>.md`. A main skill read does not inline them;
   no large OpenAPI default load is required.

For example, read `skill://clockify-time` for routine history/logging and
`skill://clockify-time/references/direct-api-workflows.md` only for explicitly
requested unsupported cleanup. These are fixed packaged resources, not an
arbitrary filesystem reader. Descriptions are parsed with `yaml.safe_load`,
including quoted and multiline YAML strings; a packaged skill without a valid
non-empty description fails registration rather than advertising misleading
metadata.

If a cached/local copy and a running server differ, use live schemas and that
server's resources. Updating source does not update an already-running server.
Coordinate deployment separately; this migration does not restart services or
change their configuration.
