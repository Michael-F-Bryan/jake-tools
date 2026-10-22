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
by a hub or another general-purpose skill. Install these focused, tool-owned
skills for supported MCP workflows. The useful unsupported Clockify procedures
from the former `clockify-api/references/workflows.md` are retained as a supporting
reference under `clockify-time`, with host-specific checkout/vault assumptions
removed. It is progressive disclosure within an existing skill, **not** an
optional skill tier. No OpenAPI dump or credential-handling script is needed for
this guidance; consult current endpoint documentation only for the requested
unsupported operation.

Old CLI-only routing is not a reason to bypass an available MCP tool. Other CLI
commands remain discoverable through `jake-tools --help` and subcommand help;
this migration does not create new tools or duplicate unrelated specialist
skills. Direct API fallback requires separate authorised credential access; MCP
server credentials are not available to callers merely because a tool works.

## Delivery and updates

Each `src/jake_tools/skills/<name>/` contains a `SKILL.md` and, when needed,
`references/`. The package build includes these data files.

```bash
jake-tools-mcp install-skills --into /absolute/path/to/target/skills
# Equivalent: python -m jake_tools.mcp install-skills --into ...
```

Choose the target explicitly, or omit `--into` only when `HERMES_HOME` already
identifies the intended profile. Installation copies the **whole** skill
directory, including nested references. `--link` links the whole directory.
Existing destinations are refused/skipped unless `--force` is supplied; review
local edits before replacing them. Installing does not remove retired broad
skills: remove stale `jake-tools`/`clockify-api` copies from the intended shared or
profile skills directory separately, with authorisation. Do not modify another
profile or runtime configuration as part of a documentation migration.

The running MCP server lists each main skill as `skill://<name>` and each
Markdown supporting reference, recursively, as
`skill://<name>/references/<path>.md`. Resource listing provides metadata, not
reference bodies. `resources/read` of a main skill does not inline its references;
read the linked reference URI only when its task needs it. These are fixed
packaged resources, not an arbitrary filesystem reader. The installer already
handles full trees; only resource delivery needed extending for references.

When a local installed copy and a running server differ, inspect live schemas
and read that server's skill resources. Updating source or installing skills
does not update an already-running server. Coordinate deployment separately;
this migration does not restart services or change their configuration.
