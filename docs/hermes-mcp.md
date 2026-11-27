# Using Jake from Hermes over MCP

Jake no longer ships a Hermes plugin. A Hermes agent reaches it as an
ordinary stdio MCP server, `jake-tools-mcp`. Nothing from this repository is
imported into Hermes.

Statements marked **(unverified)** were not confirmed against the pinned
Hermes (`e96221f`, read from its installed source) or a live deployment.

## 1. What a connected Hermes gets

- Tools: every implemented operation plus `ping`, with no opt-in setting. Hermes names
  them `mcp__<server>__<tool>`, replacing characters outside `[A-Za-z0-9_]`
  with `_` (`tools/mcp_tool_schema.py`, `mcp_prefixed_tool_name`). Server
  `jake-tools` therefore gives `mcp__jake_tools__ping` and so on.
- Hermes adds utility tools when the server advertises the capability
  (`tools/mcp_tool_registration.py`, `_select_utility_schemas`):
  `mcp__jake_tools__list_resources`, `..._read_resource` (argument `uri`),
  `..._list_prompts` and `..._get_prompt` (arguments `name`, `arguments`).
- Resources: 8 in the verified build. The guides are `skill://<name>` and
  `skill://<name>/references/<path>.md`.
- Prompts: `issue_discovery`, `issue_planning`, `issue_maintainer_review`.
  Retrieving a prompt only returns text; it starts nothing.

## 2. Installing

Install the Nix package from this flake:
`jake-tools.packages.${system}.jake-tools` (also `default`). It provides
`bin/jake-tools-mcp` and `bin/jake-tools`, on its own Python with a runtime
closure taken from `uv.lock` (built with uv2nix from the locked wheels). The consumer's `flake.lock` pins
the revision.
Do not run `uv run --project` against a checkout for a deployed agent.

## 3. The `mcp_servers.jake-tools` entry

Hermes reads `mcp_servers` from `$HERMES_HOME/config.yaml`
(`tools/mcp_tool_discovery.py`, `discover_mcp_tools`). Example:

```yaml
mcp_servers:
  jake-tools:
    command: /nix/store/<hash>-python3.14-jake-tools-<version>/bin/jake-tools-mcp
    args: []            # or ["serve"]
    env:
      TRANSCRIPTION_API_BASE_URL: https://transcribe.example.internal
      TRANSCRIPTION_API_TOKEN_FILE: /run/secrets/transcription-token
      JAKE_TOOLS_TRANSCRIPTION_TIMEOUT_SECONDS: "1800"
      CLOCKIFY_API_KEY_FILE: /run/secrets/clockify-key
      JIRA_BASE_URL: https://example.atlassian.net
      JIRA_EMAIL: agent@example.com
      JIRA_API_TOKEN_FILE: /run/secrets/jira-token
```

Use the absolute pinned path as `command`. Put names, URLs and `*_FILE` paths
in the entry, not secret values.

### Environment the server receives

`tools/mcp_tool_config.py` `_build_safe_env` passes a stdio server only:
`PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`, `TERM`, `SHELL`, `TMPDIR`, `XDG_*`;
`HERMES_HOME` (the active profile's); names injected by an external secret
source; and the entry's own `env`. A secret that exists only in Hermes' own
process environment does not reach the server (checked by a probe).

`${VAR}` inside `env` values is substituted by `_interpolate_env_vars`
through `agent.secret_scope.get_secret` (verified). `get_secret` reads the
active profile's secret scope first, then, for a single-profile process,
falls through to the process environment (its docstring: the scope is a
`.env` overlay, not a blindfold). `_load_mcp_config` also calls
`load_hermes_dotenv` before interpolating. Under profile multiplexing a scope
miss does not fall back to `os.environ`. That the scope holds the profile
`.env` and the managed `/etc/hermes/.env` was taken from the docstring and the
consumer's comments, not traced to the loader **(unverified)**; check it live
before relying on `${VAR}` for a secret kept in a profile `.env`. An unset or
empty variable keeps the placeholder and, when the value is a single `${VAR}`,
the key is omitted rather than passed literally (`_build_safe_env`,
`unset_ref`; probe-confirmed).

### Jake's secret variables

Secrets are environment-only: `NAME` directly, or `NAME_FILE` naming a file
(`config.py`, `SECRET_VARIABLES`, `_secret`). The direct variable wins. The
names are `CLOCKIFY_API_KEY`, `JIRA_BASE_URL`, `JIRA_EMAIL`,
`JIRA_API_TOKEN`, `TRANSCRIPTION_API_TOKEN`, `MICROSOFT_REFRESH_TOKEN` and
`MICROSOFT_CLIENT_SECRET`.

Use one server entry (one process) per agent environment with only its own
variables. That is a lifecycle boundary, not a sandbox: values in `env` or a
profile `.env` remain readable by the agent's terminal.

## 4. Availability

There is no capability allowlist and no enable switch: every tool is listed.
Missing credentials never stop startup. The call answers `missing_credentials`
and names the variable (for `transcribe`, `TRANSCRIPTION_API_BASE_URL` and
`TRANSCRIPTION_API_TOKEN`); a Claude Code that does not expose `Artifact` gives
`upstream_error` with `reason: tool_not_exposed`.

## 5. Updating and refresh

Code and guidance ship in one package, so tools, prompts and guides change
together. A running session sees a new catalogue or new guides only after its
MCP connection restarts. In the CLI, `/reload-mcp` (`cli_info_mixin.py`, `_reload_mcp`) calls
`shutdown_mcp_servers()` for all servers, then `discover_mcp_tools()` re-reads
the config and respawns them, so a changed `command` store path is picked up
(verified). The gateway path (`_reload_gateway_mcp`) and the multiplexed
per-scope path were not read **(unverified)**. A gateway restart is the sure
route. `hermes mcp test` also exists.

To tell the packages apart:

- Configured: the `command` store path in `config.yaml`.
- Fresh process: `jake-tools-mcp doctor` from that path prints the
  environment, every setting with its source and checks.
- Serving the session: call `ping` through the session. It returns the
  package version, the server PID and the time. Compare a guide read with
  the deployed package too, because the version alone may not change.

### Runs directory

`runs_dir` defaults to `$XDG_STATE_HOME/jake-tools/runs` (`~/.local/state/...`
when unset) and is overridden by `JAKE_TOOLS_RUNS_DIR` or `--runs-dir`.
The native plugin ran in Hermes' full environment. The MCP server gets the
filtered environment above, so the path matches only if `HOME` and `XDG_*`
agree. Run `jake-tools-mcp doctor` with the server's environment and check
`runs_dir`. Delegation also needs the `claude` CLI on the server's `PATH`,
authenticated under its `HOME`.

### Profiles

Under a profile multiplexer Hermes keys connections by `(scope, name)`
(`tools/mcp_tool_scope.py`), so two profiles with the same server name hold
separate connections. A connection may be shared when route and credentials
match. Single-profile processes use the bare name. Behaviour when several
profiles use different `env` was not tested **(unverified)**.

### Managed scope

Hermes has a managed overlay (`hermes_cli/managed_scope.py`, default
`/etc/hermes/config.yaml`). `load_config` (`hermes_cli/config.py`) returns the
user config merged with the managed scope, and `_load_mcp_config` takes
`mcp_servers` from `load_config()`, so a managed `mcp_servers` entry reaches
discovery (verified by reading both; not run live). Put the entry in
`$HERMES_HOME/config.yaml` anyway until the managed route is tested.

## 6. Delegated Claude tasks across a cutover

Workers and run directories outlive the MCP connection. Do not kill live
workers. Task IDs remain usable afterwards if `runs_dir` resolves to the same
path under the new server.

## 7. Pending nixfiles changes (not part of this repository's change)

The consumer is `/workspace/Michael-F-Bryan/nixfiles` (HEAD `4b56e42`, whose
`flake.lock` pins jake-tools `5b9d694`). Nothing there has been edited. Items
1 to 3 must land in one commit: `lib.mkHermesPlugin` no longer exists after
this change, so a bumped pin with the old wiring (or the reverse) fails to
evaluate.

1. `flake.lock` and the `flake.nix` comment on the `jake-tools` input (which
   still says "native Hermes plugin (`lib.mkHermesPlugin`)"): bump the pin.
2. `hermes/image.nix`: delete `jakePlugin`; `extraPythonPackages` keeps only
   `hindsightPlugin`; delete `pluginPythonPath` and the
   `HERMES_START_PLUGIN_PYTHONPATH=` prefix in `agentStart`; update the
   comment about retiring the MCP server. Add the standalone package
   (`jake-tools.packages.<system>.jake-tools`) to the image and use its
   store path as `command`.
3. `hermes/config.nix`: `jakePluginConfig` becomes the `mcp_servers.jake-tools`
   entry (or is replaced by one). `managedConfig` for `personal` keeps
   `plugins.enabled = [ "hermes-memory-ui" "claude-subscription-directsdk" ]`
   and `infin` has no plugin list, so remove `jake-tools` from `enabled`.
   `jakePluginEnv` (`TRANSCRIPTION_API_BASE_URL`,
   `TRANSCRIPTION_API_TOKEN_FILE`, `JAKE_TOOLS_TRANSCRIPTION_TIMEOUT_SECONDS`)
   is currently `managedEnv`, i.e. `/etc/hermes/.env`, read by the plugin
   through the secret scope. Put those names into the entry's `env`. The managed overlay route is
   verified from source only (section 5).
4. Credentials. A stdio server gets only the safe environment plus the entry's
   `env`. Two routes: `*_FILE` paths (verified to be forwarded as given; the
   file must be readable inside the container) and `${VAR}` references, which
   `get_secret` resolves from the secret scope then the process environment
   (source verified; whether the profile `.env` is in that scope is
   **unverified**, check it during the nixfiles change). Clockify, Jira and
   Microsoft values currently sit in each profile's `.env`.
5. `hermes/project-env-start.py`: delete `retire_jake_tools_mcp`,
   `PLUGIN_PROBE`, `PLUGIN_PYTHONPATH`, `LOADER_ENV`, `JAKE_TOOLS` and the
   call in `main()`. Left in place, `retire_jake_tools_mcp` would remove an
   entry whose `args` contain `jake-tools-mcp`. `configure_mcp_launchers` (the
   `uv`/`uvx` launcher rewrite) and the project-env shell hook are unrelated
   and stay.
6. `hermes/README.md`: rewrite "Jake's tools (native plugin)", which describes
   `mkHermesPlugin`, `plugins.enabled`, profile `.env` credentials and the
   cutover removal of the MCP server. Keep the "Transcription service" text.
7. `hosts/studio/transcription.nix`: unchanged; it imports
   `transcription-service` and `transcription-models` from
   `jake-tools.packages.aarch64-darwin`, which are still exported.

Cutover order:

1. Build the image and exercise the MCP path in an isolated profile first.
2. Switch each agent to exactly one Jake provider (MCP, not both).
3. Plan around live Claude workers (section 6).
4. After an authorised rollout, check: tool discovery, a harmless `ping`, no
   native duplicate tools, and a guide read that matches the deployed package.

Rollback: restore the previous nixfiles commit and pin together. Run
directories are untouched either way.

## 8. Not done here

Not done, and each needs separate authority: nixfiles edits, image build,
deployment, service restarts, live-session verification, and Mac-native
transcription acceptance through the container-to-Mac path. This change
delivers source and isolated acceptance only: `tests/test_hermes_mcp.py`, run
with `--slow`, `JAKE_TOOLS_HERMES` (a Hermes launcher) and the optional
`JAKE_TOOLS_MCP_BIN`.
