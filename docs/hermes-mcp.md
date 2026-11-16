# Using Jake from Hermes over MCP

Jake no longer ships a Hermes plugin. A Hermes agent reaches it as an
ordinary stdio MCP server, `jake-tools-mcp`. Nothing from this repository is
imported into Hermes.

Statements marked **(unverified)** were not confirmed against the pinned
Hermes (`e96221f`) or a live deployment.

## 1. What a connected Hermes gets

- Tools: 18 in the verified Nix build (`ping`, the Claude delegation tools,
  Clockify, Jira sync, newsletters, transcription, Artifacts). Hermes names
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
closure taken from `uv.lock`. The consumer's `flake.lock` pins the revision.
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

`${VAR}` inside `env` values is substituted by `_interpolate_env_vars` from
Hermes' secret scope (`agent.secret_scope.get_secret`). `_load_mcp_config`
loads the profile `.env` first. Whether a given variable then resolves from
the process environment, the profile `.env` or both is decided by
`get_secret` and was **not** read to the end **(unverified)**. If the variable
is unset or empty, the placeholder is kept and, when the value is a single
`${VAR}`, the key is omitted rather than passed literally (probe-confirmed;
`_build_safe_env`, `unset_ref`).

### Jake's secret variables

Secrets are environment-only: `NAME` directly, or `NAME_FILE` naming a file
(`config.py`, `SECRET_VARIABLES`, `_secret`). The direct variable wins. The
names are `CLOCKIFY_API_KEY`, `JIRA_BASE_URL`, `JIRA_EMAIL`,
`JIRA_API_TOKEN`, `TRANSCRIPTION_API_TOKEN`, `MICROSOFT_REFRESH_TOKEN` and
`MICROSOFT_CLIENT_SECRET`.

### Per-environment credentials

Run one server entry, hence one process, per agent environment, and give each
only its own variables. A separate process is a dependency and lifecycle
boundary, not a sandbox: values in `env` or a profile `.env` remain readable
by the agent's own terminal.

## 4. Availability

There is no capability allowlist. `transcribe` is listed only when a
transcription endpoint is configured; Artifact tools only when
`artifacts.enabled`. Missing credentials never stop startup. The call answers
`missing_credentials` and names the variable.

## 5. Updating and refresh

Code and guidance ship in one package, so tools, prompts and guides change
together. A running session sees a new catalogue or new guides only after its
MCP connection restarts. Hermes has a `/reload-mcp` command ("Reload MCP
servers from config", `hermes_cli/commands.py`) and `hermes mcp test`; whether
`/reload-mcp` respawns an unchanged entry that now points at a new store path
was not checked **(unverified)**. A gateway restart is the sure route.

`load_config()` runs once at server start (`mcp/__main__.py`), so
environment variables and the contents of `*_FILE` files are read at spawn.
Changing a credential needs a reconnect.

To tell the packages apart:

- Configured: the `command` store path in `config.yaml`.
- Fresh process: `jake-tools-mcp doctor` from that path prints the
  environment, every setting with its source, config files and checks.
- Serving the session: call `ping` through the session. It returns the
  package version, the server PID and the time. Compare a guide read with
  the deployed package too, because the version alone may not change.

### Runs directory

`runs_dir` defaults to `$XDG_STATE_HOME/jake-tools/runs` (`~/.local/state/...`
when unset) and is overridden by `JAKE_TOOLS_RUNS_DIR` or `claude.runs_dir`.
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
`/etc/hermes/config.yaml`, deep-merged over user config by
`apply_managed_overlay`). Whether `mcp_servers` set there reaches
`_load_mcp_config` was not confirmed **(unverified)**. Put the entry in
`$HERMES_HOME/config.yaml` unless that is tested.

## 6. Delegated Claude tasks across a cutover

Workers and run directories outlive the MCP connection. Do not kill live
workers. Task IDs remain usable afterwards if `runs_dir` resolves to the same
path under the new server.

## 7. Pending nixfiles changes (not part of this repository's change)

The consumer is `/workspace/Michael-F-Bryan/nixfiles` (HEAD `4b56e42`, which
pins jake-tools `5b9d694`). Nothing there has been edited.

1. `flake.lock`: bump the jake-tools pin. `lib.mkHermesPlugin` no longer
   exists, so the bump and the wiring below must land in one commit.
2. `hermes/image.nix`: drop `jakePlugin`/`mkHermesPlugin` from
   `extraPythonPackages`; add the standalone package to the runtime; drop
   `HERMES_START_PLUGIN_PYTHONPATH`.
3. `hermes/config.nix`: remove `jake-tools` from `plugins.enabled` (keep
   `hermes-memory-ui` and `claude-subscription-directsdk`); turn
   `jakePluginEnv` into the `mcp_servers.jake-tools` `env`.
4. Credentials: Clockify, Jira and Microsoft values currently live in each
   profile's `.env`, which the plugin read through Hermes' secret scope. A
   stdio server does not get the profile `.env` wholesale. The mechanism
   verified above is the entry's `env`, either `*_FILE` paths or `${VAR}`
   references. Which source `${VAR}` reads for a profile `.env` value needs
   verification before relying on it **(unverified)**.
5. `hermes/project-env-start.py`: delete `retire_jake_tools_mcp`,
   `PLUGIN_PROBE` and `PLUGIN_PYTHONPATH` handling. Left in place they would
   remove the entry this design needs.
6. `hermes/README.md`: rewrite "Jake's tools (native plugin)".
7. `hosts/studio/transcription.nix`: unchanged;
   `packages.aarch64-darwin.transcription-service` and `transcription-models`
   are still exported.

Cutover order:

1. Build the image and exercise the MCP path in an isolated profile first.
2. Switch each agent to exactly one Jake provider (MCP, not both).
3. Plan around live Claude workers (section 6).
4. After an authorised rollout, check: tool discovery, a harmless `ping`, no
   native duplicate tools, and a guide read that matches the deployed package.

Rollback: restore the previous nixfiles commit and the jake-tools pin
together. This repository no longer contains the plugin, so an old pin cannot
be mixed with new source. Run directories are untouched either way.

## 8. Not done here

No nixfiles edits, image build, deployment, service restart or live-session
verification, and no Mac-native transcription acceptance through the
container-to-Mac path. Those need separate authority. This change delivers
source and isolated acceptance only: `tests/test_hermes_mcp.py`, the `--slow`
option, `JAKE_TOOLS_HERMES` and the optional `JAKE_TOOLS_MCP_BIN`.
