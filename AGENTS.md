# Jake's Tools

Internal Python CLI tools for Jake. Stack: Python 3.14, `uv`, Click, Pydantic,
and the Claude Agent SDK (`claude-agent-sdk`) for LLM calls. Package layout
lives under `src/jake_tools/`. Human onboarding is in [README.md](README.md).

When in doubt, run `jake-tools <command> --help` for current flags.

## Project layout

```text
src/jake_tools/
  cli/            # Click commands (keep thin)
  claude.py       # Claude Agent SDK wrapper — the only LLM seam
  claude_runs/    # delegated tasks: run dirs, worker, events, inbox, service
  operations/     # agent-facing tool operations shared by the MCP adapter and the CLI
  mcp/            # the stdio MCP adapter: the one agent interface, for every client including Hermes
  polishing/      # polish_transcript: guarded clean-verbatim drafts via Claude
  artifacts/      # experimental: publish an HTML file as a private Claude Artifact, read back
  prompting.py    # typed Jinja prompts bound to a response model
  newsletters.py  # SharePoint Graph client (list paging, attachments, request/outcome classification)
  newsletter_entries.py # verified list/get/create/update workflows over the client; partial-outcome effects
  microsoft_auth.py # refresh-aware delegated tokens per resource audience; identity check
  sharepoint_files.py # Graph drive-item client (shares, channel folder, children, 302 content, upload sessions)
  file_transfers.py # verified resolve/list/download/upload/replace workflows; coverage, versions, outcomes
  transcription.py # HTTP client + transcript contract for transcription-service
tests/            # mirrors packages above
transcription-service/  # native Swift HTTP service (FluidAudio); see its README
```

`nix develop` (flake.nix, loaded by direnv via `.envrc`) provides `uv`, a Nix
Python 3.14, `swift-format`, `sox` and `nixfmt`. The Swift service builds with
nixpkgs' Swift and Apple SDK (`nix/transcription-service.nix`), never Xcode or
the Command Line Tools.

## Developing

Install once:

```bash
uv sync
uv run pre-commit install
```

Before finishing work, all of these must pass:

```bash
uv run pre-commit run --all-files
```

That runs `ruff check --fix`, `ruff format`, `pyright`, `pytest -q`, and a
`uv lock` consistency check. For a quicker loop on Python changes:

```bash
uv run ruff check --fix src tests
uv run ruff format src tests
uv run pyright
uv run pytest -q
```

`pytest -q` (and the `pre-commit`/CI runs above) skip two kinds of test by
default, via two independent markers:

- `@pytest.mark.slow` (`pytest-skip-slow`) — anything expensive: a real
  model call, a real subprocess. Run these with `uv run pytest --slow`, or
  target one with `uv run pytest --slow -k <selector>`.
- `@pytest.mark.live` — anything that hits a real external service; excluded
  via `addopts = -m "not live"` in `pyproject.toml`, independent of `--slow`.

### Testing judgement-bearing LLM behaviour

A fake model response proves only deterministic mechanics around the LLM seam.
It may test prompt rendering, schema injection and parsing, retry limits, cache
behaviour, telemetry, validation, error mapping, or CLI delegation. It cannot
prove that a prompt gives the model enough evidence, that the model follows the
instructions, or that generated output meets a semantic quality bar.

- If an acceptance criterion depends on language understanding, judgement,
  interpretation, attribution, summarisation, or a prompt working as intended,
  exercise the normal `ClaudeAgent` path through the real Claude Agent SDK in
  an opt-in `@pytest.mark.slow` test.
- Never author a compliant fake response and cite the passing test as evidence
  that the model can produce that response. That proves only what the test
  author supplied.
- Default to executing the actual production code at the lowest practical
  acceptance boundary. Do not monkeypatch the subject under test, replace its
  orchestration, or pre-author the intelligent result merely to make a test
  pass. A fake is acceptable only at a true external boundary, and the test
  must still execute the production behaviour it claims to verify.
- Keep deterministic validators and seam tests fast and hermetic, but pair them
  with representative real-model tests for the judgement-bearing behaviour
  they constrain.
- Assert stable semantic invariants and observable outcomes in real-model tests,
  not exact prose. Examples include preserved meaning, correct modality,
  evidence-grounded attribution, source coverage, and refusal to invent.
- If credentials, quota, or the provider are unavailable, report the real-model
  test as blocked or unverified. Do not replace it with a fake, weaken the
  assertion, or claim the prompt is covered.

### Code conventions

- Keep CLI commands thin; put orchestration and domain logic in package modules.
- Match surrounding code. Prefer typed Pydantic models over ad hoc dicts.
- Ruff rules: `E`, `F`, `I`, `UP`, `B`, `SIM`, `C4` (see `pyproject.toml`).
- Pyright must pass (`pyrightconfig.json`).
- All LLM calls go through `ClaudeAgent` in `claude.py`. Nothing else imports
  `claude_agent_sdk` directly.
- `AgentSpec.tools` defaults to an empty tuple, which is genuinely tool-less.
  Never pass `tools=None` to `ClaudeAgentOptions`: the SDK then omits `--tools`
  and the agent inherits Claude Code's full default toolset.
- `AgentSpec.tools` is exposure (bare built-in names); `allowed_tools` is
  approval (permission rules such as `Bash(git status)`), defaulting to the
  exposed tools. Never put a rule in `tools`, and never add a hard-coded
  list of tool names: the CLI is the authority and silently drops names it
  does not provide, so check the session's `init` tool list
  (`MessageView.tools`). Evidence, and the experimental artifact route:
  `docs/claude-tools.md`.
- `ctx.obj` is never set or read anywhere in this codebase. Every CLI
  dependency is built by the command that needs it: a decorator (see
  `cli/clockify.py`) stacks the
  relevant `click.option`s, pops their parsed values, and injects a typed
  Pydantic options model with dependency-constructor methods — e.g.
  `ClockifyOptions.inventory_client()` — via `ctx.invoke`. When a dependency takes no CLI
  flags at all (e.g. `NewsletterClient`), the handler just constructs it
  directly at the top of the function. `clockify`'s `--api-key`/
  `--api-base-url` are ordinary per-subcommand flags via a `clockify_options`
  decorator, the same shape as every other options decorator — there is no
  group-level state and no exception to this rule.


## MCP server (`jake_tools.mcp`)

`python -m jake_tools.mcp` (also the `jake-tools-mcp` console script) is the
portable stdio MCP server and the only agent-facing interface; Hermes
connects through its built-in MCP client (`docs/hermes-mcp.md`), with no Hermes
plugin and no Hermes import anywhere in this package. Sub-commands: `serve`
(default), `doctor`, and the hidden `worker RUN_DIR`. It is a separate entrypoint from `jake-tools` on purpose:

- No entrypoint (this one or `jake-tools`) imports `dotenv` or loads a `.env`,
  and nothing reads a `config.toml`. The process environment the client or
  shell passes is the only environment. `python-dotenv` is not a dependency.
  Tests pin this (`tests/test_config_contract.py`, run from a hostile cwd).
- Stdout is the transport. Log to stderr only, never print.
- Configuration is explicit: `config.load_config()` resolves flag > env >
  default, plus XDG state/cache paths. There is no config file. Every
  non-secret setting has an environment variable (the `_Spec` in `config.py`);
  `serve`/`doctor` expose the operator flags listed in `mcp/options.py`
  (`FLAGS`, one explicit row each, a test ties each to its variable). A new
  setting needs an env name, a `_Spec`, a README table row and, if
  operator-facing, a flag row. Secrets are environment-only (`NAME` or
  `NAME_FILE`), have no flag, and are redacted. Every value is a `Resolved`
  that remembers its source; `doctor` prints `display_value()`. Each
  sub-command loads its own `Config`; there is still no `ctx.obj`.
- Missing credentials never stop startup. Clients are built per call inside
  the tool that needs them and raise `ToolError("missing_credentials", ...)`
  naming the variable, never the value.
- `operations/<tool>.py` holds one thin handler each as an `Operation`
  (name, description, skill, `build(config)`, optional `available`), listed in
  `operations/catalogue.py`; there is no capability allowlist or grouping, so
  adding an operation there exposes it. The adapter never duplicates an operation's logic,
  schema or description. `mcp/server.py` registers them (plus the MCP-only
  `ping`) through `structured_tool`, which sends the handler's
  Pydantic result as structured content and turns a raised
  `operations.errors.ToolError` into an `isError` result with the stable
  `{code, message, detail?}` payload. Codes are the `ToolErrorCode` literal;
  do not invent new ones without adding them there. No `outputSchema` is
  advertised, by design, so error payloads never fail schema validation.
- Handlers are `async`. Blocking `requests` clients run under
  `anyio.to_thread.run_sync`. Timeouts and cancellation use anyio cancel
  scopes (`anyio.fail_after`, `move_on_after`), never `asyncio.timeout` or
  `wait_for`: the Claude SDK's child-process cleanup only runs under anyio
  cancellation.
- The server holds no state that matters. Delegated tasks live in run
  directories (`claude_runs/models.py` documents the files and lock order);
  the worker is its own process group and outlives the server. No queue, no
  database. The worker holds one streaming conversation
  (`ClaudeAgent.converse`). It writes normalised `events.jsonl`, which
  `claude_transcript` pages by `<task_id>:<seq>:<offset>` cursor.
  `claude_poll` never pages: it returns a compact snapshot
  (`claude_runs/observe.py`) and an opaque, non-consuming observation token
  (a position in the same log) and reports only meaningful changes since
  it; questions and ends always wake it, routine activity never does
  (unless `until=progress`). It pumps `messages/` (steering) into the
  conversation and serves the opt-in `ask_parent` local tool from `inputs/`.
  `claude_runs.service.TaskService` is the one service every `claude_*` tool
  calls. `claude_runs.tasks_protocol` projects it onto MCP Tasks 2026-07-28,
  and `mcp/tasks.py` serves that projection natively through `mcp`'s
  `Extension` API (`mcp` 2.3 has no Tasks implementation of its own):
  `server/discover` advertises it, `tasks/get|update|cancel` are bound, and
  a `tools/call` interceptor turns a successful `claude_start` into a
  `CreateTaskResult`. That happens only when the request is on a 2026-07-28
  connection and its own `_meta` declares the extension; legacy
  (`initialize`) requests, Hermes included, always get the ordinary tool
  result, and undeclared `tasks/*` is `-32021`. Worker failures are
  `completed` tasks with an `isError` result; protocol `failed` is never
  produced. Steering, bounded polls and transcripts stay ordinary tools.
  No `notifications/tasks` subscriptions.
- Real CLI semantics this relies on (CLI 2.1.293, re-checked on 2.1.309;
  see the module docs):
  - A user message sent mid-turn joins the turn after the running tool
    call. `--replay-user-messages` echoes it with its `uuid`.
  - `num_turns`/`--max-turns` restart per prompt, so the worker enforces
    `max_turns` itself.
  - Token usage is per prompt, but `total_cost_usd` is a running session
    total carried across `--resume`.
  - `--tools` drops unknown or unavailable names without an error, and
    reports `Agent` as `Task`. The worker fails a session that lacks a
    requested tool as `tools_unavailable`.
  - (2.1.309) Resuming a session whose last run left a background task,
    the CLI answers that task's end with a zero-turn `result` before it
    takes the new prompt. The worker tags the brief with its UUID and
    ignores a zero-turn result that comes before the brief's echo.
  - (2.1.309) A refusal is reported as it happens (`system`
    `permission_denied`, with `decision_reason_type`); some calls are
    refused by the CLI's own guard with only `Blocked: …` error text.
- A `sdk_error` carries a structured `failure` (`claude_runs/failures.py`):
  `kind` (`usage_limit`, `authentication_or_access`, `sdk_transport`,
  `unknown`), the original `evidence` (result `subtype`/`is_error`/text or
  exception type), and `advice` marked `basis="inferred"`. It reads only the
  runtime's own short report, which must open with the statement and not
  also state the other kind, never the result `subtype` (the reported
  failures arrived as `subtype="success"` with `is_error`); unrecognised or
  conflicting text stays `unknown`. `sdk_transport` is only the SDK's exact
  `CLIConnectionError`/`CLIJSONDecodeError`/`MessageParseError`/`ProcessError`
  (matched by package, since only `claude.py` imports the SDK); a missing
  executable or a `ResultError` is not transport. `resets_at` is set only
  for a full date with year, time and explicit zone/offset; a bare time of
  day is not a date, so it stays `reset_text`. `error`/`final_text` are never
  replaced. It does not retry, resume, switch accounts or estimate allowance.
  Cooldowns (`claude_runs/cooldowns.py`) are opt-in: the operator declares a
  non-secret `claude.account_context` label (`JAKE_TOOLS_CLAUDE_ACCOUNT_CONTEXT`,
  default unset), copied into `spec.json` at launch. Nothing infers, verifies
  or hashes an account or credential, and equal labels are the operator's
  statement of one shared quota. Only a `usage_limit` with a dated `resets_at`
  is recorded, in `<runs_dir>/.cooldowns/<digest>.json` (atomic replace under a
  leaf lock; the newer observation wins; expired or unreadable reads as
  none). It is advisory: `claude_start` returns `cooldown` and still starts;
  a failed limit task's result shows the one now on record; a model turn
  answering a prompt sent after the observation clears it, and that prompt's
  time stays as a per-context watermark (7 days) so a delayed older failure
  write is dropped rather than resurrecting the cooldown.
- Shared memory (`hindsight.py`, README "Shared Hindsight memory for workers"):
  `hindsight.api_url` + `hindsight.bank_id` (+ `access`) are copied into
  `spec.json` at launch and become one explicit HTTP MCP server on Hindsight
  0.10.1's single-bank route `/mcp/<bank>/`, with `mcp__hindsight__recall`,
  `reflect` and (`readwrite`) `retain` approved by name, independent of
  `tools`. Never the multi-bank route, never a default bank, no system prompt
  (`--system-prompt ""` stays), no automatic retention, no credential.
  `handoff.py` must keep overriding `mcp_servers`/`mcp_tools` because specs
  merge field by field. Tests: `test_hindsight.py`, and `--slow`
  `test_hindsight_live.py` (real CLI against `fake_hindsight.py`, a real MCP
  server on the same route shape).
- Delegation recipes (`skills/claude-delegation/references/recipes.md`) are
  JSON fences the live tests run verbatim
  (`tests/test_delegation_recipes_live.py`, `--slow`); receipts under
  `tests/fixtures/delegation_recipes/receipts/` keep their sanitised
  outcomes, and `tests/test_delegation_recipes.py` pins them to the documented
  text. Change a fence's brief or grants, and rerun and re-save that recipe's
  receipt (`JAKE_RECIPE_RECEIPT_DIR`). The CLI decides what a rule approves;
  never document a rule form the matrix did not run.
- Local capacity is operator policy, `claude.max_live_tasks`
  (`JAKE_TOOLS_CLAUDE_MAX_LIVE_TASKS`, a positive integer, default 2; never
  document a universal two). `start_task` admits only while the shared live
  count (workers alive in the runs directory, under `.runs.lock`) is below the
  *admitting process's own* limit; `capacity_exceeded` reports `live` and
  `limit`. Processes with different limits over one directory share the count,
  not a ceiling (no coordination store; use one value). A limit lowered below
  the live count cancels nothing and refuses starts until `live < limit`.
  Replays of a `request_id` and the same-session exclusion are checked before
  capacity. Local slots are not provider quota (see cooldowns above).
- After a `max_turns` stop the worker asks for one handoff
  (`claude_runs/handoff.py`): the same session resumed through
  `ClaudeAgent` with no tools, `max_turns=1`, capped output, inside the
  task's deadline, cancel scope and remaining budget (skipped when spend is
  unknown). It is a separate worker-reported record; the outcome stays
  `failed`/`max_turns`. Each refused call is recorded once, with its
  source: `runtime` (the real-time message or the result's record) or
  `inferred_from_tool_error` (only the exact observed error openings in
  `worker.INFERRED_REFUSAL`). `claude_list` pages run directories newest
  first. The server sends no notifications; callers poll.
- Contract models the tools return live with their domain, not in `mcp/`:
  `clockify_jira_sync.SyncReport` (also what `clockify jira-sync --json`
  prints), `claude_runs.RunState`/`ClaudeStartResult`, and the
  `claude_runs.service` results (`Observation`, `TaskView`, `SteerResult`,
  …). `claude_*` tools send compact JSON and omit null fields
  (`structured_tool(omit_none=True)`), so every nullable result field needs
  a default (a test walks the models). A framework argument-validation
  failure is converted to `invalid_argument` (`server.JakeMCPServer`), so
  input-schema bounds and enums keep the error contract.
- Tests: `tests/conftest.py` provides `mcp_server`/`mcp_env` fixtures that
  spawn the real server over stdio under an isolated `HOME` with no
  secrets. Protocol and result-shape tests go through them. Process
  lifecycle tests (worker, cancel, timeout, server death) use real processes
  and check the process table afterwards; a lifecycle test that stubs the
  process is not a lifecycle test. The `fake_env` fixture puts
  `tests/fake_claude.py` first on PATH as `claude`. It speaks the CLI's
  stream-json and control protocol, so the real server, worker and SDK run.
  It proves mechanics only; `test_claude_tasks_live.py` (slow) is the
  real-model acceptance, over both the ordinary tools and native Tasks.
  `tests/tasks_client.py` is a minimal Tasks client on `mcp`'s
  `ClientExtension` API; `test_mcp_tasks.py` drives the real server with it
  and with a legacy session. The fake must keep speaking to the in-process
  `ask_parent` server as the real CLI does (MCP handshake, fresh JSON-RPC
  ids, `notifications/cancelled` to abandon): `mcp` 2 enforces all three. Inside a delegated worker, which is a child
  subreaper, orphaned test processes linger as zombies until it exits, so
  two zombie-only group assertions in `test_claude_runs_live.py` fail
  there. They pass under a reaping init.
- Packaged skills live in `src/jake_tools/skills/<name>/SKILL.md`, are served
  as `skill://<name>` resources (references as
  `skill://<name>/references/<file>.md`); that is the only delivery route, and
  guidance must work through MCP alone (no `skill_view`, plugin skill ids or
  profile secrets). They describe what the tools actually return.
- `tests/test_hermes_mcp.py` (`--slow`) is the real-Hermes acceptance test
  (discovery, resources, prompts, one call through Hermes' MCP dispatch). It
  needs `JAKE_TOOLS_HERMES`, a Hermes launcher path; without it, it is
  blocked, not passed.

## Transcription service (`transcription-service/`)

A long-running native macOS HTTP service (Swift, Hummingbird, FluidAudio
0.17.7 pinned) that keeps Parakeet TDT v3 and the offline pyannote diarizer
warm and turns an ordered multipart upload into one structured transcript.
`jake_tools.transcription` is its only client and `operations/transcribe.py`
the only tool; both are plain HTTP and import no native dependency. Contract,
limits, concurrency and cancellation semantics: `transcription-service/README.md`.

- The tool is registered only when `transcription.api_base_url` is set.
  `TRANSCRIPTION_API_TOKEN` is a secret (environment-only, scrubbed from
  worker environments). Doctor's `transcription` check is required once an
  endpoint is configured.
- No alternate backends, no direct inference in Python, no job store, no
  polling API. One job at a time, bounded queue, client disconnect cancels at
  the engine's next check, and so does the service's own
  `--request-timeout-seconds` (answered `504 timeout`). Timeouts are reported
  with `retry_safe=false` and never retried.
- Build with Nix only: `nix build .#transcription-service` (nixpkgs' Swift
  and Apple SDK; SwiftPM dependencies vendored from `Package.resolved` by
  `fetchSwiftPMDeps`), and test with
  `nix build .#checks.aarch64-darwin.transcription-service` (the `swift-test`
  pre-commit hook). No Xcode or Command Line Tools fallback. SwiftPM fetches
  every binary target even when only a disabled trait links it, so
  `nix/fluidaudio-drop-nemo-binary-target.patch` removes FluidAudio's NeMo
  xcframework; refresh it with a FluidAudio bump. After changing
  `Package.resolved`, update the `fetchSwiftPMDeps` hash.
- Models are a Nix output too: `nix build .#transcription-models` fetches the
  files `provision` writes, pinned by revision and SHA-256 in
  `nix/transcription-models.json`, and is passed read-only as `--models-dir`.
  Regenerate the manifest whenever `ModelPins` or FluidAudio changes. Core ML
  still keeps a per-user, per-device cache under the running user's
  `~/Library/Caches` (and `/var/folders`); that is Core ML's, not ours.
- Swift tests run the real server with a stub engine: mechanics only. The
  real engine is exercised by `tests/test_transcription_live.py` (`--slow`,
  macOS; point `JAKE_TRANSCRIPTION_SERVICE_BIN` and `TRANSCRIPTION_MODELS_DIR`
  at the Nix outputs); it is the only evidence about transcription quality or
  latency, and a fake transcript is never that evidence.
- Fixtures for the live test are synthesised with macOS `say`
  (`tests/transcription_fixtures.py`); nothing binary is committed.

## Claude Artifacts (`jake_tools.artifacts`, experimental)

`artifact_publish` / `artifact_read` / `artifact_update` drive Claude Code's
`Artifact` tool through `ClaudeAgent`. Rules that keep them honest:

- The tool is hidden from SDK sessions unless the **undocumented**
  `CLAUDE_CODE_ARTIFACT` is set; it is set only through `AgentSpec.env` for
  these sessions, never persisted, never given to `claude_start` tasks, and
  the operations are off unless `artifacts.enabled`. Do not describe the route
  as supported.
- Authority is enforced *before* each call by `AgentSpec.tool_guard`, a
  `PreToolUse` hook (it runs even when an allow rule approves the call) that
  judges the **raw model-supplied input** and denies any key outside the
  permitted set, including `__*`. A `can_use_tool` second layer was tried and
  dropped: the runtime enriches its input with private metadata. Do not add a
  compatibility layer for that metadata. The post-hoc "call the guard never
  saw" check is detection, not enforcement. A managed policy that disables
  the hook is untested; a CLI upgrade needs `tests/test_artifacts_live.py`
  re-run. A read session has no write authority. Never loosen `ArtifactGuard`,
  add `url`/`edits`/`share`/`delete`/`list` permission to the NEW-publish or read
  guards, or suppress
  `setting_sources` (user/project/managed `enableArtifact`/deny rules apply).
- The model is a conduit: outcomes come from the tool's own results, and
  `verified` is a byte comparison in code of a *separate* readback with the
  pinned document shell (`parse.SHELL_PREFIX`) removed. A different shell is
  an unverified result, not an approximate pass. Re-record
  `tests/artifact_fixtures/` if Claude Code changes its wording.
- **Update is a separate, narrow tool** (`artifact_update`,
  `ArtifactGuard.for_update`): the one exact file onto the one exact existing URL,
  after a separate read session proved it is yours, private and at the caller's
  `expected_version_id` (a stale version fails with `plan_stale` before any write
  session starts). Its guard allows one `read` of that URL (the tool refuses an
  update until the live version was viewed), the `Read` of the file and one
  `publish` carrying that `url`; nothing else, and the publish is denied until
  the session's own read result was *observed* (fed to the guard as it streams)
  as owned, private and at the expected version. The `url`-in-publish restriction
  above is unchanged for new publishes. The native read and write are separate
  calls: never describe an update as an atomic compare-and-swap, and do not add
  locks or private endpoints. A write landing between the last observed read and
  the publish is undetectable without a native CAS; the tool's own stale-copy
  refusal is the only (untested) protection. Visibility and artifact id must be unchanged on the
  readback; private-only is an explicit limit.
- No call is retried (a repeat creates a second artifact, or a second version); an unknown outcome
  says `retry_safe: false` and the URLs seen. A receipt survives failures after
  a publish.
- Tests that publish are gated behind an explicit environment value and must
  never run without authority for that exact synthetic fixture.

## Microsoft newsletters (`microsoft_auth`, `newsletter_entries`)

`newsletter_list`/`newsletter_get` and
`newsletter_create`/`newsletter_update` (which change a list its readers can see). Operator docs:
`docs/microsoft-auth.md`; agent guidance: `skills/newsletters`.

- Identity is configuration, never a tool argument or a fallback:
  `microsoft.tenant_id`, `microsoft.client_id`, `microsoft.account` and the
  secret `MICROSOFT_REFRESH_TOKEN` (optional `MICROSOFT_CLIENT_SECRET`), both
  environment-only and in `SECRET_VARIABLES` (so scrubbed from workers and read
  from the server environment). The tools are always listed;
  unconfigured calls are `missing_credentials` naming variables/settings.
- `MicrosoftTokenProvider` is the refresh-token grant over `requests` (no new
  dependency; MSAL documents its refresh-token method as migration-only and
  `azure-identity` has no equivalent). One token per resource audience (Graph,
  SharePoint), refreshed in the last fifth of its life (max 5 min) under one
  lock so concurrent callers refresh once. Operations build their client per
  call, so the provider is shared per exact settings (`shared_provider`). A
  rotated refresh token lives in memory only: nothing is persisted. Every
  refresh checks the ID token's `tid`/account against the configuration
  (a guard against mis-provisioning, not an authorisation boundary).
  Failures are `MicrosoftAuthError.kind`s mapped to `access_denied`
  (`grant_invalid`, `interaction_required`, `client_rejected`,
  `resource_not_granted`, `identity_mismatch`, `identity_unverified`) or
  `upstream_error` (`unavailable`); messages carry codes and an operator
  action, never a token, secret, host or free upstream text.
- `NewsletterClient` stays the one HTTP client; `AzureCliTokenProvider` now
  honours the CLI's reported expiry and is for the interactive command only.
  Tokens are never cached for the process lifetime.
- Writes: `create_entry`/`update_entry` read back and compare (title, body,
  untouched fields, every supplied attachment by downloaded bytes). A step is
  `applied`, `failed` (a 4xx the upstream refused, or no token, so nothing was
  sent), `uncertain` (transport error, timeout, 5xx, unreadable 2xx) or
  `not_attempted`. Never retry a write or recreate an entry; recovery is
  `newsletter_update` on the known ID, which skips an attachment already
  present with identical bytes and refuses a same-name different-bytes one.
  Attachments are only added. A field PATCH is not a compare-and-swap.
- Tests (`tests/fake_microsoft.py`) run the real stdio server, real clients and
  provider against a local fake of the token endpoint, Graph and SharePoint
  (audience- and expiry-enforcing, with failure injection). That proves
  mechanics only: nothing here establishes live Microsoft compatibility
  (SharePoint rich-text normalisation, throttling, tenant policy).
- Attachment paths are read with the server's own permissions: a granted
  `newsletter_publish` agent can upload any readable file it names.

## SharePoint and Teams files (`sharepoint_files`, `file_transfers`)

Six `sharepoint_*` tools (guide: `skills/sharepoint-files`; operator CLI
`jake-tools files`, which builds the same `OPERATION.build(config)` handlers).
Same identity as the newsletter (`microsoft_settings`, `shared_provider`), plus
delegated `Files.ReadWrite.All`. Maintainer invariants:

- Items are Graph `drive_id` + `item_id`, validated and quoted into paths. Links
  must be on `files.sharepoint_hosts`; `files.graph_root` exists only for fixtures.
- Content `302`s are followed by hand WITHOUT the bearer token; pre-authenticated
  download/upload URLs, and the link being resolved, never appear in an error.
- Uploads use `createUploadSession` for NEW files only (`fail`/`rename`, never
  `replace`; one fragment, at most 50 MiB). **Replacing an existing file is
  withheld on purpose**: Microsoft documents `if-match` (412) only at session
  creation, not at the commit, and none for "Upload small files", so no safe
  conditional replace exists. Do not add one with locks, versions or retries; it
  needs a documented commit-time condition (open on #11, a separate investigation).
- A failure before content is sent (even a 5xx) is `not_applied`; no answer, a
  5xx or an unreadable 2xx on the content PUT is `write_uncertain`; success is
  only `verified: true` after the exact item's bytes were downloaded and compared.
  Never retry a write. Source/destination paths use the server's permissions.
- `tests/fake_drive.py` (extends `fake_microsoft.py`) is mechanics only; live
  link forms, rendering and throttling were not checked against a tenant.

## External dependencies

| Tool               | Used by                                               |
| ------------------ | ----------------------------------------------------- |
| `uv`                | dependency management and script runner                |
| `claude-agent-sdk`  | LLM calls; drives the local `claude` CLI               |
| `az` (Azure CLI)    | `jake-tools newsletter` commands only (the caller's own Graph token) |
| `nix`               | building and testing `transcription-service/` (Swift and SDK from nixpkgs); `nix develop` for the dev tools above |
If a required external tool is missing, report the blocker. Do not mock
preflight checks or skip them silently.

## Commands

### `codex-usage-alert`

```bash
jake-tools codex-usage-alert
```

Reads Hermes' existing OpenAI Codex OAuth access token without refreshing or
persisting credentials. Prints only on a new 20%, 10%, or 5% remaining
threshold crossing; silence with exit status 0 means no alert is due.

### `clockify`

```bash
jake-tools clockify jira-sync --dry-run --json
jake-tools clockify jira-sync --issue SF-304 --dry-run --json
jake-tools clockify jira-sync --issue SF-304 --apply --json
```

`jira-sync` defaults to planned or underway Jira work assigned to `currentUser()`
(status To Do, In Progress, Blocked, or In Review). Repeat
`--issue KEY` to reconcile exact work items regardless of assignee or status
(a missing task for a Done issue is created, then completed). Dry-run is
the default; `--apply` creates, renames, reactivates, relocates, or completes
records and then re-fetches every changed Clockify object for verification.
The command never deletes records.

Relocation (`clockify_relocation.py`, `RELOCATE_TASK`): Clockify's task update
has no `projectId`, so a misplaced task is replaced, not moved. A staging task
`[relocating KEY from SRC]` is created first, then entries are re-pointed
(full `UpdateTimeEntryRequest` echo, whole-record read-back), then the source
is retired as `[relocated KEY to NEW]`, and the staging task is promoted last.
Markers never claim the key. The staging marker carries `#BASELINE`, a hash
of the source task's fields and every entry minus its placement, written
before any entry moves. Preview resumes only exact marker relations whose
records still match it, and also in the default scope after the issue leaves
discovery; every other shape, and a baseline mismatch, is a conflict. Removing
the hash is the human "accept", and it is re-recorded first. The digest binds
whole-record fingerprints.
No rollback: a mismatch stops with `failure.recovery`. Entry rates are
read-only in the API, so a changed rate is reported, never asserted preserved.
Writes are tested over HTTP in `test_mcp_clockify_jira_sync.py`, including a
sweep that interrupts every write (refused and landed-then-500).

### `newsletter`

```bash
jake-tools newsletter list --limit 10
jake-tools newsletter list --body --json
jake-tools newsletter add "Title" < body.txt
jake-tools newsletter edit ITEM_ID --title "New title" < body.txt
```

`list` is read-only. `add` and `edit` mutate the CSU Weekly Newsletter
SharePoint list via Microsoft Graph. Body text is read from stdin for `add`
(required) and `edit` (optional). `--attach` can be supplied multiple times.

Requires `az login` to the CSU tenant for a Graph access token. That is the
interactive command line's own identity. The MCP `newsletter_*` tools
never use it (see "Microsoft newsletters" below).

### `files`

```bash
jake-tools files resolve --url "https://csuses.sharepoint.com/..." --json
jake-tools files list DRIVE_ID ITEM_ID --limit 50
jake-tools files download DRIVE_ID ITEM_ID /abs/new/path.docx
jake-tools files upload DRIVE_ID FOLDER_ID /abs/list.xlsx --on-conflict fail
```

Also `get` and `pdf`. These run the MCP operations, as the configured Microsoft
account (environment only: no `.env`, no config file; not `az login`). `upload` adds a file other
people can see. A nonzero exit includes an uncertain write, whose `outcome` and
`next_step` are printed.

## Boundaries

- Do not commit secrets, tokens, or credentials.
- `newsletter` writes to SharePoint by design. `--dry-run` suppresses those
  writes where the command offers it.
