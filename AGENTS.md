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
  prompting.py    # typed Jinja prompts bound to a response model
  newsletters.py  # SharePoint Graph client
  transcription/  # meeting-transcription pipeline stages + pipeline.py composition
tests/            # mirrors packages above
```

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

A test can (and often does) carry only `slow` — see most of `transcription/`'s
own real-model tests (`tests/test_transcription_*.py`, `tests/test_transcribe_cli.py`),
named `test_live_...` for a convenient `-k live` selector even though `live`
itself isn't the marker gating them. One test in that family is a genuine
exception, actually gated by `live` rather than just named for it:
`tests/test_transcription_asr.py::test_local_transcriber_produces_nonempty_monotonic_utterances`
downloads real ASR/diarisation models from Hugging Face Hub (HF-gated -
requires an `HF_TOKEN` that has accepted the diarisation model's licence),
which is exactly the "real external service" `live` exists for.

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
  `claude_agent_sdk` directly. `transcription/` (behind the `transcript`/
  `transcribe` commands) is the exemplar of every convention below: options
  decorators, `@coro`, and domain errors + `ClaudeAgentError` mapped to a clean
  `ClickException`.
- Commands that call the LLM use the `agent_options` decorator in
  `cli/options.py` (adds `--model`/`--effort` and injects a typed
  `AgentOptions`) and `@coro`, applied closest to the callback, which runs the
  async callback with `asyncio.run`. The handler builds the agent itself —
  `agent_options.agent()` for a `ClaudeAgent`, or `.spec()` for just the
  `AgentSpec` — there is no shared factory seam to route through. `--effort`'s
  choices come from `typing.get_args(EffortLevel)`, imported from `claude.py`
  (never `claude_agent_sdk` directly).
- `AgentSpec.tools` defaults to an empty tuple, which is genuinely tool-less.
  Never pass `tools=None` to `ClaudeAgentOptions`: the SDK then omits `--tools`
  and the agent inherits Claude Code's full default toolset.
- `ctx.obj` is never set or read anywhere in this codebase. Every CLI
  dependency is built by the command that needs it: a decorator (see
  `cli/transcript_options.py`, `cli/options.py`, `cli/clockify.py`) stacks the
  relevant `click.option`s, pops their parsed values, and injects a typed
  Pydantic options model with dependency-constructor methods — e.g.
  `ObsidianOptions.vault_client()`, `ClockifyOptions.inventory_client()`,
  `AgentOptions.agent()` — via `ctx.invoke`. When a dependency takes no CLI
  flags at all (e.g. `NewsletterClient`), the handler just constructs it
  directly at the top of the function. `clockify`'s `--api-key`/
  `--api-base-url` are ordinary per-subcommand flags via a `clockify_options`
  decorator, the same shape as every other options decorator — there is no
  group-level state and no exception to this rule.


## MCP server (`jake_tools.mcp`)

`python -m jake_tools.mcp` (also the `jake-tools-mcp` console script) is the
stdio MCP server Hermes runs; design in `_working/mcp-server/README.md`.
Sub-commands: `serve` (default), `doctor`, `install-skills`, and the hidden
`worker RUN_DIR`. It is a separate entrypoint from `jake-tools` on purpose:

- Nothing on this path imports `jake_tools.__main__`, `dotenv`, or
  `jake_tools.transcription`, and nothing calls `load_dotenv()`. The process
  environment Hermes passes is the only environment. A test pins this.
- Stdout is the transport. Log to stderr only, never print.
- Configuration comes from `config.load_config()`: XDG paths, then flag >
  env > `$XDG_CONFIG_HOME/jake-tools/config.toml` > `$XDG_CONFIG_DIRS` >
  default. Secrets are environment-only (`NAME` or `NAME_FILE`) and are never
  read from `config.toml`. Every value is a `Resolved` that remembers its
  source; `doctor` prints `display_value()`, which redacts secrets. Each
  sub-command loads its own `Config`; there is still no `ctx.obj`.
- Missing credentials never stop startup. Clients are built per call inside
  the tool that needs them and raise `ToolError("missing_credentials", ...)`
  naming the variable, never the value.
- `mcp/server.py` registers; `mcp/tools/<tool>.py` holds one thin handler
  each, registered through `structured_tool`, which sends the handler's
  Pydantic result as structured content and turns a raised
  `mcp.errors.ToolError` into an `isError` result with the stable
  `{code, message, detail?}` payload. Codes are the `ToolErrorCode` literal;
  do not invent new ones without adding them there. No `outputSchema` is
  advertised, by design, so error payloads never fail schema validation.
- Handlers are `async`. Blocking `requests` clients run under
  `anyio.to_thread.run_sync`. Timeouts and cancellation use anyio cancel
  scopes (`anyio.fail_after`, `move_on_after`), never `asyncio.timeout` or
  `wait_for`: the Claude SDK's child-process cleanup only runs under anyio
  cancellation.
- The server holds no state that matters. Delegated tasks live in run
  directories (`claude_runs/models.py` documents the files); the worker is
  its own process group and outlives the server. No queue, no database.
- Contract models the tools return live with their domain, not in `mcp/`:
  `clockify_jira_sync.SyncReport` (also what `clockify jira-sync --json`
  prints), `claude_runs.RunState`/`ClaudeStartResult`,
  `session_store.models` (phase 3).
- Tests: `tests/conftest.py` provides `mcp_server`/`mcp_env` fixtures that
  spawn the real server over stdio under an isolated `HOME` with no
  secrets. Protocol and result-shape tests go through them. Process
  lifecycle tests (worker, cancel, timeout, server death) use real processes
  and check the process table afterwards; a lifecycle test that stubs the
  process is not a lifecycle test.
- Packaged skills live in `src/jake_tools/skills/<name>/SKILL.md`, are served
  as `skill://<name>` resources, and are installed by `install-skills`. They
  describe what the tools actually return.

## External dependencies

| Tool               | Used by                                               |
| ------------------ | ----------------------------------------------------- |
| `uv`                | dependency management and script runner                |
| `claude-agent-sdk`  | LLM calls; drives the local `claude` CLI               |
| `az` (Azure CLI)    | `newsletter` commands (Microsoft Graph token)          |
| `ffmpeg`/`ffprobe`  | `transcript`/`transcribe`: merge and cut audio          |
| `obsidian` (CLI)    | `transcript`/`transcribe`: resolve vault embeds         |

If a required external tool is missing, report the blocker. Do not mock
preflight checks or skip them silently.

`transcribe`/`transcript asr` additionally need `HF_TOKEN` (see `.env`) for
pyannote's gated diarisation model — only for a fresh ASR run over new
audio; a cached run, or a run over a pre-diarised transcript source, needs
neither ffmpeg nor a token.

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

`jira-sync` defaults to active Jira work assigned to `currentUser()`. Repeat
`--issue KEY` to reconcile exact work items regardless of assignee. Dry-run is
the default; `--apply` creates, renames, reactivates, or completes records and
then re-fetches every changed Clockify object for verification. The command
never deletes records.

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

Requires `az login` to the CSU tenant for a Graph access token.

### `transcribe` / `transcript`

```bash
jake-tools transcribe "2 Areas/Sunfish/2026-08-11 Team Sync.md"
jake-tools transcribe "...md" --assign "SPEAKER_03=Nikki Staltari"
jake-tools transcribe "...md" --assign "SPEAKER_04=Unknown" --finalise
```

`transcribe` is the porcelain: the full meeting-transcription pipeline
(merge/ASR or adapt, resolve speakers, chapterise, polish, minutes,
integrate) over one Obsidian prep note, one call. If speaker resolution
can't confidently name every voice it prints `{"status": "needs_input",
"requests": [...]}` and exits 3 instead of guessing; re-run with more
`--assign "SPEAKER_NN=Name"` flags, or `--assign "...=Unknown" --finalise`
to give up on the rest. ASR/diarisation (and the text-transcript adapter)
check the run cache first, so that resume loop never redoes finished
ASR/diarisation work; chapterise/polish/minutes currently re-run their LLM
calls on a repeat invocation of an already-complete run (a known
follow-up — see `transcription/pipeline.py`'s module docstring). See
README.md's Transcription section for the full behaviour and the
requirements list.

`transcript` exposes the same eight stages individually (`merge-audio`,
`asr`, `adapt`, `speakers`, `chapterise`, `polish`, `minutes`, `integrate`)
as plumbing sub-commands sharing one run cache — useful for debugging a
single stage. Both command groups follow the options-decorator DI
convention above; `transcription/pipeline.py` is pure composition over the
stage modules in `transcription/` and contains no stage logic of its own.

A coordinating agent without direct access to Michael's judgement can ask
Jake (his Hermes agent) for supporting context via
`hermes chat --quiet --query '...'` (`--resume <session_id>` to continue a
session) — e.g. which prep note to run, or how to answer a `needs_input`
request.

## Boundaries

- Do not commit secrets, tokens, or credentials.
- `newsletter` writes to SharePoint by design. `--dry-run` suppresses those
  writes where the command offers it.
