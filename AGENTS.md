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

A test can (and often does) carry only `slow` — see `transcription/`'s own
real-model tests (`tests/test_transcription_*.py`, `tests/test_transcribe_cli.py`),
named `test_live_...` for a convenient `-k live` selector even though `live`
itself isn't the marker gating them.

### Code conventions

- Keep CLI commands thin; put orchestration and domain logic in package modules.
- Match surrounding code. Prefer typed Pydantic models over ad hoc dicts.
- Ruff rules: `E`, `F`, `I`, `UP`, `B`, `SIM`, `C4` (see `pyproject.toml`).
- Pyright must pass (`pyrightconfig.json`).
- All LLM calls go through `ClaudeAgent` in `claude.py`. Nothing else imports
  `claude_agent_sdk` directly. `transcription/` (behind the `transcript`/
  `transcribe` commands) is the exemplar of every convention below: options
  decorators, `@coro`, a `ClaudeAgent(run_query=fake)` test seam, and domain
  errors + `ClaudeAgentError` mapped to a clean `ClickException`.
- Commands that call the LLM use the `agent_options` decorator in
  `cli/options.py` (adds `--model`/`--effort` and injects a typed
  `AgentOptions`) and `@coro`, applied closest to the callback, which runs the
  async callback with `asyncio.run`. The handler builds the agent itself —
  `agent_options.agent()` for a `ClaudeAgent`, or `.spec()` for just the
  `AgentSpec` — there is no shared factory seam to route through, so a test
  fakes the call by monkeypatching `AgentOptions.agent` (or the dependency it
  returns), not by injecting a factory. `--effort`'s choices come from
  `typing.get_args(EffortLevel)`, imported from `claude.py` (never
  `claude_agent_sdk` directly).
- `AgentSpec.tools` defaults to an empty tuple, which is genuinely tool-less.
  Never pass `tools=None` to `ClaudeAgentOptions`: the SDK then omits `--tools`
  and the agent inherits Claude Code's full default toolset.
- Tests inject a fake at the `run_query` seam (see `tests/test_claude_agent.py`),
  so prompt rendering, schema injection, and usage accounting stay real.
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
- CLI tests stay thin: they monkeypatch the constructor method on an options
  model (e.g. `ClockifyOptions.inventory_client`) or the client/orchestration
  symbol in the CLI module, and assert flag parsing and delegation. Real
  behaviour — reconciliation logic, HTTP clients, etc. — is tested at the
  library seam with injected fakes, not through the CLI.

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
