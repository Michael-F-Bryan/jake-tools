# Daily Report CLI Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan. Work out the implementation details yourself; this plan fixes direction and constraints, not code.

**Goal:** Add a production `jake-tools daily-report` command that runs the proven six-lane daily-report workflow with structured LLM outputs, deterministic artefacts, mechanical validation, AI usage/cost accounting, and honest failure handling.

**Tech stack:** Python 3.14, Click, Pydantic, pytest, pyright, `jake_tools.hermes.Hermes`, `jake_tools.prompting.StructuredPrompt`.

---

## Architecture

Keep the Click command thin: parse flags into typed options, construct the concrete production stages, call a top-level coordinator, render the result.

- **Coordinator as a function:** `run_daily_report(options, stages) -> DailyReportRun` is pure orchestration. It calls methods on a `DailyReportStages` protocol in the right order and returns a typed result. Not a class with many injected callables.
- **Stages protocol:** tests swap in a fake `DailyReportStages`; production provides `HermesDailyReportStages` built on `Hermes`.
- **Structured outputs:** every lane returns a typed `LaneOutput` via `StructuredPrompt[LaneOutput]`. Agents never write files; the runner writes artefacts/evidence from the structured output.
- **Side effects at the edges:** stages write prompts/logs/artefacts/evidence/manifest/synthesis. Validation and manifest construction stay pure and easy to test.
- **Typed everything:** model lane state; do not pass `dict[str, Any]` around.

Use the existing `Hermes` abstraction and `StructuredPrompt`/Jinja machinery, but treat the touched seams as things to harden, not patterns to copy blindly.

## Seams to harden before building on them

- **Hermes structured result:** daily-report lanes need both the typed payload and a `HermesResult` for cost accounting, with explicit per-lane model/provider. Expose this as a public method (route the existing `run_structured()` through it) rather than calling a private method or mutating shared state. Preserve the existing JSON-repair behaviour.
- **No shared mutable usage state:** lanes run concurrently, so accounting must use returned `HermesResult` values, never `Hermes.last_result`.
- **Shared AI usage accounting:** the token/cost models and aggregation helpers currently live in transcript code. Daily report is the second caller — extract them into a shared `jake_tools.ai_usage` module (re-export from the old location if needed) instead of duplicating and letting the shape drift.
- **No hidden DI:** do not use Click `ctx.obj` or the existing `@hermes` decorator. Construct `Hermes()` explicitly in the command body.

## V1 cut line

Build the smallest useful version Michael would trust:

- date-scoped working directory; typed CLI options;
- six fixed lanes (inbox triage always runs, envelope-only);
- typed `StructuredPrompt[LaneOutput]` per lane;
- parallel lane execution with per-lane logs;
- runner-owned artefact/evidence writes from structured output;
- mechanical validation; `manifest.json`;
- deterministic `report.md` and `summary.json` built only from verified outputs;
- per-lane and total AI token/cost accounting;
- `--json` output; non-zero exit on failed validation;
- docs, tests, pyright green.

Do **not** build model-written final synthesis in V1. The command must **not** mutate memory, skills, email, cron, Obsidian, SharePoint, or git.

## The six lanes

`build_lane_specs(options, paths)` always returns all six, every run:

1. `session-hindsight` — judgement model
2. `memory-candidates` — evidence model
3. `skill-review` — evidence model
4. `failure-patterns` — judgement model, evidence file
5. `transcripts-and-dumc` — evidence model, evidence file
6. `inbox-triage` — evidence model, evidence file, envelope-only

## Key constraints

- **Safe default:** report/propose only. No external mutations.
- **Machine-readable output:** `--json` writes valid JSON to stdout with no progress chatter.
- **Honest evidence:** cite real evidence or explicit blockers; no simulated evidence. The runner writes files; validation checks them. Do not trust model claims.
- **Inbox is envelope-only:** never read email bodies or draft replies.
- **No import-time `Path.cwd()` defaults:** use CLI injection or `Field(default_factory=Path.cwd)`.
- **Prompts live in the daily-report package,** not the Click module.
- **Keep each commit green.**

## Suggested module layout

A starting point, not a contract — adjust as the implementation demands:

```text
src/jake_tools/
  ai_usage.py
  cli/daily_report.py
  daily_report/
    models.py paths.py lanes.py prompts.py
    stages.py manifest.py coordinator.py
    synthesis.py validation.py
```

## Validation

Mechanical checks over runner-written artefacts: missing file, empty artefact, required sections present, no banned placeholder phrases, heading normalisation. Raise a domain-specific error for hard synthesis invariants.

## Synthesis (deterministic)

`report.md` is an index over verified artefacts, not a second interpretation of the day. `summary.json` validates against a typed `DailyReportSummary` covering date, status, paths, lane count, failed lanes, aggregated findings/actions/caveats, and `ai_totals`.

## Testing approach

- Test the coordinator with a fake `DailyReportStages` that records calls and returns typed outputs; assert ordering, observable files, and status. Don't patch individual functions.
- Test production stages with `Hermes(agent_factory=...)` and a fake agent; assert provider/model pass-through and that prompt/artefact/evidence/log files are written from structured output, with usage from the returned `HermesResult`.
- Cover failure paths: a failing `run_lane()` and a failing validation should still produce manifest/report/summary with status `fail`.
- Test the CLI at the seams (`build_options()`, the result emitter, help text, invalid date) rather than invoking the full command.

## Acceptance criteria

- `jake-tools daily-report --date YYYY-MM-DD` creates a date-scoped working directory and runs all six lanes every time.
- Lanes return structured `LaneOutput`; the runner writes prompts, logs, artefacts, evidence, `manifest.json`, `report.md`, and `summary.json`.
- Lanes run through `Hermes` with explicit per-lane provider/model and `StructuredPrompt`.
- The coordinator is `run_daily_report(options, stages)` with a fakeable `DailyReportStages` protocol; no `ctx.obj` DI.
- AI usage/cost is recorded per lane and totalled.
- Validation runs before final status; the command exits non-zero on failure.
- `--json` returns stable machine-readable output; full test suite and `pyright` pass.
- Update `README.md` and `AGENTS.md`: six lanes always run, inbox is envelope-only, nothing external is mutated, all artefacts land under the work directory, `--json` is for automation, token/cost totals are in the summary.

## Deferred scope

After one successful real V1 run, consider in order: final critic lane (reads manifest + all artefacts/evidence) → model-written final synthesis → durable event JSONL / interruption recovery → deterministic session manifest instead of `session_search` for date scoping → semantic claim validation. Also deferred: any mutation workflows for memory/skills/email/cron/Obsidian/SharePoint/git.
