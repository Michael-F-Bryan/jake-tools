# Daily-report lane workflow findings — 2026-06-20

Session-specific detail from running a multi-lane daily-report workflow with separate CLI sub-agents and a final critic.

## Durable lessons

- For report workflows with several independent lanes, create a deterministic artefact manifest before final critic runs:
  - expected lane artefacts,
  - evidence files,
  - writer model/provider,
  - return code,
  - byte size,
  - timestamps,
  - validation status.
- Write lane state incrementally as JSONL on lane start and completion. Do not wait until all futures finish before recording durable state; interruption otherwise loses progress visibility.
- Give every lane exclusive ownership of its artefact path. Do not manually rerun a lane against the same file while a coordinator-spawned lane may still be running.
- Treat sub-agent self-reports as untrusted. The coordinator/final critic must read artefacts from disk and validate them before synthesis.
- Keep source classes separate in the final report:
  - target-date facts,
  - historical session evidence,
  - envelope-only triage leads,
  - vault-search results,
  - prototype/workflow feedback.
- Use stronger models for judgement-heavy lanes and cheaper models for narrow evidence lanes only when mechanical validators are strict.
- For date-bounded reports, `session_search` alone is not a date-range database. Use a deterministic session manifest from the Hermes SQLite store where available, then use `session_search` for cited transcript evidence.
- If the workflow is being productionised, keep the skill thin and put orchestration in repo code: typed lane specs, prompt templates, event logs, manifests, validators, final synthesis, and tests. The skill should define when to run it, safety boundaries, expected artefacts, and validation gates.
- Prefer a command shape such as `jake-tools daily-report --date YYYY-MM-DD --provider ... --judgement-model ... --evidence-model ...` over copying a one-off `_working/.../run_lanes.py` script between sessions.
- Model tiering is a policy, not a hard-coded model name: strongest available model for session hindsight, failure-pattern analysis, final critic, and final synthesis; cheaper models only for bounded evidence lanes with strict validators.
- Inbox lanes need an explicit consent boundary. Default to envelope-only; only read bodies or draft replies behind a deliberate flag/instruction, and never draft from envelope metadata alone.

## Repo-backed command shape

When turning a lane workflow into durable code, use typed objects rather than ad hoc dicts:

- `LaneSpec`: lane name, provider/model tier, prompt template, artefact path, evidence path, required sections, timeout, and safety mode.
- `LaneEvent`: JSONL start/completion/validation records with timestamps.
- `LaneResult`: return code, log path, artefact path, evidence path, byte sizes, and validation status.
- `Manifest`: all lane results plus required-file checks, banned-placeholder scan results, and machine-readable caveats for final critic/synthesis.

Useful validators to implement in code:

1. required artefact exists and is non-empty;
2. required headings present with mild normalisation, e.g. `Rejected / weak claims` satisfies `Rejected/weak claims`;
3. declared evidence files exist;
4. JSON summaries parse and match schema;
5. banned placeholder phrases are absent except when quoted as prompt/history evidence;
6. “no findings” claims cite exact searches/checks;
7. “tool unavailable” claims cite a preflight check;
8. envelope-only lanes do not draft replies or imply resolution.

## Validation pattern

Before synthesis:

1. Check every expected artefact exists and is non-empty.
2. Check required headings with mild normalisation, e.g. `Rejected / weak claims` should satisfy `Rejected/weak claims`.
3. Scan banned placeholder wording, but classify quoted prompt text and historical failure-query strings as references rather than violations.
4. Parse any machine-readable summaries, e.g. `summary.json`, before claiming completion.
5. Read back the final report, summary, runbook, and final critic.

## Model selection pattern used

- High-judgement lanes: `gpt-5.5` — session hindsight, failure patterns, final critic, final synthesis.
- Narrow evidence/discovery lanes: `gpt-5.4-mini` — memory candidates, skill review, transcript/DUM-C search, inbox envelope triage.

The exact model names are session-specific; the durable rule is to allocate the strongest model to cross-source judgement and final verification, not to every mechanical lane.
