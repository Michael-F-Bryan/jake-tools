# Daily Report Prototype

Condensed design notes for a date-scoped hindsight report that reviews one day's work without silently mutating durable state.

## Intended Use

Use this when the user wants a daily review that:

- takes an explicit date;
- reviews that day's sessions;
- extracts durable memory candidates;
- identifies skills used and possible patches;
- spots repeated failure patterns;
- surfaces open threads worth revisiting;
- checks adjacent evidence like Obsidian notes, transcript/DUM-C artefacts, and unreplied email.

## Recommended Shape

Prefer a **thin skill / thick repo code** split.

- The skill defines triggers, guardrails, output expectations, storage routing, and the validation contract.
- Repo code or a coordinator script does the heavy lifting, fan-out, evidence collection, report assembly, and deterministic validation.
- If this lives in a repo like `jake-tools`, expose it as a real command rather than reusing a one-off `_working/.../run_lanes.py` script.

This avoids turning SKILL.md into an application and keeps the operational logic testable.

For repo-backed implementations, model the workflow explicitly:

- `LaneSpec`: name, provider/model tier, prompt template, artefact path, evidence path, required sections, timeout, and safety mode.
- `LaneEvent`: JSONL start/completion/validation events so interrupted runs still have durable progress.
- `LaneResult`: return code, log path, artefact/evidence byte sizes, and validation status.
- `Manifest`: expected files, writer model/provider, timestamps, validation status, caveats, and final-critic inputs.

Use model tiers rather than hard-coded model names: strongest available model for judgement-heavy lanes and final verification; cheaper models only for bounded evidence lanes with strict validators.

## Safe v1 Defaults

- **Report/propose only.**
- Do not auto-write memory.
- Do not auto-patch skills.
- Do not auto-send or auto-reply to email.
- If email context is strong, draft a proposed reply; otherwise report the missing context.

## Evidence Lanes

A good daily report coordinator usually needs separate lanes for:

1. **Session hindsight** — what happened that day, grounded in session transcripts.
2. **Memory / Obsidian candidates** — durable facts or notes worth proposing.
3. **Skill review** — skills used, missing steps, and patch candidates.
4. **Failure patterns** — repeated friction, retries, or legibility issues.
5. **Transcripts + DUM-C / domain artefacts** — new external artefacts worth linking.
6. **Inbox triage** — unreplied or important messages needing attention.
7. **Final critic** — checks that the report answers the real question and does not overfit one noisy session.

Keep lanes separate so evidence and uncertainty stay visible.

## Output Layout

For repo-backed runs, use a predictable dated working directory such as:

- `_working/daily-report-YYYY-MM-DD/report.md`
- `_working/daily-report-YYYY-MM-DD/summary.json`
- `_working/daily-report-YYYY-MM-DD/notes.md`
- `_working/daily-report-YYYY-MM-DD/plan.md`
- `_working/daily-report-YYYY-MM-DD/subtasks/`
- `_working/daily-report-YYYY-MM-DD/evidence/`
- `_working/daily-report-YYYY-MM-DD/drafts/`

This keeps the final report distinct from scratch notes and per-lane artefacts.

## Key Judgement Rules

- Patch existing umbrella skills before inventing a new narrow skill.
- When checking inboxes, only draft replies when adjacent context is actually sufficient.
- Prefer reporting evidence gaps over confident filler.
- Use the report to stage future actions; let a foreground session decide what gets promoted into memory, skills, notes, or outbound comms.
- Treat sub-agent summaries as untrusted until the coordinator has read the claimed artefacts from disk.
- Reject “no findings” claims unless the lane lists exact searches/checks performed.
- Reject “tool unavailable” claims unless the lane includes a preflight check.
- For date-bounded session evidence, build a deterministic session manifest where possible; `session_search` is useful for cited transcript evidence but is not a date-range database.
- Keep envelope-only inbox findings as triage leads only. Do not draft replies, infer resolution state, or imply action completion from metadata alone.

## Validation Gates

Before final critic or final synthesis:

1. Generate an artefact manifest with expected lane files, evidence files, writer model/provider, return code, byte size, timestamps, and validation status.
2. Validate required sections mechanically, allowing harmless normalised heading variants such as `Rejected / weak claims`.
3. Scan for banned placeholder wording and classify quoted prompt/history references separately from actual evidence.
4. Parse machine-readable outputs such as `summary.json` before claiming completion.
5. Read back the final report, summary, runbook, manifest, and final critic.
