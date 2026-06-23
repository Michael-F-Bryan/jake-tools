# Daily report / analytical lane orchestration

Use this pattern when coordinating a daily report, hindsight review, inbox triage, or other multi-lane analytical workflow where each sub-agent writes a lane artefact.

## Coordinator contract

- Seed a date-scoped working directory before dispatch, e.g. `_working/daily-report-YYYY-MM-DD/`.
- Give every lane an exact output path under `subtasks/` and, where useful, an evidence path under `evidence/`.
- Post progress as lanes return if the user asked for live updates.
- Read every claimed artefact from disk before accepting the sub-agent summary.
- Treat the final critic as another fallible lane: use its critique, but verify its claims about file existence and missing evidence.

## Hard rejection rules

Reject or rewrite a lane artefact when it contains any of these:

- simulated session IDs, invented examples, or placeholder evidence;
- a “no findings” conclusion without citing the exact searches/checks performed;
- a claim that tooling is unavailable without a preflight command or equivalent check;
- a self-report that says the file was written when the file is missing;
- draft/action recommendations based only on envelope metadata or other partial context.

## Inbox lane preflight

For read-only inbox triage, use a staged preflight before judging unreplied messages:

1. Verify the mail tool exists and list configured accounts.
2. List folders for the relevant account(s).
3. List recent inbox envelopes and sent envelopes.
4. Only inspect message bodies if the current session has a clear consent boundary for read-only body access.
5. If body inspection is blocked, report envelope-only candidates and do not draft replies.

## Report synthesis

In the final report, clearly separate:

- verified activity from the target date;
- nearby-session or retrospective context;
- current-run prototype failures;
- blocked/partial findings.

The prototype failures are often the most useful migration notes, but they should not be framed as events from the target date.
