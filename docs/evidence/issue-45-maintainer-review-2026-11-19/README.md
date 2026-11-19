# Issue #45: maintainer review as a challenge to ownership (2026-11-19)

A retained, disposable before/after comparison for the `issue_maintainer_review` change
(see the decision on #26: no maintained runner, no automatic grading). It is for a human
to read; it does not establish reliability.

## What is here

| Path | Content |
|---|---|
| `prompt.diff` | The prompt change, against `origin/main` `23730e05` (the "before"). The final text is `src/jake_tools/mcp/prompts.py`. |
| `inputs/<case>/` | The frozen synthetic inputs: issue, discovery comment, planning comment, code excerpts. |
| `reviews/<case>/before.md`, `after.md` | The full review from one run of each. Only trailing whitespace/end-of-file newlines were normalised by the repository's pre-commit hooks; the text is otherwise as produced. `raw-runs.tar.gz` keeps the exact original outputs. |
| `raw-runs.tar.gz` | Everything else: every run's exact prompt, review and record, the superseded wordings and inputs, the harness scripts, `status.log`, and `MANIFEST.txt` saying what each directory is. |

## How it was run

Production `ClaudeAgent.run` through the real Claude Agent SDK and CLI (default model
`claude-sonnet-5`, `tools=()`, `setting_sources=()`, `max_turns=2`), once per cell. The
prompt text is the review prompt for a placeholder `synthetic-case:` URL, plus one
identical harness note (no tools; return the comment text instead of posting it) and the
case inputs appended. The reviewer therefore sees only the thread as given and cannot
inspect more code. All inputs are synthetic; no Jira or other repository was accessed.
Inputs were frozen before the `after` run of each case. Every retained run ended
`status: ok`.

Cases: `syn-705-subtle` is SF-705-style: a real responder limitation is found and the plan
answers it with an assess/publish split, state v2 with an in-memory v1 upgrade, and a
schema CI step. `syn-small` is a small sound fix. `syn-large` is a ~90-line pager module
replacing three hand-written loops that each caused an incident.

## What changed in the judgement

**`syn-705-subtle`.** Both variants found the same small fix for the board bug (write
`last_seen` on every run, keep one file) and the same signal fix. The difference is how the
rest was classified. In the baseline, 2 of 3 samples called the v2 state, migration and
schema CI step a correction to make "during implementation" (the retained sample asks
Michael only if he wants schema governance); one of three blocked on a decision. The final
`after` sample calls the CI job "a material expansion of ownership, not a routine
implementation choice" that needs Michael, narrows the responder change to the one-line
write, and keeps the required behaviour (sustained outage reported once, recovery
announced) explicit. That is a shift in where the line is drawn, seen in one sample, and
within the baseline's own spread, so I do not claim a rate. The baseline's third sample
caught a hazard (what `Alert.count` becomes under the new rule) that the retained `after`
sample did not; I cannot attribute that to the prompt.

**`syn-small`.** The first proposal drew corrections for real hazards in every variant, so
it did not show acceptance of a sound change; it is archived as superseded. The hazards
(exception type, zone-less dates, a `now` default evaluated at import, malformed versus
missing header) were fixed in the proposal and the case re-run once per variant. Both now
say "Ready to implement" and raise only non-blocking points. The `after` review adds an
explicit comparison with a cheaper alternative (swallow the error) and rejects it for the
stated purpose; neither review asks for a decision.

**`syn-large`.** Both accept the shared module and make the same substantive correction
(resume dedup would not fix INC-2). The baseline simply reviews the pager's contract; the
`after` review first weighs patching the three loops in place against the module, says the
module is proportionate because the contract interpretation would otherwise stay
duplicated, and so does not treat fewer lines as better.

## Limitations

- Synthetic inputs written by the implementer, who also chose the planted weaknesses.
- One sample per retained cell for `syn-small` and `syn-large`, three baseline samples for
  `syn-705-subtle`, one final-prompt sample per case. Model nondeterminism is visible
  across baseline samples, so differences above are tendencies in one run, not measures.
- The final prompt text is exercised only by the three `after` runs listed in the manifest.
  Earlier wordings and the blatant `syn-705` case (no final-prompt run) are in the archive,
  labelled as such. I did not re-verify each review's technical claims.
- No semantic assertion is made in code. `tests/test_mcp_prompts.py` pins only
  deterministic structure.
- The review prompt grew by roughly 170 words; no existing instruction was removed beyond
  the sentences the new text replaces.
