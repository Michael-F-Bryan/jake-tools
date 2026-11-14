# Artifact exposure probes, 2026-11-14 (issue #15)

Sanitised, machine-readable results of the probes summarised in
`docs/claude-tools.md` ("Re-checked 2026-11-14"). Read-only: no artifact was
published, edited or shared. Connector (`mcp__*`) tool and server names,
session ids, working directories, stderr and artifact titles/URLs are removed;
only built-in tool names, counts, versions and result status remain.

## Environment

- `claude-agent-sdk` 0.2.182 in a throwaway venv (`uv venv` + `uv pip install
  claude-agent-sdk`), bundled CLI 2.1.325 (binary sha256
  `7b6e484b1989a5a14a4b354709ccc8672518a3ec5b9ccd7981fc6254327fde67`),
  and the host's installed CLI 2.1.318 (probe B).
- claude.ai Max login, `apiProvider: firstParty`. Runtime, account and
  settings were not changed.
- Every run unset the parent session's `CLAUDECODE`, `CLAUDE_CODE_CHILD_SESSION`,
  `CLAUDE_CODE_SESSION_ID`, `CLAUDE_CODE_SESSION_ATTENDED`,
  `CLAUDE_CODE_MESSAGING_*`, `CLAUDE_CODE_EXECPATH`, `CLAUDE_PID` and
  `CLAUDE_EFFORT` (`env -u ...`). The SDK itself sets
  `CLAUDE_CODE_ENTRYPOINT=sdk-py`; nothing overrode it. For probe E the
  inherited `CLAUDE_CODE_ENTRYPOINT` and `CLAUDE_AGENT_SDK_VERSION` were also
  unset and the effective entrypoint was **not observed**.

## Probes

| File | Launch | Result |
| --- | --- | --- |
| `A-...` | SDK `tools=["Read","Artifact"]`, CLI 2.1.325 | `Artifact` not in `init.tools` |
| `B-...` | same, `cli_path` = installed CLI 2.1.318 | not exposed |
| `C-...` | `claude_code` preset, `setting_sources=["user"]` | not exposed |
| `E-...` | `claude -p --tools Read,Artifact`, CLI 2.1.325 | not exposed (gate unattributed) |
| `F-...` | **undocumented** `env={"CLAUDE_CODE_ARTIFACT": "1"}`, otherwise as A, `allowed_tools=["Artifact"]` | exposed (`Artifact`, `ArtifactComments`, `ArtifactData`); one `{"action": "list"}` call returned without error |

`F` is a disposable, process-local probe of an undocumented mechanism read
from the CLI's source. It is not a supported route and not an accepted
dependency. Its list result is stored only as a size (`content_chars`).

## Later receipts (same day)

- `guard-*.json` (from `guard_receipt.py.txt`): the hook-only form of the guard on
  the real runtime, with a blanket allow rule: consulted for allow-approved calls,
  denies stopped them (harmless calls only).
- `hooks-vs-settings-flags.json` (`hooks_settings_probe.py.txt`): `disableAllHooks` /
  `allowManagedHooksOnly` through `--settings` did not stop an SDK hook callback.
- `enforcement-layers.json` (`enforcement_matrix.py.txt`): hook with an allow rule
  (also under project `disableAllHooks`), and `can_use_tool` behind an `ask` rule
  with no allow rule. The `can_use_tool` design is NOT shipped (the runtime
  enriches that callback's input with private metadata and a real read failed). Managed-settings rows were discarded: the control
  (`managed_effective.py.txt`) showed `CLAUDE_CODE_MANAGED_SETTINGS_PATH` is not
  honoured, so a real managed policy is untested.
- The disposable artifact's publish, readback and update receipts are summarised
  in `docs/claude-tools.md` ("Demonstrated 2026-11-14"); their raw results are
  kept outside the repository, and the shell-bearing readback is
  `tests/artifact_fixtures/read_result.txt`.

## Archived probe source (research receipts)

`probe_exposure.py.txt`, `probe_override_list.py.txt`, `guard_receipt.py.txt`,
`hooks_settings_probe.py.txt`, `enforcement_matrix.py.txt`, `managed_effective.py.txt` and
`sanitise_evidence.py.txt` are the exact source of the one-off probes that
produced these files, archived as receipts. They are not repository code:
they are not maintained, linted or type-checked, and they call
`claude_agent_sdk` directly only because they were throwaway research
probes, which `AGENTS.md` reserves for `claude.py` in anything that ships.
Any shipped Artifact integration must go through `ClaudeAgent`. The `.txt`
suffix marks them as archived text; it is not a way around that rule.

To re-run one, copy it out of the repository (shape of the commands; `PY` is
a venv with `claude-agent-sdk`):

```text
env -u CLAUDECODE -u CLAUDE_CODE_CHILD_SESSION -u CLAUDE_CODE_SESSION_ID \
    -u CLAUDE_CODE_SESSION_ATTENDED -u CLAUDE_CODE_MESSAGING_SOCKET \
    -u CLAUDE_CODE_MESSAGING_TOKEN -u CLAUDE_CODE_EXECPATH -u CLAUDE_PID \
    -u CLAUDE_EFFORT  $PY -I probe_exposure.py LABEL OUTDIR SPEC_JSON
# A: '{"tools":["Read","Artifact"]}'
# B: '{"tools":["Read","Artifact"],"cli_path":"<installed claude>"}'
# C: '{"tools":"preset","setting_sources":["user"]}'
# F: $PY -I probe_override_list.py OUT.json        (the undocumented override)
python3 -I sanitise_evidence.py RAW_DIR OUT_DIR
```

Probe F's raw list result was inspected privately and is not stored. It
was an account artifact listing (header with total, 25 entries with kind,
title, URL and date, a note about a `limit` input), not operation help. Its
saved record keeps only `content_chars`.

Source reading (gate order, `isArtifactSdkDefaultOff`, tool operations) was
done on the bundled binary with plain text search; search for
`CLAUDE_CODE_ARTIFACT`, `sdk_default_off`, `isArtifactSdkDefaultOff` and
`publish_version`. Excerpts of the proprietary binary are not committed.
