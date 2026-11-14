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

## Reproduce

Scripts are `.txt` so lint, pyright and the "only `claude.py` imports the
SDK" convention do not treat them as repository code. Copy them out, then
(shape of the commands; `PY` is a venv with `claude-agent-sdk`):

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

Source reading (gate order, `isArtifactSdkDefaultOff`, tool operations) was
done on the bundled binary with plain text search; search for
`CLAUDE_CODE_ARTIFACT`, `sdk_default_off`, `isArtifactSdkDefaultOff` and
`publish_version`. Excerpts of the proprietary binary are not committed.
