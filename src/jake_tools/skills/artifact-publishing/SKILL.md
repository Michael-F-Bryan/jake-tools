---
name: artifact-publishing
description: Publish an existing self-contained HTML file as a NEW private Claude Artifact with `artifact_publish`, and read an artifact back read-only with `artifact_read`; what a publish and a readback do and do not prove, what visibility they report, the undocumented mechanism behind them, and how to handle each error.
---

# Artifact publishing

`artifact_publish(file_path)` publishes an existing `.html`/`.htm` file (an absolute path as
jake-tools sees it, UTF-8, at most 1 MB) as a **new** artifact on claude.ai through the signed-in
Claude account, then reads it back in a separate session. `artifact_read(url, ...)` reads one
artifact back and changes nothing. The tools exist only when `artifacts.enabled` is set
(`JAKE_TOOLS_ARTIFACTS_ENABLED`) and, under Hermes, the `artifact` capability is granted.

## What it is built on (read this first)

Claude Code withholds its `Artifact` tool from Agent SDK sessions unless the **undocumented**
environment variable `CLAUDE_CODE_ARTIFACT` is set. Jake sets it for these sessions only. It is not
a documented or supported Anthropic option and may change without notice; Michael accepted it as
experimental. If Claude Code stops exposing the tool, the call fails with `upstream_error`
(`reason: tool_not_exposed`) and publishes nothing. `enableArtifact: false`, an `Artifact` deny rule,
or an account/organisation policy that disallows artifacts has the same effect: Jake does not
override your Claude settings.

## Authority is enforced before each call

A model can be steered by what it reads, so a prompt is not a permission. Each session is guarded
before every tool call: a publish session may `Read` the one file and make one `publish` of it; a
read session may make one `read` of the one URL. `list`, `share`, `delete`, updates (`url`), `edits`,
other files or URLs, a second publish and every other tool are denied before they run. Calls the
guard refused are listed in `denied_before_running`; they had no effect.

## What a result tells you

`artifact_publish` returns `url` (the stable link), `artifact_id`, `version`/`version_id`, the
`source` receipt (path, bytes, sha256 of your file when the call started) and a `readback`.

- `verified: true` means a **separate** session read the artifact back, the readback reported the
  same artifact and version, reported it as `private`, and the hosted file's bytes equal your file's
  bytes. The comparison is done in code on the tool's own result, not on anything the model said.
- **The whole hosted response is not your file.** Claude Code wraps a published file in a fixed
  HTML document shell, and the readback sees that shell. Jake removes exactly that pinned shell
  (recorded from Claude Code 2.1.318) and compares what is left. If the shell is different, the
  readback says `shell_removed: false`, no content claim is made and `verified` is false.
- `visibility` in the readback is what the service reported **now** (`private`). Anthropic's
  documentation says a new artifact is visible only to you; the readback is the evidence for this
  artifact, and `private_confirmed` is true only when it said `private`. Jake never shares an
  artifact or changes sharing; that stays manual from the page's Share menu.
- `verified: false` with `problems` still means the artifact **exists** at `url`. Read `problems`
  (hosted bytes differ, version differs, not private, shell not recognised, readback failed, the
  session ended abnormally, the local file changed or vanished) before deciding anything.

`artifact_read` returns the same readback fields. Pass `file_path` for `matches_file`,
`expected_version_id` for `version_matches` (false when the live version is different or could not
be seen), and `output_path` to save the hosted file, shell removed, to a new path.

## Errors and retries

- Never retry a publish blindly: each call creates a new artifact. An error with
  `retry_safe: false` means a publish may have happened. Check `urls_seen` and
  claude.ai/code/artifacts first. `retry_safe: true` means every call was denied before it ran.
- `invalid_argument`: the path is not absolute, not `.html`/`.htm`, empty, over 1 MB or not
  valid UTF-8 (Claude Code also refuses U+FFFD), or the URL is not an artifact link.
- `upstream_error` with `publish_attempted`/`publish_allowed`/`retry_safe`/`urls_seen`: the
  Artifact tool returned no published URL, the session failed, or more than one publish happened.

## What it does not do

No updates, no listing, no sharing or visibility changes, no deleting, no packaging of
dependencies (publish a self-contained file), no authoring of HTML. Updates need the current
version viewed first and a conflict policy, and are a later, separate slice. Use it for a file you
are willing to send to Anthropic; its contents leave this machine.
