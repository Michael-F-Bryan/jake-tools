---
name: artifact-publishing
description: Publish an existing self-contained HTML file as a NEW private Claude Artifact with `artifact_publish`, read an artifact back read-only with `artifact_read`, and publish its next version with `artifact_update` (private, expected version required); what a publish and a readback do and do not prove, what visibility they report, the undocumented mechanism behind them, and how to handle each error.
---

# Artifact publishing

`artifact_publish(file_path)` publishes an existing `.html`/`.htm` file (an absolute path as
jake-tools sees it, UTF-8, at most 1 MB) as a **new** artifact on claude.ai through the signed-in
Claude account, then reads it back in a separate session. `artifact_read(url, ...)` reads one
artifact back and changes nothing. The tools exist only when `artifacts.enabled` is set
(`JAKE_TOOLS_ARTIFACTS_ENABLED`).

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
before every tool call by a `PreToolUse` hook that judges the raw model input: a publish session
may `Read` the one file (its exact canonical path) and make one `publish` of it; a read session may
make one `read` of the one URL. `list`, `share`, `delete`, updates (`url`), `edits`, any extra input
key, other files or URLs, a second publish and every other tool are denied before they run. Calls
the guard refused are listed in `denied_before_running`; they had no effect. A managed (policy)
settings file that disabled the hook was not tested, and a Claude Code upgrade needs a recheck; a
call that escaped the hook is only detected afterwards (the result is unverified), not prevented. See
`docs/claude-tools.md`.

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

## Updating an existing artifact

`artifact_update(url, file_path, expected_version_id)` publishes the file as the **next version of
an existing private artifact you own**, same URL and id. It is a separate tool
because it overwrites live content.

- Pass the version you believe is live (`version_id` from `artifact_read`/`artifact_publish`/a
  previous update). A separate read checks, **before anything is written**, that the artifact is
  yours, private and at that version. A different live version fails with `plan_stale` (detail:
  `live_version_id`, `write_attempted: false`); read it, reconcile, and call again with the live
  version. A shared artifact or one you do not own fails with `access_denied`; updates are
  private-only.
- The write session can publish only after its own read of the target was **observed** as yours,
  private and at your expected version; otherwise the publish is denied before it runs (nothing
  changed, a retry is safe).
- It is **not atomic**. The native read and write are separate calls, and an edit that lands between
  that last observed read and the publish cannot be detected by Jake. Claude Code's own refusal of a
  stale copy is the only protection there, and it is untested. Do not describe it as compare-and-swap.
- `verified: true` needs a third, separate read to show the same artifact id, unchanged visibility
  (private), the published version, and the hosted file (shell removed) equal to your file. The
  result's `before` is the pre-write read and `readback` the post-write read.
- Never retried. After an `UNKNOWN` error, or any `verified: false`, read the artifact with
  `artifact_read` and decide from its live version before doing anything; a blind repeat could write
  another version.

## What it does not do

No sharing or visibility changes, no listing, no deleting, no partial edits (an update replaces the
whole file), no updating of shared artifacts, no packaging of dependencies (publish a
self-contained file), no authoring of HTML. Use it for a file you are willing to send to Anthropic;
its contents leave this machine.
