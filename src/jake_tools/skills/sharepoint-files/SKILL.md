---
name: sharepoint-files
description: Resolve SharePoint/Teams links and channels, list and download files, upload new files and conditionally replace existing ones with verified readback; version identity, conflict and uncertain-write handling.
version: 0.1.0
author: Michael F. Bryan (Michael-F-Bryan), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
---

# SharePoint and Teams files

Use jake-tools to move files between the jake-tools host and a SharePoint or
Teams document library: find a folder or file, list it, download it, upload a
new file, or replace an existing one only if nobody else has changed it. The
tools are served by the `jake-tools` MCP server; names below are bare names (an
MCP client may prefix them, as Hermes does with `mcp__<server>__`). Inspect the
live input schema before calling.

| Tool | Effect |
| --- | --- |
| `sharepoint_resolve` | Read-only. A SharePoint link, or a Teams `team_id` + `channel_id`, to a drive item. |
| `sharepoint_get` | Read-only. One item by `drive_id` + `item_id`, with its current version. |
| `sharepoint_list` | Read-only. One bounded page of a folder, with explicit coverage. |
| `sharepoint_download` | Read-only on SharePoint. Save a file's current version to a NEW local path. |
| `sharepoint_pdf` | Read-only on SharePoint. Save Microsoft's PDF rendering to a NEW local path. |
| `sharepoint_upload` | **Write.** Add a NEW file; never overwrites. Then verify. |
| `sharepoint_replace` | **Write.** Replace a file only if it is still at the `etag` you saw. Then verify. |

This transports files that something else made. Not tools: editing or
composing Word/Excel content, comments, deleting, moving or renaming, sharing
or permissions, versions/restore, other Microsoft services.

## Identity and authority

Every call acts as ONE Microsoft account the operator configured on the
jake-tools host (the same delegated grant the newsletter tools use; see
`docs/microsoft-auth.md`): it can do what that person can do, nothing more.
You supply no credentials and see none. There is no fallback to another account.
The application also needs the delegated Graph permission `Files.ReadWrite.All`
(read-only calls would work with less); without it calls answer `access_denied`.

## IDs, not names

Everything is addressed by `drive_id` and `item_id`, both returned by
`sharepoint_resolve`, `sharepoint_list` and `sharepoint_get`. Copy them
exactly. Start with `sharepoint_resolve`:

- `url`: a SharePoint link the user gave you, on a host the operator allowed
  (`files.sharepoint_hosts`; any other host is refused with
  `reason=host_not_allowed`). It is resolved with Microsoft's shares API, which is
  documented for *sharing links*; whether an ordinary browser address of a file
  or folder also resolves was not checked against a live tenant, so if it does
  not (`invalid_argument`, not found), ask for a sharing link or use the channel.
  Resolving never grants access: a link the account cannot already open is not
  found or `access_denied`.
- `team_id` + `channel_id` (both required): the channel's files folder. Take the
  IDs from the user; do not guess them.

The result is the item: `kind` is `file`, `folder` or `other` (a notebook or
shortcut: not supported). A link or channel usually gives a folder; pass its
`item_id` to `sharepoint_list`, then pick a file's `item_id` from the entries.

## Version identity

Every item carries `etag` (changes with ANY change, including properties),
`ctag` (changes only with content), `modified`, `modified_by`, `size_bytes` and
`web_url`. Quote `etag` exactly as shown, quotes and braces included.
`sharepoint_download` returns the version it actually read and accepts the save
only if the etag did not change while reading and the size matches. A result
describes one moment: re-read with `sharepoint_get` before acting on it.

## Listing

`sharepoint_list` returns at most `limit` (1..200, default 50) children.
`complete=true` means the page reached the end. Otherwise call again with the
same `drive_id`, `item_id`, `limit` and the returned `next_cursor`. Order is the
service's, and paging is not a snapshot: re-read rather than assume.

## Downloading and PDFs

`destination` is an absolute path on the machine running the MCP server and must
not exist; a local file is never overwritten and a failed download leaves no
file. At most 50 MiB. The result gives the saved `size_bytes` and `sha256` and
`compared`, what was checked.

`sharepoint_pdf` asks Microsoft for its own PDF rendering. Only the source types
Microsoft documents for PDF conversion are attempted; anything else (a PDF, an
image, a zip) fails with `invalid_argument`, `reason=unsupported_conversion` and
`supported_extensions`, before anything is requested. Microsoft can also refuse
a supported file (its documented limit is a 2 MB source); that is the same
reason with `http_status`. Do not offer to convert it another way: tell the user.
The rendering is checked to be a PDF and the source unchanged while rendering; its
layout and fidelity are Microsoft's and were not checked.

## Writing

Both write tools change files other people use. Write only what the user has
approved in this conversation: the file, the folder and, for a replacement, which
existing file. Drafting a file is not approval to publish it. Never "test" on a
real library. `source_path` is an ABSOLUTE path on the MCP server's host, 1 byte to
50 MiB; the server reads it with its own permissions, so name only files the user
named. Content is transported byte for byte.

**New file** (`sharepoint_upload`): `parent_id` is a folder's `item_id`; `name`
defaults to the source file's name (no `" * : < > ? / \ |`, no leading or trailing
space, no trailing dot). `on_conflict` is `fail` (default: if the name exists,
nothing is written) or `rename` (SharePoint keeps both; the result has
`renamed: true` and the real name). There is no overwrite option.

**Replacement** (`sharepoint_replace`): needs `expected_etag`, the `etag` of the
version you last read. The call first reads the file: at any other etag it fails
with `plan_stale` and writes nothing. It then sends the etag as `If-Match` when
the upload starts, and Microsoft answers 412 if it went stale in between (also
`plan_stale`). The error shows the file's current version (`current`/`item`,
`modified_by`). Do NOT just retry with the new etag: someone else's edits
are in that version. Download it, tell the user, and reconcile outside this tool
(the human or another tool merges); replace again only with the merged file and
the new etag. Honest limit: the etag is checked once, when the upload starts; an
edit landing between that and the commit is not detected, and the later readback
proves the bytes, not that nobody else wrote in that instant.

**Success means `verified: true`.** After the write the exact item is read back
and its bytes downloaded and compared with the source (`compared` lists what
was checked: content, size, kind, parent folder). `item` is the saved item with
its new `etag`; `etag_at_write` is the one the upload reported, and
`changed_since_write: true` means something touched the item after your upload
(the bytes still matched). Never report an upload done on the request alone.

## When a write did not complete

A write failure carries `detail`: `outcome`, `reason`, `retry_safe`,
`next_step`, and as known `item`, `observed`, `mismatches`. Nothing is ever
retried for you.

| `outcome` | Meaning | Do |
| --- | --- | --- |
| `not_applied` | Nothing was written. `reason`: `name_exists` (`invalid_argument`), `version_conflict` (`plan_stale`), `locked`, `rejected`, or none (start failed). | Fix the cause (`retry_safe` is true) and call again; for `version_conflict` follow the replacement rules above. |
| `write_uncertain` | The upload got no usable answer: it may or may not have been applied. `observed` is an immediate look (`found_identical`, `found_different`, `not_found`, `unreadable`). | **Do not repeat it blindly** (a repeat can make a second file or overwrite). `sharepoint_get` / `sharepoint_list` to see the real state, then tell the user. `found_identical` means the bytes are already there. |
| `verification_failed` | Written, but the readback differs (`mismatches`; `changed_since_write` says whether the item changed after the write). | Inspect with `sharepoint_get`/`sharepoint_download`; tell the user. Do not repeat the write blindly. |
| `verification_unavailable` | Written, but could not be read back. | `sharepoint_get`/`sharepoint_download` to check before calling it done. |

A file that was written is never rolled back by these tools; removing or
restoring one is a manual SharePoint action for the user.

## Errors

- `missing_credentials`: Microsoft is not configured on the host; `detail` names
  the variables/settings. Tell the user; do not look for credentials.
- `access_denied` with `detail.reason`: the grant problems the newsletter guide
  lists (`grant_invalid`, `interaction_required`, `client_rejected`,
  `resource_not_granted`, `identity_mismatch`, `identity_unverified`) with
  `detail.action`, or `token_rejected` (HTTP 401) / `insufficient_permission`
  (HTTP 403: the account cannot open or change that site, folder or file, or the
  application lacks `Files.ReadWrite.All`). These need a person with the right
  authority; do not retry or try other accounts.
- `invalid_argument` with `detail.argument`/`reason`: nothing was changed. Bad
  path, existing destination, wrong kind (`not_a_file`/`not_a_folder`),
  `too_large`, an unusable name, `name_exists`, `not_found` (not found or not
  visible to the account), `host_not_allowed`, `unsupported_conversion`.
- `plan_stale`: the file is not at the version you expected; nothing was written.
- `upstream_error`: Microsoft failed or could not be reached (`reason=throttled`
  for HTTP 429: wait). A read can be repeated; a write follows the outcome table.
  A read that could not be trusted says why (`changed_during_download`,
  `size_mismatch`, `not_a_pdf`) and saved nothing.

## Limits (honest)

- Verified only against a local stand-in for Microsoft's endpoints. Live
  behaviour (which links resolve, how SharePoint treats `If-Match` and upload
  sessions, throttling, tenant policy, private-channel sites, rendering
  fidelity) was not checked here; a real difference shows up as an error or
  `verification_failed`, never a silent pass.
- Single-request upload and download only, up to 50 MiB; no resumable or
  chunked transfer.
- A replacement is not atomic compare-and-swap (see Writing). Paging is not a
  snapshot.
