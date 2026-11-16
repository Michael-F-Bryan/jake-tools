---
name: newsletters
description: Read, create and update CSU Weekly Newsletter entries and attachments with verified readback; publication boundary and partial-failure recovery.
version: 0.1.0
author: Michael F. Bryan (Michael-F-Bryan), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
---

# CSU Weekly Newsletter entries

Use jake-tools to list and read newsletter list entries and to
create or update them with attachments. The tools are served by the
`jake-tools` MCP server; names below are bare names (an MCP client may prefix
them, as Hermes does with `mcp__<server>__`). Inspect the live input schema
before calling.

| Tool | Effect |
| --- | --- |
| `newsletter_list` | Read-only. One bounded page, newest first, with explicit coverage. |
| `newsletter_get` | Read-only. One entry with saved content and attachment names/URLs. |
| `newsletter_create` | **Publication.** Create one entry (+ attachments), then verify. |
| `newsletter_update` | **Publication.** Change title/body and/or add attachments, then verify. |

Create and update change a list its readers can see. Not
tools: composing or choosing content, sending the newsletter, deleting entries,
replacing or removing attachments, mail, OneNote, other SharePoint lists/files.

## Identity and authority

Every call acts as ONE Microsoft account that the operator configured on the
jake-tools host (`microsoft.account`, in one tenant, through one registered
application) using a delegated grant: it can do exactly what that person can do
in SharePoint, nothing the application alone could. You supply no credentials and
see none. Jake never falls back to another account, the Azure CLI's login, or a
cached browser session; a grant for a different tenant or account is refused.
Say which account acted only if the user asks: it is the configured one
(`doctor` prints `microsoft.account`).

## Publication boundary

An entry is visible to its readers when saved. Create or update only with
the title, body and files the user has approved in this conversation; drafting,
summarising or selecting content is not approval to publish. Show the user what will
be published (title, body, attachment names) and wait when there is any doubt.
Editorial choices stay with the user. Never use these tools to "test" a change on
the live list.

## Reading

- `newsletter_list` returns at most `limit` (1..100, default 10) entries.
  `complete=true` means the page reached the end of the list. Otherwise call
  again with the returned `next_cursor` and the SAME `limit`. A list is not a
  snapshot: entries created or removed while you page can shift it, so re-read
  rather than assume. Bodies are omitted unless `include_body=true`.
- `newsletter_get` returns the saved plain-text body and the attachments'
  names and URLs. SharePoint does not list sizes or digests, so they are null
  there; a create/update result carries them for the files it verified.
- Entry IDs are SharePoint list item numbers (digits only).

## Writing

`body` is plain text: blank lines separate paragraphs, a single newline is a
line break, HTML is escaped (not rendered). `attachments` are ABSOLUTE paths on
the machine running the jake-tools MCP server, up to 10
files of at most 25 MiB, with unique names; a name SharePoint cannot hold is
refused before anything is written. The tool reads those files with the server's
own permissions: attach only files the user named.

Create makes the entry once, adds each attachment in order, then reads back.
Update reads the entry first, writes only the supplied fields (an
attachments-only update sends no field write at all and verifies title and body
unchanged), adds the attachments, then reads back. Attachments are only ever
added; one already on the entry with identical bytes is left alone, and one
with the same name and different bytes is refused before any write.

**Success means `verified: true`.** `compared` lists exactly what the readback
checked: title, body, untouched fields unchanged, existing attachments still
present, and every supplied attachment downloaded and compared byte for byte.
`entry` is the saved entry as read back (id, URL, content, attachment metadata
with `size_bytes` and `sha256` for the verified files). Do not report a
publication done on a POST alone.

## When it did not complete

A failure carries `detail`: `outcome`, `entry_id`, `entry_url`, `effects` (one
per step: `kind`, `subject`, `status`, `verified`), `mismatches`, `retry_safe`
and `next_step`. Statuses: `applied`; `unchanged` (already identical);
`failed` (rejected, it did not happen); `uncertain` (no answer, it may have
happened); `not_attempted`. Nothing is ever retried for you.

| `outcome` | Meaning | Do |
| --- | --- | --- |
| `not_applied` | The first write was rejected. Nothing changed. | Fix the cause (`retry_safe` is true) and call again. |
| `create_uncertain` | The create got no answer; the entry may exist and its ID is unknown. | **Never create again.** `newsletter_list` and look for the title, then continue with `newsletter_update` on that ID. |
| `partial` | The entry exists; some steps failed or are uncertain. | **Never create again.** Call `newsletter_update` with the same `entry_id` and the attachments still missing (or all of them: identical ones are skipped). |
| `verification_failed` | Written, but the readback differs (`mismatches`). | `newsletter_get` and compare; tell the user. Do not repeat the write blindly. |
| `verification_unavailable` | Written, but could not be read back. | `newsletter_get` to check before calling it done. |

An entry that exists is never "rolled back" by these tools; deleting it is a
manual SharePoint action for the user.

## Errors

- `missing_credentials`: Microsoft is not configured on the host. `detail`
  names the variables/settings. Tell the user; do not look for credentials.
- `access_denied` with `detail.reason`:
  `grant_invalid` (the refresh token expired or was revoked),
  `interaction_required` (a new sign-in or consent is needed),
  `client_rejected` (application credentials rejected),
  `resource_not_granted` (no consent for Graph or SharePoint),
  `identity_mismatch` / `identity_unverified` (the grant is not for the
  configured tenant/account; nothing was used),
  `token_rejected` (HTTP 401), `insufficient_permission` (HTTP 403: the account
  lacks the permission). `detail.action` says what the operator must do. These
  need a person with the right authority; retrying or switching accounts will
  not help, and you must not try to.
- `upstream_error` with `reason: unavailable`, or a failed request: the service
  could not be reached or failed. Reads can be repeated; writes follow the
  outcome table above.
- `invalid_argument`: nothing was changed (bad ID, a name conflict, an
  unusable file, a missing entry).

## Limits (honest)

- Verified against a local stand-in for Microsoft's endpoints. How live
  SharePoint normalises rich text, its throttling and tenant policies were not
  checked here; a real normalisation difference shows up as
  `verification_failed` rather than a silent pass.
- A field update is not a compare-and-swap: a concurrent edit to the entry can
  interleave (an untouched field that changes is caught by the readback).
- Paging is not an atomic snapshot.
