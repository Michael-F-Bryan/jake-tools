# Microsoft authentication for jake-tools

Jake's newsletter tools (`newsletter_list`, `newsletter_get`,
`newsletter_create`, `newsletter_update`) act on SharePoint through Microsoft
Graph and SharePoint REST as **one configured person**, using a *delegated*
grant. This page is for the operator who provisions and renews it. It is not
a general sign-in service: Jake never obtains a grant itself.

## Identity and authority

| Question | Answer |
| --- | --- |
| Who acts? | The account in `microsoft.account` (its sign-in name, in `microsoft.tenant_id`), through the registered application `microsoft.client_id`. |
| With what authority? | The delegated permissions that application was consented for, limited by what that person can do in SharePoint. Not application-wide access. |
| Where is the grant kept? | In the operator-provisioned secret `MICROSOFT_REFRESH_TOKEN` (and `MICROSOFT_CLIENT_SECRET` if the application is a confidential client). Never in `config.toml`, tool arguments, results or logs. |
| Which resources? | Two token audiences: Microsoft Graph (`https://graph.microsoft.com`, list items) and SharePoint (`https://<tenant>.sharepoint.com`, attachments). Each is requested and cached separately; a Graph token is not accepted by SharePoint REST. |
| Which agents? | Every Hermes profile with the plugin enabled and every MCP client of a server with the configuration has all four tools (list/get read; create/update publish). |
| Fallback? | None. There is no Azure CLI fallback and no other account. The `jake-tools newsletter` command is separate: it uses the person's own `az login`. |

Jake never changes tenant grants, never consents on anyone's behalf and never
provisions or edits the application. The existing delegated application can be
reused if it is already consented for Graph and SharePoint; that is the
operator's decision, and its secrets are not read by Jake's tooling.

## Configuration

Non-secret (flag-free; environment or `$XDG_CONFIG_HOME/jake-tools/config.toml`):

```toml
[microsoft]
tenant_id = "0a3a5574-cfda-4314-952e-c0b3e1dcac6d"   # a GUID
client_id = "<application (client) id GUID>"
account   = "person@example.org"                       # the sign-in name the grant MUST belong to
account_object_id = "<the user's object id GUID>"      # optional, recommended: immutable pin (oid)
# authority_url = "https://login.microsoftonline.com"  # default
```

Environment equivalents: `JAKE_TOOLS_MICROSOFT_TENANT_ID`,
`JAKE_TOOLS_MICROSOFT_CLIENT_ID`, `JAKE_TOOLS_MICROSOFT_ACCOUNT`,
`JAKE_TOOLS_MICROSOFT_ACCOUNT_OBJECT_ID`,
`JAKE_TOOLS_MICROSOFT_AUTHORITY_URL`.

Secrets (environment only, each also as `<NAME>_FILE`; for Hermes, profile secrets):

- `MICROSOFT_REFRESH_TOKEN` — required.
- `MICROSOFT_CLIENT_SECRET` — only if the application is a confidential client.

`newsletters.graph_root` / `newsletters.sharepoint_root`
(`JAKE_TOOLS_NEWSLETTERS_GRAPH_ROOT`, `..._SHAREPOINT_ROOT`) default to the CSU
site and exist so tests can point at a local stand-in. Anything other than
HTTPS (or plain HTTP to `localhost`) is refused, because bearer tokens travel there.

Missing values are not a startup error: the tools are listed, and a call returns
`missing_credentials` naming the variables and settings. `jake-tools-mcp doctor`
shows each setting's source and whether a secret is set (never its value).

## Provisioning a grant

Use the sign-in flow your organisation already uses for this application,
signing in **as the configured account** (a device-code flow against the
tenant's `/oauth2/v2.0/devicecode` and `/token` endpoints is the usual route;
see Microsoft's
[device code flow](https://learn.microsoft.com/entra/identity-platform/v2-oauth2-device-code)).

Microsoft's rule is **one resource audience per token request**: the scopes of
a token request "must all be from a single resource, along with OIDC scopes",
and a refresh request spanning several resources gets a token for the first
one ([auth code flow](https://learn.microsoft.com/entra/identity-platform/v2-oauth2-auth-code-flow)).
So do not send Graph and SharePoint scopes in one request. Instead:

1. **Establish delegated consent for both resources first** (a tenant
   administrator's consent, or the account's own consent where policy allows,
   given for the application): Graph list access and SharePoint attachment
   access. Jake never grants it.
2. **Sign in once**, requesting `openid profile offline_access` plus the scope
   of *one* resource (for example `https://graph.microsoft.com/.default`). The
   refresh token that results is valid "for all permissions that your client has
   already received consent for", so it can be redeemed for the other resource.
3. Jake then **redeems that one refresh token per audience** on its own,
   requesting `<resource>/.default openid profile offline_access` for Graph and
   for `https://<tenant>.sharepoint.com` in separate requests (see "How refresh
   works"). Check both with the doctor command: it fails per audience if the
   second resource was never consented.

Then:

4. Write the refresh token to a file readable only by the service user
   (`umask 077`), never to shell history, a ticket, chat or the repository,
   and point `MICROSOFT_REFRESH_TOKEN_FILE` at it (or set the Hermes profile
   secret `MICROSOFT_REFRESH_TOKEN`).
5. Set `microsoft.tenant_id`, `microsoft.client_id`, `microsoft.account` (and, recommended, `microsoft.account_object_id`).
6. Restart the service so it reads them (Jake does not hot-reload; changing
   configuration or deploying is a separate, authorised step).
7. Call `newsletter_list` with `limit: 1`. Success proves the Graph audience;
   `newsletter_get` on an entry with an attachment proves the SharePoint one.

Jake did not run this flow while the feature was built; it is Microsoft's
documented protocol, summarised here, not a Jake-tested procedure.

## How refresh works

Access tokens last about an hour. Jake exchanges the refresh token for one per
resource (`grant_type=refresh_token`), keeps it until shortly before it expires
(the last fifth of its life, at most five minutes), and then refreshes again. A
single lock means concurrent callers trigger one refresh, not many. If
Microsoft returns a new refresh token Jake uses it from then on, **in memory
only**. Nothing is persisted, so a restart returns to the provisioned token;
Microsoft does not revoke the old one when it is used, but it can expire (for
example after a long inactivity) or be revoked.

### The identity check, and what it relies on

Every refresh response is checked against `microsoft.tenant_id` and
`microsoft.account` (and `microsoft.account_object_id` when set), read from the
response's `id_token`. What Microsoft's documentation does and does not
promise, and what Jake does about it:

- *The ID token is only present when `openid` was requested.* The
  [refresh response](https://learn.microsoft.com/entra/identity-platform/v2-oauth2-auth-code-flow)
  lists `id_token` as "Only provided if `openid` scope was requested", and calls
  it "unsigned", to be read but not relied on "for any authorization or security
  boundaries". Jake sends `scope=<resource>/.default openid profile
  offline_access` on every refresh, and a response without an identity is
  refused (`identity_unverified`), never assumed.
- *The requested scopes must be within the original grant.* The same page says a
  refresh request's scopes "must be equivalent to or a subset of the scopes
  requested in the original `authorization_code` request leg". So the
  **provisioning constraint** is: authorise with `openid profile
  offline_access` (plus the resources). Microsoft does not document whether a
  grant that lacked them is answered with an error (`invalid_scope`, reported
  as `resource_not_granted`, whose action text names this) or with no
  `id_token` (`identity_unverified`); Jake handles both and uses neither.
- *Which claims.* The
  [claims reference](https://learn.microsoft.com/entra/identity-platform/id-token-claims-reference)
  lists `tid` among the default claims; `preferred_username` and `oid` require
  the `profile` scope. `preferred_username` is "mutable ... can't be used to
  make authorization decisions"; `oid` is the "immutable identifier" of the
  user. Therefore `microsoft.account` (a sign-in name) is a mis-provisioning
  guard, and `microsoft.account_object_id` is the stable pin: set it. A renamed
  account fails loudly with `identity_mismatch` until the configuration is
  updated.

Not documented, therefore not assumed: that every refresh returns an
`id_token` with these claims at all times. If one does not, the tools stop with
`identity_unverified` rather than proceeding unverified. The test stand-in
emits these claims because the documentation says they exist; that is not
evidence of what Microsoft sends. In either case the check is a guard against
the wrong person's grant, not an authorisation boundary: what the account may
do is decided by SharePoint/Graph per request.

Requests are never redirected: the token POST (it carries the refresh token) and
every Graph/SharePoint call use `allow_redirects=False`, so a `307`/`308`
cannot silently replay a write or re-send a secret elsewhere. A redirect is
reported as an error (a write's outcome is then `uncertain`).

## Failures and what to do

| `detail.reason` | Cause | Operator action |
| --- | --- | --- |
| `grant_invalid` | Refresh token expired, revoked, or invalidated (password change, policy, inactivity). | Reauthorise (below). |
| `interaction_required` | Microsoft wants a new sign-in, MFA or consent. | Reauthorise interactively. |
| `client_rejected` | Wrong `client_id`, or a missing/expired `MICROSOFT_CLIENT_SECRET`. | Fix configuration or renew the secret. |
| `resource_not_granted` | Microsoft rejected the scope: no delegated consent for Graph/SharePoint, or the original sign-in lacked `openid profile offline_access`. | Reauthorise with those scopes; if consent is missing, a tenant administrator grants it. Jake will not. |
| `identity_mismatch` / `identity_unverified` | The grant is for another tenant/account/object id, or the response did not identify one. | Provision for the configured account (with `openid profile offline_access`), or correct the configuration. |
| `token_rejected` (HTTP 401) | The resource refused a token. | Retry shortly; reauthorise if it persists. |
| `insufficient_permission` (HTTP 403) | The account cannot do this in SharePoint. | Grant that person site/list access. Jake will not switch accounts. |
| `unavailable` | Sign-in service unreachable or 5xx. | Retry later. |

## Reauthorisation

Repeat provisioning as the same account and replace the secret's value (the
file's contents or the profile secret), then restart the service. No tool
argument ever carries it. If the account itself must change, change
`microsoft.account` **and** the grant together; a mismatch is refused, never
used.

## Verification boundary

The tests run the real clients and handlers against a local HTTP stand-in for
the token endpoint, Graph and SharePoint REST. That proves refresh, expiry,
concurrency, audience separation, readback and partial-outcome mechanics. It
does **not** prove live Microsoft compatibility (token shapes, SharePoint's
rich-text normalisation, throttling, tenant policy, consent state). A live check
needs separate authority: `newsletter_get`/`newsletter_list` against the real
list, then a write only on an entry the owner has approved, reading back
each result as the tools do.
