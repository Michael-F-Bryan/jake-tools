"""Server-configured, refresh-aware delegated Microsoft access tokens.

One small component for the one consumer that exists (the newsletter list):
exchange an operator-provisioned *refresh token* for short-lived access tokens,
one per resource audience (Microsoft Graph, SharePoint), and keep each only
until shortly before it expires.

What it is not: an OAuth gateway, an interactive sign-in, or a provider
framework. It never obtains a grant; an operator provisions one (see
``docs/microsoft-auth.md``). It makes the grant's authority explicit:

* the tenant, client application and the *account* the grant must belong to
  are configuration; a token response for another tenant or account is refused
  (``identity_mismatch``) rather than used;
* a revoked or expired grant, a missing consent, a rejected client and an
  unreachable authority are distinct, actionable :class:`MicrosoftAuthError`
  kinds, none of which carries a token or secret.

Library choice: MSAL Python documents ``acquire_token_by_refresh_token`` as a
migration aid, and ``azure-identity`` has no refresh-token credential, so the
supported route would be MSAL's own interactive/cache machinery, which this
server cannot drive. The refresh-token grant is a single documented POST
(https://learn.microsoft.com/entra/identity-platform/v2-oauth2-auth-code-flow),
so it is made directly with ``requests`` and adds no dependency.

Rotated refresh tokens are kept in memory only (Microsoft asks clients to
replace the old one; it does not revoke it when used). A restart goes back to
the provisioned token. Nothing is written to disk.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import requests

from .http import HttpSession, UpstreamError

DEFAULT_AUTHORITY_URL = "https://login.microsoftonline.com"
GUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
OIDC_SCOPES = "openid profile offline_access"
REFRESH_MARGIN_SECONDS = 300.0
"""Refresh this long before expiry (or a fifth of the lifetime, if shorter)."""
REQUEST_TIMEOUT_SECONDS = 30

AuthErrorKind = Literal[
    "grant_invalid",
    "interaction_required",
    "client_rejected",
    "resource_not_granted",
    "identity_mismatch",
    "identity_unverified",
    "unavailable",
    "rejected",
]
"""Why a token could not be had; each has a different next step (see
:meth:`MicrosoftAuthError.action`)."""

_ACTIONS: dict[AuthErrorKind, str] = {
    "grant_invalid": (
        "The refresh token is expired, revoked or no longer valid (password change, "
        "sign-in policy, inactivity). Reauthorise the account and replace "
        "MICROSOFT_REFRESH_TOKEN; see docs/microsoft-auth.md."
    ),
    "interaction_required": (
        "Microsoft requires a new interactive sign-in or consent for this account. "
        "Reauthorise it and replace MICROSOFT_REFRESH_TOKEN."
    ),
    "client_rejected": (
        "Microsoft rejected the application credentials. Check microsoft.client_id "
        "and MICROSOFT_CLIENT_SECRET (a public client has no secret; a confidential "
        "client needs a current one)."
    ),
    "resource_not_granted": (
        "The application has no delegated consent for this resource. A tenant "
        "administrator must grant it; Jake never changes grants."
    ),
    "identity_mismatch": (
        "The grant belongs to a different tenant or account than configured. No "
        "token was used. Provision a grant for the configured account, or correct "
        "microsoft.tenant_id / microsoft.account if the configuration is wrong."
    ),
    "identity_unverified": (
        "The token response did not say which account it belongs to, so it was not "
        "used. Reauthorise requesting the scopes `openid profile offline_access`."
    ),
    "unavailable": "The Microsoft sign-in service could not be reached; retry later.",
    "rejected": "Microsoft refused the token request; see the error code.",
}


class MicrosoftAuthError(UpstreamError):
    """A token could not be obtained. Never carries a token or secret."""

    def __init__(
        self,
        kind: AuthErrorKind,
        resource: str,
        *,
        error: str | None = None,
        aadsts: str | None = None,
        status: int | None = None,
    ) -> None:
        self.kind: AuthErrorKind = kind
        self.resource = resource
        self.error = error
        self.aadsts = aadsts
        self.status = status
        facts = ", ".join(
            part
            for part in (
                f"error={error}" if error else None,
                aadsts,
                f"http {status}" if status else None,
            )
            if part
        )
        message = (
            f"Microsoft token request for {resource} failed ({kind}"
            f"{': ' + facts if facts else ''}). {_ACTIONS[kind]}"
        )
        super().__init__(message, safe_summary=message)

    @property
    def action(self) -> str:
        return _ACTIONS[self.kind]


def require_secure_url(url: str, setting: str) -> str:
    """``url`` unchanged if it is HTTPS (or plain HTTP to loopback), else ValueError.

    Bearer tokens and the refresh token travel to these endpoints, so a
    misconfigured cleartext remote is refused instead of used.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return url
    if parts.scheme == "http" and parts.hostname in LOOPBACK_HOSTS:
        return url
    raise ValueError(f"{setting} must be an https URL (or http to localhost)")


@dataclass(frozen=True)
class MicrosoftSettings:
    """Everything that selects the grant. Hashable: it keys the shared provider."""

    tenant_id: str
    client_id: str
    account: str
    refresh_token: str = field(repr=False)
    client_secret: str | None = field(default=None, repr=False)
    authority_url: str = DEFAULT_AUTHORITY_URL

    def __post_init__(self) -> None:
        for name in ("tenant_id", "client_id"):
            if not GUID_PATTERN.match(getattr(self, name)):
                raise ValueError(f"microsoft.{name} must be a GUID")
        if "@" not in self.account:
            raise ValueError("microsoft.account must be the account's sign-in name")
        require_secure_url(self.authority_url, "microsoft.authority_url")

    @property
    def token_url(self) -> str:
        tenant = urllib.parse.quote(self.tenant_id, safe="")
        return f"{self.authority_url.rstrip('/')}/{tenant}/oauth2/v2.0/token"


@dataclass(frozen=True)
class _CachedToken:
    value: str = field(repr=False)
    refresh_at: float


class MicrosoftTokenProvider:
    """``provider(resource) -> access token``, safe to call from many threads.

    ``resource`` is an audience such as ``https://graph.microsoft.com`` or
    ``https://<tenant>.sharepoint.com``; each gets its own cached token
    (a Graph token is not accepted by SharePoint REST). Concurrent callers
    needing a fresh token wait on one lock, and only the first performs the
    refresh: the rest find the new token in the cache.
    """

    def __init__(
        self,
        settings: MicrosoftSettings,
        *,
        session: HttpSession | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._session = session or requests.Session()
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens: dict[str, _CachedToken] = {}
        self._refresh_token = settings.refresh_token

    def __call__(self, resource: str) -> str:
        cached = self._tokens.get(resource)
        if cached is not None and self._clock() < cached.refresh_at:
            return cached.value
        with self._lock:
            cached = self._tokens.get(resource)
            if cached is not None and self._clock() < cached.refresh_at:
                return cached.value
            fresh = self._refresh(resource)
            self._tokens[resource] = fresh
            return fresh.value

    def _refresh(self, resource: str) -> _CachedToken:
        form = {
            "grant_type": "refresh_token",
            "client_id": self._settings.client_id,
            "refresh_token": self._refresh_token,
            "scope": f"{resource}/.default {OIDC_SCOPES}",
        }
        if self._settings.client_secret:
            form["client_secret"] = self._settings.client_secret
        try:
            response = self._session.request(
                "POST",
                self._settings.token_url,
                data=form,
                headers={"Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            # The exception text can carry the URL; the class name is enough.
            raise MicrosoftAuthError("unavailable", resource) from exc
        body = _json_object(response)
        if response.status_code >= 400:
            raise _failure(resource, response.status_code, body)
        access_token = body.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise MicrosoftAuthError("rejected", resource, status=response.status_code)
        self._check_identity(resource, body.get("id_token"))
        rotated = body.get("refresh_token")
        if isinstance(rotated, str) and rotated:
            self._refresh_token = rotated
        lifetime = _lifetime(body.get("expires_in"))
        margin = min(REFRESH_MARGIN_SECONDS, lifetime / 5)
        return _CachedToken(access_token, self._clock() + lifetime - margin)

    def _check_identity(self, resource: str, id_token: object) -> None:
        """Refuse a grant for another tenant or account.

        The ID token comes straight from the token endpoint over TLS and is
        read, not signature-checked: this guards against a mis-provisioned
        grant (wrong person signed in), it is not an authorisation boundary.
        """
        claims = _claims(id_token)
        if claims is None:
            raise MicrosoftAuthError("identity_unverified", resource)
        tenant = claims.get("tid")
        account = next(
            (
                claims[key]
                for key in ("preferred_username", "upn", "email")
                if isinstance(claims.get(key), str)
            ),
            None,
        )
        if not isinstance(tenant, str) or account is None:
            raise MicrosoftAuthError("identity_unverified", resource)
        if (
            tenant.lower() != self._settings.tenant_id.lower()
            or account.casefold() != self._settings.account.casefold()
        ):
            raise MicrosoftAuthError("identity_mismatch", resource)


_providers: dict[MicrosoftSettings, MicrosoftTokenProvider] = {}
_providers_lock = threading.Lock()


def shared_provider(settings: MicrosoftSettings) -> MicrosoftTokenProvider:
    """The one provider for this exact grant, so tokens outlive a single call.

    Operations build a client per call; a per-call provider would refresh every
    time. Equal settings (including the secrets) share a provider, and
    different grants (another Hermes profile) never do.
    """
    with _providers_lock:
        provider = _providers.get(settings)
        if provider is None:
            provider = _providers[settings] = MicrosoftTokenProvider(settings)
        return provider


def _json_object(response: Any) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _lifetime(value: object) -> float:
    try:
        seconds = float(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return 0.0
    return max(seconds, 0.0)


def _claims(id_token: object) -> dict[str, Any] | None:
    if not isinstance(id_token, str):
        return None
    parts = id_token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(payload))
    except binascii.Error, ValueError:
        return None
    return data if isinstance(data, dict) else None


def _failure(resource: str, status: int, body: dict[str, Any]) -> MicrosoftAuthError:
    raw_error = body.get("error")
    # A short code only: this text reaches callers, so never echo free text.
    error = (
        raw_error
        if isinstance(raw_error, str) and re.fullmatch(r"[a-z_]{1,40}", raw_error)
        else None
    )
    description = body.get("error_description")
    match = re.search(r"AADSTS\d+", description if isinstance(description, str) else "")
    kind: AuthErrorKind
    if error == "invalid_grant":
        kind = "grant_invalid"
    elif error in ("interaction_required", "consent_required", "login_required"):
        kind = "interaction_required"
    elif error in ("invalid_client", "unauthorized_client"):
        kind = "client_rejected"
    elif error in ("invalid_scope", "invalid_resource"):
        kind = "resource_not_granted"
    elif status >= 500 or error in ("temporarily_unavailable", "server_error"):
        kind = "unavailable"
    else:
        kind = "rejected"
    return MicrosoftAuthError(
        kind,
        resource,
        error=error,
        aadsts=match.group(0) if match else None,
        status=status,
    )
