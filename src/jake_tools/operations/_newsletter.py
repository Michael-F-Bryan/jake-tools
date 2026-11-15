"""Newsletter-only configuration and safe domain-error translation.

The client is built per call, but the token provider is shared
(:func:`~jake_tools.microsoft_auth.shared_provider`) so a token outlives the
call that fetched it. Nothing here reads the environment: the account, tenant,
application and secrets all come from the resolved :class:`Config`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import anyio.to_thread
from pydantic import BaseModel, Field

from ..config import Config
from ..microsoft_auth import (
    MicrosoftAuthError,
    MicrosoftSettings,
    require_secure_url,
    shared_provider,
)
from ..newsletter_entries import (
    NewsletterInputError,
    NewsletterWriteError,
    PreparedAttachment,
    prepare_attachments,
)
from ..newsletters import NewsletterClient, NewsletterError
from .errors import ToolError

EntryId = Annotated[str, Field(pattern=r"^[0-9]{1,10}$")]
"""A SharePoint list item ID. It is interpolated into request paths, so digits only."""

REAUTHORISE = "docs/microsoft-auth.md"


def microsoft_settings(config: Config) -> MicrosoftSettings:
    required = (
        (
            "microsoft.tenant_id",
            "JAKE_TOOLS_MICROSOFT_TENANT_ID",
            config.microsoft_tenant_id,
        ),
        (
            "microsoft.client_id",
            "JAKE_TOOLS_MICROSOFT_CLIENT_ID",
            config.microsoft_client_id,
        ),
        ("microsoft.account", "JAKE_TOOLS_MICROSOFT_ACCOUNT", config.microsoft_account),
        (
            "MICROSOFT_REFRESH_TOKEN",
            "MICROSOFT_REFRESH_TOKEN",
            config.microsoft_refresh_token,
        ),
    )
    missing = [(name, env) for name, env, setting in required if not setting.value]
    if missing:
        raise ToolError(
            "missing_credentials",
            "Microsoft is not configured for the newsletter. Set "
            + ", ".join(env for _name, env in missing)
            + f" where jake-tools runs (see {REAUTHORISE}). No other account is used.",
            detail={
                "variables": [env for name, env in missing if name.isupper()],
                "settings": [name for name, _env in missing if not name.isupper()],
            },
        )
    try:
        return MicrosoftSettings(
            tenant_id=str(config.microsoft_tenant_id.value),
            client_id=str(config.microsoft_client_id.value),
            account=str(config.microsoft_account.value),
            refresh_token=str(config.microsoft_refresh_token.value),
            client_secret=config.microsoft_client_secret.value or None,
            authority_url=config.microsoft_authority_url.value,
            account_object_id=config.microsoft_account_object_id.value or None,
        )
    except ValueError as exc:
        raise ToolError(
            "invalid_argument", f"Invalid Microsoft configuration: {exc}"
        ) from exc


def newsletter_client(config: Config) -> NewsletterClient:
    settings = microsoft_settings(config)
    try:
        return NewsletterClient(
            token_provider=shared_provider(settings),
            graph_root=require_secure_url(
                config.newsletters_graph_root.value, "newsletters.graph_root"
            ),
            sharepoint_root=require_secure_url(
                config.newsletters_sharepoint_root.value, "newsletters.sharepoint_root"
            ),
        )
    except ValueError as exc:
        raise ToolError(
            "invalid_argument", f"Invalid newsletter configuration: {exc}"
        ) from exc


def attachment_paths(raw: Sequence[str]) -> list[Path]:
    paths = [Path(item).expanduser() for item in raw]
    for path in paths:
        if not path.is_absolute():
            raise ToolError(
                "invalid_argument",
                f"attachments must be absolute paths on the jake-tools host: {path}",
                detail={"argument": "attachments"},
            )
    return paths


def prepare(raw: Sequence[str]) -> Callable[[], list[PreparedAttachment]]:
    """Read the attachments (blocking) when called inside the worker thread."""
    paths = attachment_paths(raw)
    return functools.partial(prepare_attachments, paths)


async def call_newsletter[ResultT: BaseModel](
    config: Config,
    operation: Callable[[NewsletterClient], ResultT],
    *,
    entry_id: str | None = None,
) -> ResultT:
    client = newsletter_client(config)
    try:
        return await anyio.to_thread.run_sync(functools.partial(operation, client))
    except NewsletterInputError as exc:
        raise ToolError(
            "invalid_argument",
            str(exc),
            detail={"argument": exc.argument, "applied": False},
        ) from exc
    except NewsletterWriteError as exc:
        raise _write_error(exc) from exc
    except MicrosoftAuthError as exc:
        raise _auth_error(exc) from exc
    except NewsletterError as exc:
        raise _upstream_error(exc, entry_id) from exc


def _auth_error(exc: MicrosoftAuthError) -> ToolError:
    detail: dict[str, object] = {
        "integration": "microsoft",
        "reason": exc.kind,
        "resource": exc.resource,
        "action": exc.action,
    }
    if exc.error:
        detail["error"] = exc.error
    if exc.aadsts:
        detail["aadsts"] = exc.aadsts
    if exc.kind == "unavailable":
        return ToolError("upstream_error", str(exc), detail=detail)
    return ToolError("access_denied", str(exc), detail=detail)


def _upstream_error(exc: NewsletterError, entry_id: str | None) -> ToolError:
    summary = exc.safe_summary or "newsletter request failed"
    detail: dict[str, object] = {"integration": "microsoft", "http_status": exc.status}
    if exc.status == 401:
        detail["reason"] = "token_rejected"
        return ToolError(
            "access_denied",
            f"Microsoft rejected the access token ({summary}). The grant may have "
            f"been revoked; reauthorise (see {REAUTHORISE}) or retry shortly.",
            detail=detail,
        )
    if exc.status == 403:
        detail["reason"] = "insufficient_permission"
        return ToolError(
            "access_denied",
            f"The configured account lacks permission for this request ({summary}). "
            "Jake does not switch accounts or change grants; ask a tenant "
            "administrator for the delegated permission or site access.",
            detail=detail,
        )
    if exc.status == 404 and entry_id is not None:
        detail["argument"] = "entry_id"
        return ToolError(
            "invalid_argument",
            f"Newsletter entry {entry_id} was not found in the list ({summary}).",
            detail=detail,
        )
    return ToolError(
        "upstream_error", f"Newsletter request failed: {summary}", detail=detail
    )


def _write_error(exc: NewsletterWriteError) -> ToolError:
    detail = exc.detail()
    detail["integration"] = "microsoft"
    cause = exc.__cause__
    code = "upstream_error"
    if isinstance(cause, MicrosoftAuthError):
        detail["reason"] = cause.kind
        detail["action"] = cause.action
        code = "upstream_error" if cause.kind == "unavailable" else "access_denied"
    elif isinstance(cause, NewsletterError) and cause.status in (401, 403):
        detail["reason"] = (
            "token_rejected" if cause.status == 401 else "insufficient_permission"
        )
        code = "access_denied"
    return ToolError(code, str(exc), detail=detail)  # type: ignore[arg-type]
