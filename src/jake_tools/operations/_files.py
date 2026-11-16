"""Shared plumbing for the ``sharepoint_*`` operations.

The client is built per call over the *same* refresh-aware provider the
newsletter tools use (:func:`~jake_tools.microsoft_auth.shared_provider`), so
there is one Microsoft identity and one set of settings. Nothing here reads the
environment: tenant, application, account and secrets all come from the
resolved :class:`~jake_tools.config.Config`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import anyio.to_thread
from pydantic import BaseModel, Field

from ..config import Config
from ..file_transfers import (
    FileInputError,
    FileReadError,
    FileTransferError,
)
from ..microsoft_auth import MicrosoftAuthError, require_secure_url, shared_provider
from ..sharepoint_files import DriveClient, FilesError
from ._newsletter import REAUTHORISE, _auth_error, microsoft_settings
from .errors import ToolError

DriveId = Annotated[str, Field(pattern=r"^[A-Za-z0-9!_=-]{1,200}$")]
"""A Graph drive ID (``b!...``). It is interpolated into request paths."""
ItemId = Annotated[str, Field(pattern=r"^[A-Za-z0-9!_=-]{1,200}$")]
"""A drive item ID, as ``sharepoint_resolve``/``sharepoint_list`` return it."""
TeamId = Annotated[
    str, Field(pattern=r"^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
]
ChannelId = Annotated[str, Field(pattern=r"^[A-Za-z0-9:@._-]{1,200}$")]


def drive_client(config: Config) -> DriveClient:
    settings = microsoft_settings(config, purpose="SharePoint files")
    try:
        return DriveClient(
            token_provider=shared_provider(settings),
            graph_root=require_secure_url(
                config.files_graph_root.value, "files.graph_root"
            ),
        )
    except ValueError as exc:
        raise ToolError(
            "invalid_argument", f"Invalid files configuration: {exc}"
        ) from exc


def host_path(raw: str, argument: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ToolError(
            "invalid_argument",
            f"{argument} must be an absolute path on the jake-tools host: {path}",
            detail={"argument": argument},
        )
    return path


async def call_files[ResultT: BaseModel](
    config: Config,
    operation: Callable[[DriveClient], ResultT],
    *,
    writes: bool = False,
) -> ResultT:
    """Run ``operation`` in a worker thread and translate its failures.

    ``writes`` says the operation can change SharePoint, so an error that
    precedes the write states that nothing was changed.
    """
    client = drive_client(config)
    try:
        return await anyio.to_thread.run_sync(functools.partial(operation, client))
    except FileInputError as exc:
        detail: dict[str, object] = {"argument": exc.argument, "applied": False}
        if exc.reason:
            detail["reason"] = exc.reason
        detail.update(exc.extra)
        raise ToolError("invalid_argument", str(exc), detail=detail) from exc
    except FileReadError as exc:
        detail = {"reason": exc.reason, "applied": False}
        if exc.item is not None:
            detail["item"] = exc.item.model_dump(mode="json", exclude_none=True)
        raise ToolError("upstream_error", str(exc), detail=detail) from exc
    except FileTransferError as exc:
        raise _write_error(exc) from exc
    except MicrosoftAuthError as exc:
        error = _auth_error(exc)
        if writes:
            error.detail = {**(error.detail or {}), "applied": False}
        raise error from exc
    except FilesError as exc:
        raise _files_error(exc, applied=False if writes else None) from exc


def _files_error(exc: FilesError, *, applied: bool | None = None) -> ToolError:
    detail: dict[str, object] = {"integration": "microsoft", "http_status": exc.status}
    if exc.code:
        detail["error"] = exc.code
    if applied is not None:
        detail["applied"] = applied
    if exc.status == 401:
        detail["reason"] = "token_rejected"
        return ToolError(
            "access_denied",
            f"Microsoft rejected the access token ({exc}). The grant may have been "
            f"revoked; reauthorise (see {REAUTHORISE}) or retry shortly.",
            detail=detail,
        )
    if exc.status == 403:
        detail["reason"] = "insufficient_permission"
        return ToolError(
            "access_denied",
            f"The configured account lacks permission for this request ({exc}). "
            "Jake does not switch accounts or change grants; the account needs "
            "access to that site or folder, and the application needs the "
            "delegated Files.ReadWrite.All permission.",
            detail=detail,
        )
    if exc.status == 404:
        detail["reason"] = "not_found"
        return ToolError(
            "invalid_argument",
            f"Not found, or not visible to the configured account ({exc}). Check "
            "the link or IDs; a result's drive_id and item_id are the ones to use.",
            detail=detail,
        )
    if exc.status == 429:
        detail["reason"] = "throttled"
        return ToolError(
            "upstream_error",
            f"Microsoft is throttling requests ({exc}); wait before trying again.",
            detail=detail,
        )
    return ToolError(
        "upstream_error", f"SharePoint files request failed: {exc}", detail=detail
    )


def _write_error(exc: FileTransferError) -> ToolError:
    detail = exc.detail()
    detail["integration"] = "microsoft"
    cause = exc.__cause__
    if exc.outcome == "not_applied" and exc.reason == "name_exists":
        return ToolError("invalid_argument", str(exc), detail=detail)
    if isinstance(cause, FilesError) and cause.status in (401, 403):
        detail["reason"] = (
            "token_rejected" if cause.status == 401 else "insufficient_permission"
        )
        return ToolError("access_denied", str(exc), detail=detail)
    return ToolError("upstream_error", str(exc), detail=detail)
