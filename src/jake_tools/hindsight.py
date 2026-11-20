"""The Hindsight memory a delegated worker shares with its parent agent.

Same server, same bank, handed over as one explicit MCP server: no Hindsight
plugin or hooks, and no ambient MCP configuration (the worker keeps
``strict_mcp_config`` and ``setting_sources=()``). Nothing is retained
automatically; a worker may call ``retain`` itself when ``access`` allows.

Observed on the pinned Hindsight 0.10.1 (live server and source):

- ``<api_url>/mcp/<bank_id>/`` is the single-bank route over streamable HTTP.
  The bank comes from the path only: no tool takes a ``bank_id``, a header
  cannot change it, and ``list_banks``/``create_bank`` are absent, so a model
  cannot name another bank. The multi-bank route (``/mcp/``) is never used.
- Tools are bare names (``recall``, ``reflect``, ``retain``), seen by Claude as
  ``mcp__hindsight__<name>``. The route also carries destructive tools
  (``delete_bank``, ``clear_memories``, ...): they are listed but never
  approved, so under ``dontAsk`` a call is refused.
- The pod's server is unauthenticated and loopback-only: no credential is
  involved. Adding one would need an environment-only secret, never a value in
  ``spec.json`` or an argument vector.
"""

from __future__ import annotations

import urllib.parse
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    StringConstraints,
)

SERVER_NAME = "hindsight"
"""The MCP server name a worker sees, so its tools are ``mcp__hindsight__*``."""

HindsightAccess = Literal["read", "readwrite"]

READ_TOOLS = ("recall", "reflect")  # both read-only on the server
WRITE_TOOLS = ("retain",)  # one explicit, asynchronous write, with ``readwrite``
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def check_api_url(value: str) -> str:
    """``value`` as a bare base URL without a trailing slash, else ``ValueError``.

    HTTPS or loopback HTTP, no credentials and no path or query: the bank is
    chosen by the route built from it, never by the endpoint.
    """
    parts = urllib.parse.urlsplit(value.strip())
    try:
        host, _ = parts.hostname, parts.port
    except ValueError as exc:
        raise ValueError("is not a valid URL") from exc
    if not (
        (parts.scheme == "https" and host)
        or (parts.scheme == "http" and host in LOOPBACK_HOSTS)
    ):
        raise ValueError("must be an https URL (or http to localhost)")
    if parts.username is not None or parts.password is not None:
        raise ValueError("must not contain credentials")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("must be a bare server address with no path or query")
    return f"{parts.scheme}://{parts.netloc}"


def _blank_is_unset(value: Any) -> Any:
    return None if isinstance(value, str) and not value.strip() else value


BANK_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
"""One plain URL path segment: it is quoted into the route, never interpreted."""
BankId = Annotated[str, StringConstraints(pattern=BANK_ID_PATTERN)]
ApiUrl = Annotated[str, AfterValidator(check_api_url)]
# For settings: an empty value is the same as unset.
OptionalApiUrl = Annotated[ApiUrl | None, BeforeValidator(_blank_is_unset)]
OptionalBankId = Annotated[BankId | None, BeforeValidator(_blank_is_unset)]


class HindsightMemory(BaseModel):
    """One worker's memory connection, as ``spec.json`` records it (no secret)."""

    model_config = ConfigDict(frozen=True)

    api_url: ApiUrl
    bank_id: BankId
    access: HindsightAccess = "readwrite"

    @property
    def mcp_url(self) -> str:
        return f"{self.api_url}/mcp/{urllib.parse.quote(self.bank_id, safe='')}/"

    @property
    def tool_names(self) -> tuple[str, ...]:
        names = READ_TOOLS + (WRITE_TOOLS if self.access == "readwrite" else ())
        return tuple(f"mcp__{SERVER_NAME}__{name}" for name in names)
