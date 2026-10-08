from __future__ import annotations

from typing import Any, Protocol


class HttpSession(Protocol):
    """The subset of requests.Session used by our HTTP-backed clients.

    Shared by clockify.py and newsletters.py so both accept the same fake
    session in tests instead of each declaring an identical Protocol.
    """

    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


class UpstreamError(RuntimeError):
    """An error from an HTTP-backed client, with a host-free summary.

    ``str(error)`` is the full diagnostic for a person at their own terminal:
    it may include the request URL's host (from a ``requests`` exception) or
    a slice of the response body. ``safe_summary``, when set, says the same
    thing without any of that: the operation (method and path, never the
    host), plus the HTTP status or the exception class name. Callers that
    report errors to someone else (the MCP server) use it instead of the
    message. ``None`` means the message is the client's own text and carries
    no upstream content.
    """

    def __init__(self, message: str, *, safe_summary: str | None = None) -> None:
        super().__init__(message)
        self.safe_summary = safe_summary
