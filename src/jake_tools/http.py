from __future__ import annotations

from typing import Any, Protocol


class HttpSession(Protocol):
    """The subset of requests.Session used by our HTTP-backed clients.

    Shared by clockify.py and newsletters.py so both accept the same fake
    session in tests instead of each declaring an identical Protocol.
    """

    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...
