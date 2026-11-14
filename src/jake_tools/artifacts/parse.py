"""Pure parsing of what Claude Code's ``Artifact`` tool reports.

Nothing here talks to a model or the network. The formats are what CLI
2.1.318 and 2.1.325 printed (see ``docs/claude-tools.md``); they are
Claude Code's wording, not a documented contract, so every function fails
closed: an unrecognised shape yields ``None``, never a guess. Callers turn
that into an unverified result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

URL_PATTERN = re.compile(
    r"^https://claude\.ai/(?:code/)?artifact/(?P<id>[A-Za-z0-9_-]+)$"
)
"""An artifact link: the short ID is the identity the publish tool keeps."""

_PUBLISHED = re.compile(
    r"Published (?P<path>.+?) at "
    r"(?P<url>https://claude\.ai/(?:code/)?artifact/[A-Za-z0-9_-]+) "
    r"\(Version (?P<version>\d+), version id (?P<version_id>[\w.-]+)\)"
)
_HEADER = re.compile(
    r"^\[Artifact (?P<artifact_id>[0-9a-f]{8}-[0-9a-f-]{27}) "
    r"\(version (?P<version_id>[\w.-]+)\) — (?P<rest>[^\n]*)\]$"
)
_OPEN = "<cowritten-artifact-html>\n"
_CLOSE = "\n</cowritten-artifact-html>"

SHELL_PREFIX = (
    "<!doctype html><html><head><meta charset=utf8>"
    '<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">'
    "<style>:root{color-scheme:light;box-sizing:border-box;"
    "padding-top:env(safe-area-inset-top,0px);"
    "padding-bottom:env(safe-area-inset-bottom,0px)}"
    "html{scroll-padding-top:env(safe-area-inset-top,0px)}"
    "body{margin:0;padding:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif;"
    "background:#fff;color:#000}img{max-width:100%}"
    "[hidden]:not([hidden=until-found i]){display:none!important}"
    "</style></head><body>\n"
)
SHELL_SUFFIX = "\n</body></html>"
"""The document shell Claude Code wraps around a published file, pinned.

The artifacts page says Claude Code "wraps the file you publish in an HTML
document shell". A hosted ``read`` returns that shell around the file: this
exact 531-byte prefix, then the file as published, then this suffix. They are
the bytes recorded in ``tests/artifact_fixtures/read_result.txt`` (CLI 2.1.318)
and are not documented or promised: Claude Code may change them. Only an exact
match is removed. Any other wrapper, or an artifact that declares runtime
capabilities, yields no content claim at all, never an approximate one."""


@dataclass(frozen=True)
class Published:
    path: str
    url: str
    version: int
    version_id: str


@dataclass(frozen=True)
class Header:
    """The first line of a hosted ``read``."""

    artifact_id: str
    version_id: str
    owned_by_you: bool
    visibility: str | None
    """The state the tool reports, e.g. ``private``; ``None`` when the line
    does not say (for example an artifact someone else owns)."""
    line: str


def artifact_short_id(url: str) -> str | None:
    match = URL_PATTERN.match(url)
    return match["id"] if match else None


def parse_published(text: str) -> Published | None:
    """The identity a successful ``publish`` call reported."""
    match = _PUBLISHED.search(text)
    if match is None:
        return None
    return Published(
        path=match["path"],
        url=match["url"],
        version=int(match["version"]),
        version_id=match["version_id"],
    )


def parse_header(text: str) -> Header | None:
    """The identity, version and visibility line of a hosted ``read``."""
    first = text.split("\n", 1)[0]
    match = _HEADER.match(first)
    if match is None:
        return None
    rest = match["rest"]
    owned = rest.startswith("owned by you, ")
    visibility: str | None = None
    if owned:
        visibility = re.split(r" \(|;", rest[len("owned by you, ") :], maxsplit=1)[
            0
        ].strip()
    return Header(
        artifact_id=match["artifact_id"],
        version_id=match["version_id"],
        owned_by_you=owned,
        visibility=visibility or None,
        line=first,
    )


def hosted_html(text: str) -> str | None:
    """The HTML between the ``cowritten-artifact-html`` tags of a ``read``.

    The file may itself contain the closing tag, so the last one ends it."""
    start = text.find(_OPEN)
    end = text.rfind(_CLOSE)
    if start < 0 or end < start + len(_OPEN):
        return None
    return text[start + len(_OPEN) : end]


def remove_shell(html: str) -> str | None:
    """The hosted file, with exactly the pinned shell removed; else ``None``.

    What remains is whatever sat between the pinned prefix and suffix. That is
    the published file only insofar as the service adds nothing else there."""
    if not (html.startswith(SHELL_PREFIX) and html.endswith(SHELL_SUFFIX)):
        return None
    if len(html) < len(SHELL_PREFIX) + len(SHELL_SUFFIX):
        return None
    return html[len(SHELL_PREFIX) : len(html) - len(SHELL_SUFFIX)]
