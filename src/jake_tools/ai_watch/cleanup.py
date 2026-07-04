from __future__ import annotations

import re

_CHROME_PATTERNS: list[re.Pattern[str]] = [
    # Line is just an image link (avatar, logo, hero image, etc.)
    re.compile(r"^\s*\[!\[.*?\]\(.*?\)\]\(.*?\)\s*$"),
    re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$"),
    # Line is just a number or number sequence (like/comment counts)
    re.compile(r"^\s*\d+(?:\.\d+[kK])?\s*$"),
    re.compile(r"^\s*\d+\s*$"),
    # Single-word social/UI tokens
    re.compile(
        r"^\s*(?:Share|Subscribe|Sign\s*in|See\s*all|PreviousNext|CommentsRestacks|TopLatestDiscussions)\s*$",
        re.IGNORECASE,
    ),
    # Error/loading fallback patterns
    re.compile(r"^\s*\{\{\s*message\s*\}\}\s*$"),
    re.compile(r"^\s*You signed (in|out) with another tab or window"),
    re.compile(r"^\s*Dismiss alert\s*$"),
    re.compile(r"^\s*There was an error while loading\."),
    # GitHub UI chrome
    re.compile(r"^\s*\[?Notifications\]?.*You must be signed in"),
    re.compile(r"^\s*\[?Fork"),
    re.compile(r"^\s*\[?Star"),
    re.compile(r"^\s*\|?\s*Copy link\s*$"),
    re.compile(r"^\s*\[New issue\].*$"),
    re.compile(r"^\s*Open\s*$"),
    re.compile(r"^\s*\[?Skip to content\]?"),
    # Substack/nav chrome
    re.compile(r"^\s*SubscribeSign in\s*$"),
    re.compile(r"^\s*\[AINews:.*\]\("),
    # Avatar-byline patterns
    re.compile(r"^\s*\[!\[@?.*?['’]?s? avatar\]\]?"),
    re.compile(r"^\s*opened\s+\[?on\s+.+"),
]

_END_SECTION_TRIGGERS: list[str] = [
    "#### Subscribe to",
    "#### Discussion about",
    "## Activity",
    "## Metadata",
    "## Issue actions",
    "### Assignees",
    "### Labels",
    "### Type",
    "### Projects",
    "### Milestone",
    "### Development",
    "### Participants",
    "### Ready for more",
    "### Discussion about",
    "#### Discussion about",
]


def strip_page_chrome(content: str) -> str:
    """Strip common page chrome from web-extracted article markdown.

    Removes navigational elements, avatar images, social-proof widgets,
    subscription/sign-in prompts, and everything after known 'end of article'
    markers. Preserves the article body, tables, and substantive content.
    """
    if not content.strip():
        return ""

    lines = content.splitlines()
    stripped: list[str] = []

    for line in lines:
        # Check for end-of-article section triggers
        trimmed = line.strip()
        if _is_end_section(trimmed):
            break
        # Skip pure-chrome lines
        if _is_chrome_line(trimmed):
            continue
        stripped.append(line)

    # Remove trailing blank lines
    while stripped and not stripped[-1].strip():
        stripped.pop()

    result = "\n".join(stripped)

    # Collapse multiple consecutive blank lines into one
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def _is_chrome_line(line: str) -> bool:
    if not line:
        return False
    return any(pattern.search(line) for pattern in _CHROME_PATTERNS)


def _is_end_section(line: str) -> bool:
    return any(line.startswith(trigger) for trigger in _END_SECTION_TRIGGERS)
