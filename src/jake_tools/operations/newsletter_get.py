"""newsletter_get: one entry with its saved content and attachments."""

from __future__ import annotations

import functools

from ..config import Config
from ..newsletter_entries import NewsletterEntry, get_entry
from . import Handler, Operation
from ._newsletter import EntryId, call_newsletter

DESCRIPTION = (
    "Read-only. Fetch one CSU Weekly Newsletter entry by numeric entry_id: id, URL, "
    "title, saved plain-text body, created/modified, and attachment names with URLs. "
    "Attachment size and sha256 are not listed by SharePoint, so they are null here "
    "(a create/update result carries them for content it verified). Uses the "
    "server's configured Microsoft account; never another."
)


def build(config: Config) -> Handler:
    async def newsletter_get(entry_id: EntryId) -> NewsletterEntry:
        return await call_newsletter(
            config, functools.partial(get_entry, entry_id=entry_id), entry_id=entry_id
        )

    return newsletter_get


OPERATION = Operation(
    name="newsletter_get",
    capability="newsletter",
    skill="newsletters",
    description=DESCRIPTION,
    build=build,
)
