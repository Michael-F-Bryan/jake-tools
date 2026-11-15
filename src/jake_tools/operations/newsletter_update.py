"""newsletter_update: change an existing entry's fields and/or add attachments."""

from __future__ import annotations

from ..config import Config
from ..newsletter_entries import NewsletterWriteResult, update_entry
from ..newsletters import NewsletterClient
from . import Handler, Operation
from ._newsletter import EntryId, call_newsletter, prepare
from .errors import ToolError

DESCRIPTION = (
    "PUBLICATION ACTION: changes an existing CSU Weekly Newsletter entry that readers "
    "can see. Only call with changes the user approved. Supply any of title, body "
    "(plain text, as for newsletter_create) and attachments (absolute paths on the "
    "jake-tools host); at least one. Omitted fields are not sent and are verified "
    "unchanged, so an attachments-only update never touches title or body. Attachments "
    "are only ever ADDED: one already on the entry with identical bytes is left "
    "unchanged (so this is the safe way to finish after a partial create), and one "
    "with the same name but different bytes is refused before anything is written. "
    "The entry is read first, then written, then read back with each attachment "
    "downloaded and compared byte for byte. Success means verified=true. On failure "
    "the error detail has the entry_id and an effect per step (applied, failed, "
    "uncertain, not_attempted); nothing is retried automatically, and a field update "
    "is not a compare-and-swap, so a concurrent edit can interleave. Acts as the "
    "server's configured Microsoft account."
)


def build(config: Config) -> Handler:
    async def newsletter_update(
        entry_id: EntryId,
        title: str | None = None,
        body: str | None = None,
        attachments: list[str] | None = None,
    ) -> NewsletterWriteResult:
        if title is None and body is None and not attachments:
            raise ToolError(
                "invalid_argument",
                "nothing to update: supply title, body or attachments",
            )
        if title is not None and (not title.strip() or len(title) > 255):
            raise ToolError(
                "invalid_argument",
                "title must be 1..255 characters",
                detail={"argument": "title"},
            )
        if body is not None and (not body.strip() or len(body) > 100_000):
            raise ToolError(
                "invalid_argument",
                "body must be non-blank and at most 100000 characters",
                detail={"argument": "body"},
            )
        read = prepare(attachments or [])

        def run(client: NewsletterClient) -> NewsletterWriteResult:
            return update_entry(
                client, entry_id, title=title, body=body, attachments=read()
            )

        return await call_newsletter(config, run, entry_id=entry_id)

    return newsletter_update


OPERATION = Operation(
    name="newsletter_update",
    capability="newsletter_publish",
    skill="newsletters",
    description=DESCRIPTION,
    build=build,
)
