"""newsletter_create: publish one new entry, with optional attachments, verified."""

from __future__ import annotations

from ..config import Config
from ..newsletter_entries import NewsletterWriteResult, create_entry
from ..newsletters import NewsletterClient
from . import Handler, Operation
from ._newsletter import call_newsletter, prepare
from .errors import ToolError

DESCRIPTION = (
    "PUBLICATION ACTION: creates a new CSU Weekly Newsletter entry, visible to its "
    "readers once saved. Only call with the title, body and files the user approved; "
    "never draft-and-publish on your own. title (<=255 chars) and plain-text body "
    "(blank lines separate paragraphs; HTML is escaped) are required. attachments are "
    "absolute paths of files on the jake-tools host (<=10, each <=25 MiB, unique "
    "names). Steps: create the entry once, add each attachment, then read everything "
    "back and download each attachment to compare bytes. Success means verified=true "
    "with the saved entry. On failure the error detail has the entry_id (if one "
    "exists) and an effect per step: applied, failed (rejected, did not happen), "
    "uncertain (no answer; may have happened) or not_attempted. NEVER call "
    "newsletter_create again after a failure that has an entry_id or outcome "
    "create_uncertain: it would duplicate the entry. Recover with newsletter_update. "
    "Nothing is retried automatically. Acts as the server's configured Microsoft account."
)


def build(config: Config) -> Handler:
    async def newsletter_create(
        title: str, body: str, attachments: list[str] | None = None
    ) -> NewsletterWriteResult:
        if not title.strip() or len(title) > 255:
            raise ToolError(
                "invalid_argument",
                "title must be 1..255 characters",
                detail={"argument": "title"},
            )
        if not body.strip() or len(body) > 100_000:
            raise ToolError(
                "invalid_argument",
                "body must be non-blank and at most 100000 characters",
                detail={"argument": "body"},
            )
        read = prepare(attachments or [])

        def run(client: NewsletterClient) -> NewsletterWriteResult:
            return create_entry(client, title=title, body=body, attachments=read())

        return await call_newsletter(config, run)

    return newsletter_create


OPERATION = Operation(
    name="newsletter_create",
    capability="newsletter_publish",
    skill="newsletters",
    description=DESCRIPTION,
    build=build,
)
