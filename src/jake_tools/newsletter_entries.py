"""Typed newsletter entry operations: list, get, create, update, with readback.

The low-level HTTP lives in :class:`~jake_tools.newsletters.NewsletterClient`;
this module owns what a caller can trust:

* **Coverage.** A list is one bounded page plus the cursor to the next, never
  a silently truncated list.
* **Verification.** A mutation is accepted only after a separate readback:
  title and body re-read, untouched fields unchanged, and every supplied
  attachment *downloaded* and compared byte for byte.
* **Honest partial outcomes.** The entry is created first, then each
  attachment is added in turn. When a step fails, :class:`NewsletterWriteError`
  carries the entry ID and an effect for every step: ``applied``, ``failed``
  (the upstream rejected it, so it did not happen), ``uncertain`` (no answer;
  it may have happened) or ``not_attempted``. Nothing is retried, and a
  create is never repeated: recover by updating the known entry. Updating
  with an attachment already present with identical bytes is a no-op, so
  recovery cannot duplicate or overwrite it.

Publishing is a real-world action; callers decide whether to do it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .http import UpstreamError
from .microsoft_auth import MicrosoftAuthError
from .newsletters import (
    NewsletterAttachmentRef,
    NewsletterClient,
    NewsletterError,
    NewsletterItem,
    body_from_html,
    body_to_html,
)

MAX_ATTACHMENTS = 10
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
"""Jake's own bound (SharePoint accepts larger), keeping a call's memory known."""
MAX_TITLE_CHARS = 255
FORBIDDEN_NAME_CHARACTERS = frozenset('~"#%&*:<>?/\\{|}')

EffectKind = Literal["create_entry", "update_fields", "add_attachment"]
EffectStatus = Literal["applied", "unchanged", "failed", "uncertain", "not_attempted"]
WriteOutcome = Literal[
    "not_applied",
    "create_uncertain",
    "partial",
    "verification_failed",
    "verification_unavailable",
]


class AttachmentInfo(BaseModel):
    """An attachment. ``size_bytes`` and ``sha256`` are set only for content
    this call downloaded; listing alone does not reveal them."""

    name: str
    url: str
    size_bytes: int | None = None
    sha256: str | None = None


class NewsletterEntry(NewsletterItem):
    """A saved entry. ``attachments`` is ``None`` when they were not read."""

    attachments: list[AttachmentInfo] | None = None


class NewsletterListResult(BaseModel):
    """One page of entries, newest first.

    ``complete`` is true only when this page reaches the end of the list. When
    false, call again with ``next_cursor`` and the same ``limit``. Paging is
    not a snapshot: entries created or removed between pages can shift it.
    """

    entries: list[NewsletterEntry]
    count: int
    limit: int
    next_cursor: str | None = None
    complete: bool
    bodies_included: bool


class Effect(BaseModel):
    kind: EffectKind
    subject: str
    """``entry`` for the entry's own fields, or the attachment's file name."""
    status: EffectStatus
    verified: bool | None = None
    """Whether the readback confirmed it; ``None`` when no readback happened."""
    detail: str | None = None


class NewsletterWriteResult(BaseModel):
    """A mutation the readback confirmed in full."""

    operation: Literal["create", "update"]
    verified: Literal[True] = True
    entry: NewsletterEntry
    effects: list[Effect]
    compared: list[str]
    """What the readback compared, e.g. ``title``, ``attachment:a.pdf (bytes)``."""


class NewsletterInputError(ValueError):
    """The request cannot be carried out; nothing was changed."""

    def __init__(self, argument: str, message: str) -> None:
        super().__init__(message)
        self.argument = argument


class NewsletterWriteError(RuntimeError):
    """A mutation did not complete or could not be confirmed."""

    def __init__(
        self,
        outcome: WriteOutcome,
        message: str,
        *,
        entry_id: str | None,
        entry_url: str | None = None,
        effects: list[Effect],
        mismatches: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome: WriteOutcome = outcome
        self.entry_id = entry_id
        self.entry_url = entry_url
        self.effects = effects
        self.mismatches = mismatches or []

    @property
    def retry_safe(self) -> bool:
        """True only when the upstream rejected the first write outright."""
        return self.outcome == "not_applied"

    @property
    def next_step(self) -> str:
        match self.outcome:
            case "not_applied":
                return "Nothing was changed. Fix the cause and call again."
            case "create_uncertain":
                return (
                    "The entry may or may not exist and its ID is unknown. Do NOT "
                    "create again. List recent entries, find it by title, and "
                    "continue with newsletter_update on that ID."
                )
            case "partial":
                return (
                    "The entry exists. Do NOT create again. Check each effect: "
                    "retry only attachments marked failed or not_attempted by "
                    "calling newsletter_update on this entry ID; an attachment "
                    "already present with identical bytes is left unchanged."
                )
            case "verification_failed":
                return (
                    "The write completed but the readback differs from what was "
                    "sent (see mismatches). Inspect the entry with newsletter_get "
                    "before changing it again; do not blindly repeat the write."
                )
            case "verification_unavailable":
                return (
                    "The write completed but could not be read back. Inspect the "
                    "entry with newsletter_get before reporting it done."
                )

    def detail(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "entry_id": self.entry_id,
            "entry_url": self.entry_url,
            "effects": [effect.model_dump(mode="json") for effect in self.effects],
            "mismatches": self.mismatches,
            "retry_safe": self.retry_safe,
            "next_step": self.next_step,
        }


@dataclass(frozen=True)
class PreparedAttachment:
    name: str
    content: bytes
    sha256: str


def prepare_attachments(paths: Sequence[Path]) -> list[PreparedAttachment]:
    """Read every local attachment up front, so a bad path fails before any write."""
    if len(paths) > MAX_ATTACHMENTS:
        raise NewsletterInputError(
            "attachments", f"at most {MAX_ATTACHMENTS} attachments per call"
        )
    prepared: list[PreparedAttachment] = []
    seen: set[str] = set()
    for path in paths:
        name = path.name
        bad = sorted(set(name) & FORBIDDEN_NAME_CHARACTERS)
        if not name or name.startswith(".") or name.endswith(".") or bad:
            raise NewsletterInputError(
                "attachments",
                f"{name!r} is not a usable SharePoint file name"
                + (f" (contains {' '.join(bad)})" if bad else "")
                + "; rename the file",
            )
        if name.casefold() in seen:
            raise NewsletterInputError(
                "attachments", f"two attachments are named {name!r}"
            )
        seen.add(name.casefold())
        if not path.is_file():
            raise NewsletterInputError(
                "attachments", f"{path} is not a file readable by this process"
            )
        if path.stat().st_size > MAX_ATTACHMENT_BYTES:
            raise NewsletterInputError(
                "attachments",
                f"{name!r} exceeds {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MiB",
            )
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise NewsletterInputError(
                "attachments", f"{path} cannot be read: {exc.strerror}"
            ) from exc
        prepared.append(
            PreparedAttachment(name, content, hashlib.sha256(content).hexdigest())
        )
    return prepared


def list_entries(
    client: NewsletterClient,
    *,
    limit: int,
    cursor: str | None,
    include_body: bool,
) -> NewsletterListResult:
    page = client.list_page(limit=limit, cursor=cursor)
    entries = [
        NewsletterEntry(
            **(item.model_dump() if include_body else item.model_dump() | {"body": ""})
        )
        for item in page.items
    ]
    return NewsletterListResult(
        entries=entries,
        count=len(entries),
        limit=limit,
        next_cursor=page.next_cursor,
        complete=page.next_cursor is None,
        bodies_included=include_body,
    )


def get_entry(client: NewsletterClient, entry_id: str) -> NewsletterEntry:
    item = client.get_item(entry_id)
    return _entry(client, item, client.list_attachments(entry_id), {})


@dataclass(frozen=True)
class _Expected:
    """What the readback must show. Labels say which request it answers."""

    title: str
    title_label: str
    body: str
    body_label: str


def create_entry(
    client: NewsletterClient,
    *,
    title: str,
    body: str,
    attachments: Sequence[PreparedAttachment],
) -> NewsletterWriteResult:
    create = Effect(kind="create_entry", subject="entry", status="not_attempted")
    adds = [_pending(a) for a in attachments]
    try:
        created = client.create_entry(title=title, body=body)
    except UpstreamError as exc:
        _fail_step(create, exc)
        rejected = create.status == "failed"
        raise NewsletterWriteError(
            "not_applied" if rejected else "create_uncertain",
            "creating the entry "
            + ("was rejected" if rejected else "has an unknown outcome")
            + f": {_why(exc)}",
            entry_id=None,
            effects=[create, *adds],
        ) from exc
    create.status = "applied"
    return _add_and_verify(
        client,
        "create",
        created.id,
        created.web_url,
        [create],
        adds,
        attachments,
        supplied=attachments,
        expected=_Expected(title, "title", _normal_body(body), "body"),
        before_refs=[],
        creating=True,
    )


def update_entry(
    client: NewsletterClient,
    entry_id: str,
    *,
    title: str | None,
    body: str | None,
    attachments: Sequence[PreparedAttachment],
) -> NewsletterWriteResult:
    # Everything that can fail without changing anything happens first.
    before = client.get_item(entry_id)
    before_refs = client.list_attachments(entry_id)
    existing = {ref.name.casefold(): ref for ref in before_refs}
    to_add: list[PreparedAttachment] = []
    unchanged: list[Effect] = []
    for attachment in attachments:
        ref = existing.get(attachment.name.casefold())
        if ref is None:
            to_add.append(attachment)
            continue
        if (
            ref.name != attachment.name
            or client.download_attachment(entry_id, ref.name) != attachment.content
        ):
            raise NewsletterInputError(
                "attachments",
                f"the entry already has an attachment named {ref.name!r} with "
                "different content; Jake does not replace attachments. Rename "
                "the new file or remove the old one in SharePoint.",
            )
        unchanged.append(
            Effect(
                kind="add_attachment",
                subject=attachment.name,
                status="unchanged",
                detail="already attached with identical bytes",
            )
        )
    expected = _Expected(
        title if title is not None else before.title,
        "title" if title is not None else "title unchanged",
        _normal_body(body) if body is not None else before.body,
        "body" if body is not None else "body unchanged",
    )
    adds = [_pending(a) for a in to_add]
    fields: list[Effect] = []
    if title is not None or body is not None:
        step = Effect(kind="update_fields", subject="entry", status="not_attempted")
        fields.append(step)
        try:
            client.update_fields(entry_id, title=title, body=body)
        except UpstreamError as exc:
            _fail_step(step, exc)
            rejected = step.status == "failed"
            raise NewsletterWriteError(
                "not_applied" if rejected else "partial",
                "updating the entry's fields "
                + ("was rejected" if rejected else "has an unknown outcome")
                + f": {_why(exc)}",
                entry_id=entry_id,
                entry_url=before.url,
                effects=[*fields, *adds, *unchanged],
            ) from exc
        step.status = "applied"
    return _add_and_verify(
        client,
        "update",
        entry_id,
        before.url,
        fields,
        adds,
        to_add,
        supplied=attachments,
        expected=expected,
        before_refs=before_refs,
        unchanged=unchanged,
    )


def _add_and_verify(
    client: NewsletterClient,
    operation: Literal["create", "update"],
    entry_id: str,
    entry_url: str,
    done: list[Effect],
    adds: list[Effect],
    to_add: Sequence[PreparedAttachment],
    *,
    supplied: Sequence[PreparedAttachment],
    expected: _Expected,
    before_refs: list[NewsletterAttachmentRef],
    creating: bool = False,
    unchanged: list[Effect] | None = None,
) -> NewsletterWriteResult:
    """Add the attachments in order (stopping at the first failure), then
    verify the whole entry by an independent readback."""
    effects = [*done, *adds, *(unchanged or [])]
    for attachment, effect in zip(to_add, adds, strict=True):
        try:
            client.add_attachment_bytes(entry_id, attachment.name, attachment.content)
        except UpstreamError as exc:
            _fail_step(effect, exc)
            _reconcile(client, entry_id, adds, to_add)
            raise NewsletterWriteError(
                "partial",
                f"adding attachment {attachment.name!r} "
                + (
                    "was rejected"
                    if effect.status == "failed"
                    else "has an unknown outcome"
                )
                + f": {_why(exc)}",
                entry_id=entry_id,
                entry_url=entry_url,
                effects=effects,
            ) from exc
        effect.status = "applied"
    try:
        item = client.get_item(entry_id)
        refs = client.list_attachments(entry_id)
        listed = {ref.name.casefold() for ref in refs}
        downloaded = {
            a.name: client.download_attachment(entry_id, a.name)
            for a in supplied
            if a.name.casefold() in listed
        }
    except UpstreamError as exc:
        raise NewsletterWriteError(
            "verification_unavailable",
            f"the entry was written but could not be read back: {_why(exc)}",
            entry_id=entry_id,
            entry_url=entry_url,
            effects=effects,
        ) from exc

    compared: list[str] = []
    mismatches: list[str] = []
    _check(compared, mismatches, expected.title_label, item.title == expected.title)
    _check(compared, mismatches, expected.body_label, item.body == expected.body)
    for ref in sorted(before_refs, key=lambda r: r.name.casefold()):
        _check(
            compared,
            mismatches,
            f"attachment:{ref.name} still present",
            ref.name.casefold() in listed,
        )
    for attachment in supplied:
        _check(
            compared,
            mismatches,
            f"attachment:{attachment.name} (bytes)",
            downloaded.get(attachment.name) == attachment.content,
        )
    if creating:
        wanted = {a.name.casefold() for a in supplied}
        _check(compared, mismatches, "no unexpected attachments", listed == wanted)
    for effect in effects:
        if effect.kind == "add_attachment":
            effect.verified = f"attachment:{effect.subject} (bytes)" not in mismatches
        else:
            effect.verified = not {expected.title_label, expected.body_label} & set(
                mismatches
            )
    if mismatches:
        raise NewsletterWriteError(
            "verification_failed",
            "the readback differs from what was written: " + "; ".join(mismatches),
            entry_id=entry_id,
            entry_url=item.url or entry_url,
            effects=effects,
            mismatches=mismatches,
        )
    return NewsletterWriteResult(
        operation=operation,
        entry=_entry(client, item, refs, {a.name: a for a in supplied}),
        effects=effects,
        compared=compared,
    )


def _entry(
    client: NewsletterClient,
    item: NewsletterItem,
    refs: list[NewsletterAttachmentRef],
    verified: dict[str, PreparedAttachment],
) -> NewsletterEntry:
    infos = []
    for ref in refs:
        known = verified.get(ref.name)
        infos.append(
            AttachmentInfo(
                name=ref.name,
                url=client.absolute_url(ref.server_relative_url),
                size_bytes=None if known is None else len(known.content),
                sha256=None if known is None else known.sha256,
            )
        )
    return NewsletterEntry.model_validate(
        {**item.model_dump(), "attachments": infos}
    )


def _pending(attachment: PreparedAttachment) -> Effect:
    return Effect(
        kind="add_attachment", subject=attachment.name, status="not_attempted"
    )


def _fail_step(effect: Effect, exc: UpstreamError) -> None:
    """Record ``exc`` on ``effect``: ``failed`` if the upstream refused it or
    nothing was sent (a token problem), otherwise ``uncertain``."""
    definitely_not_applied = isinstance(exc, MicrosoftAuthError) or (
        isinstance(exc, NewsletterError) and exc.rejected
    )
    effect.status = "failed" if definitely_not_applied else "uncertain"
    effect.detail = _why(exc)


def _reconcile(
    client: NewsletterClient,
    entry_id: str,
    adds: list[Effect],
    to_add: Sequence[PreparedAttachment],
) -> None:
    """Best-effort: settle ``uncertain`` attachment effects from a readback.

    Promotes ``uncertain`` to ``applied`` only when the identical bytes are
    verifiably there. Changes nothing upstream and never raises.
    """
    uncertain = [
        (a, e) for a, e in zip(to_add, adds, strict=True) if e.status == "uncertain"
    ]
    if not uncertain:
        return
    try:
        present = {r.name for r in client.list_attachments(entry_id)}
        for attachment, effect in uncertain:
            if attachment.name in present and (
                client.download_attachment(entry_id, attachment.name)
                == attachment.content
            ):
                effect.status = "applied"
                effect.verified = True
                effect.detail = "readback found identical bytes after an unknown reply"
    except UpstreamError:
        return


def _check(compared: list[str], mismatches: list[str], name: str, ok: bool) -> None:
    compared.append(name)
    if not ok:
        mismatches.append(name)


def _normal_body(body: str) -> str:
    return body_from_html(body_to_html(body))


def _why(exc: UpstreamError) -> str:
    return exc.safe_summary or type(exc).__name__
