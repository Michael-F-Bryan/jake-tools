"""M13: writing a render into its Obsidian note, and only into the part
of it we own.

This is the one operation in the engine that writes outside the repo, so
every rule here is fail-closed. The generated sections live inside an
explicit marker pair::

    <!-- jake-tools:transcript:begin bundle=<bundle_id> -->
    ...generated sections...
    <!-- jake-tools:transcript:end -->

Everything above and below those markers is the author's, and this module
preserves it **byte for byte**. That is the specific bug class M13 calls
out: ``merge.py``'s ``_strip_existing_generated_sections`` truncates
everything after the first generated heading, so a note with authored
content below the transcript loses it on every regeneration. Here the
prefix and suffix are sliced out before the write and compared again
after it.

Migration (first apply to a note that predates markers) treats the legacy
``## Meeting Notes`` / ``## Chapters`` / ``## Transcript`` sections as the
owned region -- but **fails closed** if any non-generated content sits
between the first and last of them, because one marker pair cannot wrap
that without absorbing the author's work. An explicit operator flag is
the only override, and it is recorded on the apply record.

Refusals, all before any byte is written:

- the target's current hash does not match the render-bound pre-write
  hash (someone edited the note since the render was computed);
- the render is of a revision that is no longer the head, without
  ``--allow-stale-render``;
- markers are duplicated, nested, unbalanced, or name a different bundle;
- legacy migration would absorb authored content.

The write itself is temp-file-plus-rename, then read back and verified
(full-file hash *and* an authored-content-unchanged check). The apply
record names one of M13's four states -- ``not-written``,
``partially-written``, ``written-unverified``, ``verified`` -- and is
written for refusals too, because an apply that did not happen is
evidence as well.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..errors import TranscriptError
from ..merge import GENERATED_HEADINGS
from .document import NoDocumentYet, project_head, project_revision
from .ids import ApplyId, BundleId, RenderId, mint_id
from .product_review import ProductReviewRequiredError, require_accepted_product_review
from .records import ApplyRecord, ApplyState, RenderRecord
from .store import BundleStore

_BEGIN_MARKER_RE = re.compile(
    r"<!--\s*jake-tools:transcript:begin\s+bundle=(?P<bundle_id>[^\s>]+)\s*-->"
)
_END_MARKER_RE = re.compile(r"<!--\s*jake-tools:transcript:end\s*-->")
_HEADING_RE = re.compile(r"^#{1,2} .*$", re.MULTILINE)


class ApplyError(TranscriptError):
    """Base class for every error this module raises."""


class AmbiguousOwnedRegionError(ApplyError):
    """The note's markers are duplicated, nested, unbalanced, or name a
    different bundle. Fail closed (M13): a region whose boundaries are
    ambiguous cannot be rewritten without risking the author's content.
    """


class LegacyMigrationRefusedError(ApplyError):
    """Adopting the legacy generated sections would absorb authored
    content that sits between them. One marker pair cannot wrap that."""


class StaleTargetError(ApplyError):
    """The note changed since the render was computed (M13 precondition).

    Re-ingest the note and re-render: the render was built against a
    snapshot that no longer describes the file, so applying it would
    silently discard whatever changed.
    """


class StaleRenderError(ApplyError):
    """The render is of a revision that is no longer the document head.

    ``allow_stale_render=True`` overrides this explicitly and the choice
    is recorded on the apply record (M13 §12.22).
    """


class ReadBackFailedError(ApplyError):
    """The bytes on disk after the write are not the bytes intended."""


@dataclass(frozen=True)
class OwnedRegion:
    """Where the generated sections live in a note, as byte offsets.

    ``prefix``/``suffix`` are the exact authored text on either side --
    kept as strings rather than as indices so the read-back check compares
    the thing it actually promised to preserve.
    """

    prefix: str
    suffix: str
    migrated_legacy_headings: bool
    adopted_existing_region: bool


def begin_marker(bundle_id: BundleId) -> str:
    return f"<!-- jake-tools:transcript:begin bundle={bundle_id} -->"


END_MARKER = "<!-- jake-tools:transcript:end -->"


def _locate_markers(text: str, *, bundle_id: BundleId) -> OwnedRegion | None:
    begins = list(_BEGIN_MARKER_RE.finditer(text))
    ends = list(_END_MARKER_RE.finditer(text))
    if not begins and not ends:
        return None
    if len(begins) != 1 or len(ends) != 1:
        raise AmbiguousOwnedRegionError(
            f"note has {len(begins)} begin marker(s) and {len(ends)} end marker(s); "
            "exactly one of each is required (M13 fails closed on duplicated, "
            "nested, or unbalanced markers)."
        )
    begin, end = begins[0], ends[0]
    if begin.start() > end.start():
        raise AmbiguousOwnedRegionError(
            "the note's end marker appears before its begin marker; the owned "
            "region's boundaries are ambiguous."
        )
    found_bundle = begin.group("bundle_id")
    if found_bundle != bundle_id:
        raise AmbiguousOwnedRegionError(
            f"the note's owned region belongs to bundle {found_bundle!r}, not "
            f"{bundle_id!r}. Applying would overwrite another bundle's output."
        )
    return OwnedRegion(
        prefix=text[: begin.start()],
        suffix=text[end.end() :],
        migrated_legacy_headings=False,
        adopted_existing_region=True,
    )


def _locate_legacy_region(text: str) -> OwnedRegion | None:
    """M13's migration case: no markers, but the legacy headings are there.

    Fails closed if any heading that is *not* one of the legacy generated
    headings appears between the first and last of them -- that is the
    "one marker pair cannot wrap this without absorbing authored content"
    case, and silently absorbing it is precisely the data loss this whole
    module exists to prevent.
    """
    positions = [
        (text.index(heading), heading)
        for heading in GENERATED_HEADINGS
        if heading in text
    ]
    if not positions:
        return None
    positions.sort()
    first_start = positions[0][0]
    last_start = positions[-1][0]

    intruders = [
        match.group(0)
        for match in _HEADING_RE.finditer(text)
        if first_start <= match.start() <= last_start
        and match.group(0).strip() not in GENERATED_HEADINGS
    ]
    if intruders:
        raise LegacyMigrationRefusedError(
            "the note has authored heading(s) between its legacy generated "
            f"sections ({intruders}); one marker pair cannot wrap the generated "
            "sections without absorbing them. Move the authored content outside "
            "the generated block, or pass the adopt flag to authorise it "
            "explicitly."
        )

    # The owned region ends where the last generated section does: at the
    # next heading after it that is not itself generated, or at EOF.
    region_end = len(text)
    for match in _HEADING_RE.finditer(text):
        if match.start() <= last_start:
            continue
        if match.group(0).strip() in GENERATED_HEADINGS:
            continue
        region_end = match.start()
        break
    return OwnedRegion(
        prefix=text[:first_start],
        suffix=text[region_end:],
        migrated_legacy_headings=True,
        adopted_existing_region=False,
    )


def locate_owned_region(text: str, *, bundle_id: BundleId) -> OwnedRegion:
    """Where this bundle's generated sections go in ``text`` (M13).

    Marker pair wins outright; otherwise the legacy headings are adopted
    (migration); otherwise this is a first-ever write and the region is
    appended after everything the author already has.
    """
    region = _locate_markers(text, bundle_id=bundle_id)
    if region is not None:
        return region
    region = _locate_legacy_region(text)
    if region is not None:
        return region
    return OwnedRegion(
        prefix=text.rstrip() + "\n" if text.strip() else "",
        suffix="",
        migrated_legacy_headings=False,
        adopted_existing_region=False,
    )


def compose_note(region: OwnedRegion, *, bundle_id: BundleId, body: str) -> str:
    """Splice ``body`` between the markers, preserving both sides exactly.

    The only string concatenation that ever produces a note this module
    writes -- so "authored content survives byte-identically" is a
    property of one expression, checkable by reading it, rather than a
    behaviour spread across a rewrite routine.
    """
    prefix = region.prefix
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    if prefix and not prefix.endswith("\n\n"):
        prefix += "\n"
    # The end marker's own trailing newline comes from the suffix, never
    # from the block: adding one here *and* keeping the suffix's would
    # grow the file by a newline on every apply, so re-applying the same
    # render would never be a no-op.
    block = f"{begin_marker(bundle_id)}\n{body.rstrip()}\n{END_MARKER}"
    suffix = region.suffix
    if not suffix:
        suffix = "\n"
    elif not suffix.startswith("\n"):
        suffix = "\n" + suffix
    return prefix + block + suffix


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    descriptor, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ApplyOutcome:
    record: ApplyRecord
    written: bool
    unchanged: bool


def apply_render(
    store: BundleStore,
    *,
    render_id: RenderId,
    target_path: Path,
    allow_stale_render: bool = False,
    adopt_edited_region: bool = False,
) -> ApplyOutcome:
    """M13: write one render into ``target_path``'s owned region.

    Idempotent: applying the same render to an already-matching note
    writes nothing and records ``verified`` -- re-running is safe and
    costs one read.

    Every refusal is recorded as an apply record in state ``not-written``
    *and* raised, so the bundle's own history shows the refusal even
    though the caller sees an exception.
    """
    record = store.load_render(render_id)
    try:
        require_accepted_product_review(store, record)
    except ProductReviewRequiredError as exc:
        raise _refuse(
            store,
            record,
            target_path,
            precondition_sha256=_target_hash(target_path),
            detail=str(exc),
            error=ProductReviewRequiredError,
        ) from exc
    body = store.load_render_output(render_id).decode("utf-8")

    if not allow_stale_render:
        head = project_head(store)
        head_revision_id = None if isinstance(head, NoDocumentYet) else head.revision_id
        if head_revision_id != record.revision_id:
            raise _refuse(
                store,
                record,
                target_path,
                precondition_sha256=_target_hash(target_path),
                detail=(
                    f"render {render_id} is of revision {record.revision_id}, but the "
                    f"document head is now {head_revision_id}; pass "
                    "--allow-stale-render to apply it anyway (recorded)."
                ),
                error=StaleRenderError,
            )

    expected_hash = _expected_target_hash(store, record)
    current_text = (
        target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    )
    current_hash = _sha256_text(current_text)
    if (
        expected_hash is not None
        and current_hash != expected_hash
        and not _is_our_own_previous_write(store, record, current_hash)
    ):
        raise _refuse(
            store,
            record,
            target_path,
            precondition_sha256=current_hash,
            detail=(
                f"target {target_path} hashes to {current_hash}, but this render was "
                f"computed against the snapshot {expected_hash}. Re-ingest the note "
                "and re-render rather than overwriting changes made since."
            ),
            error=StaleTargetError,
        )

    try:
        region = locate_owned_region(current_text, bundle_id=record.bundle_id)
    except (AmbiguousOwnedRegionError, LegacyMigrationRefusedError) as exc:
        if not (adopt_edited_region and isinstance(exc, LegacyMigrationRefusedError)):
            raise _refuse(
                store,
                record,
                target_path,
                precondition_sha256=current_hash,
                detail=str(exc),
                error=type(exc),
            ) from exc
        region = _adopt_legacy_region(current_text)

    proposed = compose_note(region, bundle_id=record.bundle_id, body=body)
    if proposed == current_text:
        return ApplyOutcome(
            record=store.add_apply(
                ApplyRecord(
                    apply_id=_mint_apply_id(),
                    bundle_id=record.bundle_id,
                    render_id=render_id,
                    revision_id=record.revision_id,
                    target_path=str(target_path),
                    precondition_sha256=current_hash,
                    post_write_sha256=current_hash,
                    state=ApplyState.VERIFIED,
                    allow_stale_render=allow_stale_render,
                    adopted_edited_region=adopt_edited_region,
                    migrated_legacy_headings=region.migrated_legacy_headings,
                    detail="target already matches this render; nothing written",
                    created_at=record.created_at,
                )
            ),
            written=False,
            unchanged=True,
        )

    _write_atomic(target_path, proposed)
    read_back = target_path.read_text(encoding="utf-8")
    read_back_hash = _sha256_text(read_back)
    authored_preserved = read_back.startswith(region.prefix) and read_back.endswith(
        region.suffix
    )
    if read_back != proposed or not authored_preserved:
        stored = store.add_apply(
            ApplyRecord(
                apply_id=_mint_apply_id(),
                bundle_id=record.bundle_id,
                render_id=render_id,
                revision_id=record.revision_id,
                target_path=str(target_path),
                precondition_sha256=current_hash,
                post_write_sha256=read_back_hash,
                state=ApplyState.WRITTEN_UNVERIFIED,
                allow_stale_render=allow_stale_render,
                adopted_edited_region=adopt_edited_region,
                migrated_legacy_headings=region.migrated_legacy_headings,
                detail=(
                    "read-back does not match the intended content, or authored "
                    "content outside the owned region did not survive"
                ),
                created_at=record.created_at,
            )
        )
        raise ReadBackFailedError(
            f"apply {stored.apply_id} wrote {target_path} but read-back verification "
            "failed; the file is in state 'written-unverified' and must be inspected "
            "by hand."
        )

    return ApplyOutcome(
        record=store.add_apply(
            ApplyRecord(
                apply_id=_mint_apply_id(),
                bundle_id=record.bundle_id,
                render_id=render_id,
                revision_id=record.revision_id,
                target_path=str(target_path),
                precondition_sha256=current_hash,
                post_write_sha256=read_back_hash,
                state=ApplyState.VERIFIED,
                allow_stale_render=allow_stale_render,
                adopted_edited_region=adopt_edited_region,
                migrated_legacy_headings=region.migrated_legacy_headings,
                created_at=record.created_at,
            )
        ),
        written=True,
        unchanged=False,
    )


def _adopt_legacy_region(text: str) -> OwnedRegion:
    """The operator-authorised override for M13's legacy migration.

    Adopts the span from the first to the last legacy generated heading
    *including* whatever authored content sits between them. Only reached
    when the operator asked for it explicitly, and always recorded on the
    apply record -- authorised data loss is still data loss, and it must
    be attributable.
    """
    positions = sorted(
        text.index(heading) for heading in GENERATED_HEADINGS if heading in text
    )
    first_start, last_start = positions[0], positions[-1]
    region_end = len(text)
    for match in _HEADING_RE.finditer(text):
        if match.start() <= last_start:
            continue
        if match.group(0).strip() in GENERATED_HEADINGS:
            continue
        region_end = match.start()
        break
    return OwnedRegion(
        prefix=text[:first_start],
        suffix=text[region_end:],
        migrated_legacy_headings=True,
        adopted_existing_region=True,
    )


def _is_our_own_previous_write(
    store: BundleStore, record: RenderRecord, current_hash: str
) -> bool:
    """Whether this bundle previously verified the exact current target bytes.

    A later accepted render must be able to replace an earlier render's owned
    region without pretending the note was untouched. The verified apply record
    is the evidence; arbitrary edited bytes still match no record and fail closed.
    """
    return any(
        applied.bundle_id == record.bundle_id
        and applied.state == ApplyState.VERIFIED
        and applied.post_write_sha256 == current_hash
        for applied in store.iter_applies()
    )


def _expected_target_hash(store: BundleStore, record: RenderRecord) -> str | None:
    """The note snapshot this render was computed against (M13).

    ``None`` when the render was not bound to a destination snapshot at
    all -- there is then no precondition to check, and the apply is a
    first write to whatever path the operator named.
    """
    if record.destination_snapshot_artefact_id is None:
        return None
    return store.load_artefact(record.destination_snapshot_artefact_id).sha256


def _target_hash(target_path: Path) -> str:
    if not target_path.exists():
        return _sha256_text("")
    return _sha256_text(target_path.read_text(encoding="utf-8"))


def _mint_apply_id() -> ApplyId:
    return mint_id("apply")


def _refuse(
    store: BundleStore,
    record: RenderRecord,
    target_path: Path,
    *,
    precondition_sha256: str,
    detail: str,
    error: type[TranscriptError],
) -> TranscriptError:
    """Record a refusal and build the exception to raise for it.

    Returns rather than raises so the call sites read ``raise _refuse(...)``
    -- the record is always written first, so a refusal is visible in the
    bundle's own history even though the operator only sees the error.
    """
    store.add_apply(
        ApplyRecord(
            apply_id=_mint_apply_id(),
            bundle_id=record.bundle_id,
            render_id=record.render_id,
            revision_id=record.revision_id,
            target_path=str(target_path),
            precondition_sha256=precondition_sha256,
            state=ApplyState.NOT_WRITTEN,
            detail=detail,
            created_at=record.created_at,
        )
    )
    return error(detail)


def render_is_current(store: BundleStore, render_id: RenderId) -> bool:
    """Whether ``render_id`` is a render of the bundle's current head."""
    record = store.load_render(render_id)
    head = project_head(store)
    if isinstance(head, NoDocumentYet):
        return False
    return head.revision_id == record.revision_id


def rendered_revision(store: BundleStore, render_id: RenderId) -> str:
    """Which revision a render is bound to -- the only correct way to ask,
    since a render never looks at the head (M17)."""
    record = store.load_render(render_id)
    project_revision(store, record.revision_id)
    return record.revision_id
