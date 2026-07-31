"""Filesystem-backed bundle store (M16).

``BundleStore`` owns one bundle's directory end to end: minting every ID a
bundle ever sees, the content-addressed blob store, the append-only
artefact/revision/run/attempt records, the transactional ``manifest.json``
head pointer, and the single-active-run lease.

Two concerns are kept deliberately separate:

- **Crash safety** (a process dies mid-write) is handled per-write by one
  of two IO primitives: :meth:`BundleStore._write_json_exclusive` (records
  minted once and never rewritten — artefacts, revisions, freshly-created
  runs, the lease file — via a temp file + the injectable ``link`` seam
  (default ``os.link``), which is atomic *and* exclusive, so a crash never
  leaves a torn file that append-only rules would then block ever
  correctly rewriting) and :meth:`BundleStore._write_json_atomic` (files
  that legitimately change after creation — ``manifest.json`` and a run's
  own record as it moves through its state machine — via a temp file and
  the injectable ``replace`` seam). Both seams default to the real syscall
  and exist so tests can simulate a crash at the exact commit point of
  each primitive (between the temp write and the rename/link). Note
  ``os.link`` requires the bundle root to sit on a filesystem that
  supports hard links (true of the default ``_working/`` on APFS/ext4/
  etc.); since ``root`` is operator-overridable, pointing it at exFAT,
  FUSE, or another filesystem without hard-link support would break this
  primitive.
- **Cross-process concurrency** (two processes interleave between a read
  and the write it informs) is handled by :meth:`BundleStore._locked`, a
  single advisory ``flock`` per bundle, reentrant per thread, held for the
  duration of one read-modify-write critical section. This covers not
  just manifest/lease read-then-write sequences (``register_source``,
  ``create_run``'s manifest update, ``update_head``, ``acquire_lease``,
  ``release_lease``) but *any* decision made from a read before a write —
  notably ``ingest_artefact``'s dedup check (does a matching artefact
  already exist) is a read-modify-write too, not a pure append, and holds
  this same lock. Only genuinely unconditional writes skip it: minting a
  revision, and the content-addressed blob write itself (racing writers
  of the same ``sha256`` write byte-identical content, so the race is
  harmless).
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..errors import TranscriptError
from .components import (
    ComponentBody,
    ComponentKind,
    ComponentRecord,
    assemble_component_record,
    component_as_body,
    component_input_refs,
)
from .ids import (
    ApplyId,
    ArtefactId,
    ComponentId,
    RenderId,
    ReviewId,
    RevisionId,
    RunId,
    SourceId,
    mint_id,
)
from .records import (
    DURABLE_RUN_STATES,
    ApplyRecord,
    ArtefactRecord,
    BundleManifest,
    Lease,
    NoDocumentYet,
    OperationRef,
    RenderRecord,
    ReviewRecord,
    RevisionRecord,
    RunRecord,
    RunState,
    SourceAssociation,
    SourceMembershipRecord,
)

ReplaceFn = Callable[[str, str], None]
LinkFn = Callable[[str, str], None]
CapabilityValidator = Callable[[RevisionRecord], None]

_SUBDIRECTORIES = (
    "blobs",
    "artefacts",
    "revisions",
    "components",
    "runs",
    "attempts",
    "reviews",
    "renders",
    "applies",
)

_LOCK_FILENAME = ".store.lock"

# Lock-nesting depths per thread, keyed by resolved bundle root. Module
# level (not per instance) so a callback that constructs its OWN
# BundleStore for the same root -- the natural shape for a
# validate_capabilities implementation, which receives no store handle --
# nests instead of self-deadlocking on the flock.
_LOCK_DEPTHS = threading.local()


def _lock_depths() -> dict[str, int]:
    depths = getattr(_LOCK_DEPTHS, "by_root", None)
    if depths is None:
        depths = {}
        _LOCK_DEPTHS.by_root = depths
    return depths


_ACQUIRABLE_RUN_STATES = (RunState.CREATED, *DURABLE_RUN_STATES)

# M2's legal state edges, keyed by the *current* state. Enforced generically
# by `_replace_run_fields` so every transition -- from `acquire_lease` and
# `release_lease` alike -- goes through the same table rather than each call
# site re-deriving which moves are legal.
_LEGAL_RUN_STATE_EDGES: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.RUNNING}),
    RunState.RUNNING: frozenset(
        {
            RunState.REVIEW_REQUIRED,
            RunState.REFUSED,
            RunState.FAILED,
            RunState.COMPLETED,
        }
    ),
    RunState.REVIEW_REQUIRED: frozenset({RunState.RUNNING}),
    RunState.REFUSED: frozenset({RunState.RUNNING}),
    RunState.FAILED: frozenset({RunState.RUNNING}),
    RunState.COMPLETED: frozenset(),
}

_APPLY_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(ApplyId)
_ARTEFACT_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(ArtefactId)
_RENDER_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(RenderId)
_REVIEW_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(ReviewId)
_REVISION_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(RevisionId)
_RUN_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(RunId)
_COMPONENT_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(ComponentId)
_COMPONENT_RECORD_ADAPTER: TypeAdapter[ComponentRecord] = TypeAdapter(ComponentRecord)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _component_content_hash(model: ComponentBody | ComponentRecord) -> str:
    """M1: SHA-256 of a component's canonical JSON body.

    Always hashes :func:`.components.component_as_body`'s *reconstruction*
    of ``model`` -- for a bare ``*Body`` that is ``model`` itself; for a
    stored ``*Component`` record it is rebuilt field by field with every
    store-minted identity field dropped, including ones nested *inside* a
    field (MINOR C: ``NotesComponent.sections`` each carry their own
    minted ``section_id``, invisible to a top-level ``include=``/``exclude=``
    filter). This is what makes ``hash(body) == hash(record)`` for the
    same content -- a field-name filter alone got this wrong for notes
    once ``NotesComponent`` stopped being a ``NotesComponentBody``
    subclass -- and is what makes :meth:`BundleStore.add_component`'s
    dedup check and :meth:`BundleStore.load_component`'s content_hash
    verification both correct.
    """
    body = component_as_body(model)
    payload = body.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True)
    return _sha256_hex(canonical.encode("utf-8"))


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A process owned by someone else still counts as alive.
        return True
    return True


def _no_op_capability_validator(revision: RevisionRecord) -> None:
    """Structural-only ``validate_capabilities`` seam (M16) -- the explicit
    escape hatch for ``validate_capabilities=None``.

    Structural closure — revision/ancestor/artefact/component existence
    and refs — is always checked by :meth:`BundleStore.update_head` via
    :meth:`BundleStore._validate_structural_closure`, regardless of this
    seam. This function skips *semantic* (M4 registry) validation
    entirely; the default seam (:meth:`BundleStore._default_capability_validator`)
    enforces the registry, so this is only reached when a caller
    deliberately opts out (e.g. a diagnostic script, or a test that needs
    to construct a deliberately-invalid revision without the registry
    refusing it).
    """


class _UseRegistryDefault:
    """Sentinel type for ``BundleStore``'s ``validate_capabilities`` default.

    Distinct from ``None`` (the explicit "structural only" escape hatch)
    and from a caller-supplied callable: it means "bind the real M4
    registry to *this* store's own root", resolved lazily in
    :meth:`BundleStore._default_capability_validator` so ``store.py``
    never has to import ``document.py`` at module level (that import runs
    the other way -- ``document.py`` imports ``BundleStore`` from here).
    """


_USE_REGISTRY_DEFAULT = _UseRegistryDefault()


@dataclasses.dataclass(frozen=True)
class RevisionClosure:
    """The result of resolving one revision's complete component graph (M16).

    ``ancestors`` is the revision's own ancestor closure (keyed by
    revision ID, including ``target`` itself); ``components`` is every
    component reachable from any ancestor's ``component_ids``, resolved
    and fail-closed-checked by :meth:`BundleStore._validate_structural_closure`.
    Returned by :meth:`BundleStore.resolve_revision_closure`, the shared
    entry point used by both ``update_head`` and the document projection.
    """

    target: RevisionRecord
    ancestors: Mapping[RevisionId, RevisionRecord]
    components: Mapping[ComponentId, ComponentRecord]


class BundleStoreError(TranscriptError):
    """Base class for every error `BundleStore` raises."""


class BundleAlreadyExistsError(BundleStoreError):
    pass


class NotABundleError(BundleStoreError):
    """``root`` has no ``manifest.json``: it is not (yet) a bundle."""


class InvalidIdError(BundleStoreError):
    """An ID argument at a store boundary is malformed.

    Raised before the ID is ever used to build a filesystem path, so a
    caller-supplied ID that isn't a syntactically valid ``<prefix>_<uuid7>``
    (e.g. a path-traversal string) never reaches ``open``/``read_text``.
    """


class UnknownRevisionError(BundleStoreError):
    pass


class UnknownArtefactError(BundleStoreError):
    pass


class UnknownRunError(BundleStoreError):
    pass


class UnknownSourceError(BundleStoreError):
    pass


class UnknownComponentError(BundleStoreError):
    pass


class ArtefactBlobHashMismatchError(BundleStoreError):
    """An artefact's stored bytes no longer hash to the value its record
    declares. Source evidence is immutable (M16); reading it anyway would
    let edited bytes flow into a transform as if they were the acquired
    evidence."""


class UnknownReviewError(BundleStoreError):
    pass


class UnknownRenderError(BundleStoreError):
    pass


class RenderOutputHashMismatchError(BundleStoreError):
    """A render's declared ``output_sha256`` does not match the bytes it
    was stored with, or the blob those bytes live in has since changed.
    M17 makes a render's identity complete precisely so this is
    detectable; returning the bytes anyway would let an edited blob be
    applied to a note under a trusted render ID.
    """


class UnknownComponentKindError(BundleStoreError):
    """A persisted component's ``component_kind`` is outside v1's closed
    set (M1). Fail-closed: an unrecognised kind is refused rather than
    silently skipped or guessed at during closure resolution.
    """


class InvalidComponentFileError(BundleStoreError):
    """A component file on disk is not trustworthy content.

    Two distinct causes, both distinct from :class:`UnknownComponentKindError`
    (a syntactically valid record with an unrecognised ``component_kind``):
    a file that fails validation outright (corrupt JSON, missing required
    fields) -- raised by :meth:`BundleStore._iter_components`, naming the
    offending path, so one bad file fails closed for that file's own
    resolution rather than silently poisoning every future
    :meth:`BundleStore.add_component` dedup scan with an opaque
    ``ValidationError`` -- and a file whose stored ``content_hash`` does
    not match its own recomputed content (MINOR C) -- raised by
    :meth:`BundleStore.load_component`, naming the path, so a hand-edited
    or corrupted component is never trusted just because it parses.
    """


class InvalidSupersessionError(BundleStoreError):
    """A revision's ``superseded_component_ids`` claim is malformed (M21).

    Raised when a revision supersedes a component ID absent from its own
    ancestors' closure, or supersedes an ID it also carries itself.
    """


class ArtefactMetadataConflictError(BundleStoreError):
    """Same acquisition identity (hash, source, locator), different metadata.

    Ingestion is idempotent on content hash + source + locator (M16), but
    that idempotency must never silently paper over a caller re-ingesting
    the same acquisition with a *different* kind/producer/derived_from —
    that would return a record whose metadata lies about how it was made.
    """


class RecordIdCollisionError(BundleStoreError):
    """A freshly minted ID collided with an existing record file.

    uuid7 collisions are not a realistic operational concern; this exists
    so the append-only guarantee is enforced (and diagnosable) rather than
    silently overwriting a record if it ever did happen.
    """


class UnresolvedClosureError(BundleStoreError):
    """A revision's dependency closure does not fully resolve (M16)."""


class HeadNotAncestorError(BundleStoreError):
    """The target revision does not build on the current head (M18)."""


class LeaseHeldError(BundleStoreError):
    """A second run tried to acquire a lease already held by a live run."""


class NotLeaseHolderError(BundleStoreError):
    """A run tried to move the head or release a lease it does not hold."""


class RunNotAcquirableError(BundleStoreError):
    """`run_id` does not exist, or is not in a state that may acquire the lease."""


class TakeOverRefusedError(BundleStoreError):
    """A take-over was requested but its precondition (M2) did not hold."""


class InvalidRunTransitionError(BundleStoreError):
    """A run state transition is not on M2's legal-edges table."""


class NextActionPreconditionError(BundleStoreError):
    """`next_action` presence disagrees with M2's rule for the target state."""


class BundleStore:
    """Owns one bundle's directory: ``root`` per M16's layout.

    One instance is scoped to one bundle (``root`` is that bundle's
    directory, e.g. ``_working/transcripts/bundles/<bundle_id>/``); picking
    and creating that directory is a caller/CLI concern, out of scope here.
    """

    def __init__(
        self,
        root: Path,
        *,
        replace: ReplaceFn = os.replace,
        link: LinkFn = os.link,
        validate_capabilities: CapabilityValidator
        | None
        | _UseRegistryDefault = _USE_REGISTRY_DEFAULT,
    ) -> None:
        """``validate_capabilities`` defaults ON (M16/MAJOR 5): every head
        move is gated by the real M4 registry, bound to *this instance's*
        own ``root`` -- never an independently-supplied second root that
        could silently disagree with the store actually performing the
        move. Pass ``None`` for the explicit structural-only escape hatch,
        or a callable to inject a fully custom seam (as the store's own
        tests do to exercise the seam mechanism itself).
        """
        self._root = root
        self._replace = replace
        self._link = link
        if isinstance(validate_capabilities, _UseRegistryDefault):
            self._validate_capabilities = self._default_capability_validator
        elif validate_capabilities is None:
            self._validate_capabilities = _no_op_capability_validator
        else:
            self._validate_capabilities = validate_capabilities

    @property
    def root(self) -> Path:
        return self._root

    def _default_capability_validator(self, revision: RevisionRecord) -> None:
        """Default ``validate_capabilities`` seam: the real M4 registry
        (MAJOR 5), bound to this store's own ``root``.

        Imported lazily -- ``document.py`` imports ``BundleStore`` from
        this module at module level, so this module cannot import
        ``document.py`` at module level too without a cycle; by the time
        any store actually moves a head (this method only ever runs from
        inside :meth:`update_head`), both modules have long finished
        loading, so the import here is cheap and safe. Always passes
        ``self`` (never a fresh, independently-rooted store) so the
        validated root can never disagree with the store doing the move.
        """
        from .document import capability_validating_seam

        capability_validating_seam(self)(revision)

    # -- paths ------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self._root / "manifest.json"

    def _lease_path(self) -> Path:
        return self._root / "runs" / "ACTIVE"

    def _blob_path(self, sha256: str) -> Path:
        return self._root / "blobs" / sha256

    def _validated_record_path(
        self, subdirectory: str, adapter: TypeAdapter[str], value: str, *, kind: str
    ) -> Path:
        """Validate ``value`` against its ID pattern, then resolve a path
        for it under ``subdirectory`` -- refusing if that path would
        resolve outside ``root`` (fix for the vault-escape class of bug:
        a caller-supplied ID is untrusted input until it has passed both
        checks, never just string-concatenated into a filesystem path).
        """
        try:
            validated = adapter.validate_python(value)
        except ValidationError as exc:
            raise InvalidIdError(f"{value!r} is not a valid {kind} ID.") from exc
        path = (self._root / subdirectory / f"{validated}.json").resolve()
        if not path.is_relative_to(self._root.resolve()):
            raise InvalidIdError(
                f"{kind} ID {value!r} resolves outside the bundle root."
            )
        return path

    def _artefact_path(self, artefact_id: ArtefactId) -> Path:
        return self._validated_record_path(
            "artefacts", _ARTEFACT_ID_ADAPTER, artefact_id, kind="artefact"
        )

    def _revision_path(self, revision_id: RevisionId) -> Path:
        return self._validated_record_path(
            "revisions", _REVISION_ID_ADAPTER, revision_id, kind="revision"
        )

    def _run_path(self, run_id: RunId) -> Path:
        return self._validated_record_path("runs", _RUN_ID_ADAPTER, run_id, kind="run")

    def _component_path(self, component_id: ComponentId) -> Path:
        return self._validated_record_path(
            "components", _COMPONENT_ID_ADAPTER, component_id, kind="component"
        )

    def _review_path(self, review_id: ReviewId) -> Path:
        return self._validated_record_path(
            "reviews", _REVIEW_ID_ADAPTER, review_id, kind="review"
        )

    def _render_path(self, render_id: RenderId) -> Path:
        return self._validated_record_path(
            "renders", _RENDER_ID_ADAPTER, render_id, kind="render"
        )

    def _apply_path(self, apply_id: ApplyId) -> Path:
        return self._validated_record_path(
            "applies", _APPLY_ID_ADAPTER, apply_id, kind="apply"
        )

    # -- locking and generic atomic IO ---------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialize one read-modify-write critical section across processes.

        The per-file write primitives below protect against a *crash*
        mid-write; they say nothing about two processes interleaving
        between a read and the write it informs (e.g. two processes both
        reading "no lease held" and both proceeding to acquire, or two
        processes both deciding an artefact needs a fresh record). This is
        a single advisory ``flock`` per bundle for exactly that.

        Reentrant per thread via a module-level depth registry keyed by
        resolved bundle root (so nesting works across distinct BundleStore
        instances for the same bundle): a callback invoked while
        the lock is held (chiefly ``validate_capabilities``, whose whole
        purpose is to host the future capability registry, which *will*
        resolve components through other store methods) can call back into
        another locking method without self-deadlocking. A second
        ``flock(LOCK_EX)`` from a fresh ``open()`` on the *same* process
        would otherwise block forever waiting for a lock this same thread
        already holds -- ``flock`` treats distinct file descriptors
        independently even within one process, so re-opening the lock file
        does not see this thread's own hold as "mine". Only the outermost
        call actually touches the filesystem lock; nested calls just track
        depth. Requires ``root`` to already exist -- refuses with
        `NotABundleError` rather than creating it as a side effect.
        """
        lock_key = str(self._root.resolve())
        depths = _lock_depths()
        depth = depths.get(lock_key, 0)
        if depth > 0:
            depths[lock_key] = depth + 1
            try:
                yield
            finally:
                depths[lock_key] = depth
            return

        if not self._root.is_dir():
            raise NotABundleError(f"{self._root} is not a bundle (no manifest.json).")
        with open(self._root / _LOCK_FILENAME, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            depths[lock_key] = 1
            try:
                yield
            finally:
                depths.pop(lock_key, None)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _fsync_directory(self, directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write_json_atomic(self, path: Path, payload: BaseModel) -> None:
        """Write ``payload`` via temp file + the injectable ``replace`` seam.

        Readers never observe a partially written file: either the old
        content is still there, or the new content is there complete. If
        ``replace`` raises (a real crash, or a test injecting one), the
        stray temp file is cleaned up and the target is left untouched.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        descriptor, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            self._replace(tmp_path, str(path))
            self._fsync_directory(path.parent)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

    def _write_json_exclusive(
        self, path: Path, payload: BaseModel, *, conflict_error: type[BundleStoreError]
    ) -> None:
        """Create ``path`` exclusively *and* atomically; refuse if it exists.

        Writes the full record to a temp file first, then links it into
        place via the injectable ``link`` seam (default ``os.link``) --
        one atomic syscall that also raises ``FileExistsError`` if the
        target already exists. Writing straight to the target with
        ``O_EXCL`` would leave a torn, permanently unwritable file behind
        if the process died mid-write (append-only means the target can
        never be rewritten to fix it); linking a fully-written temp file
        in means ``path`` only ever exists complete -- a crash before the
        link leaves it simply absent, safe to retry.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        descriptor, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                self._link(tmp_path, str(path))
            except FileExistsError as exc:
                raise conflict_error(f"{path} already exists.") from exc
            self._fsync_directory(path.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)

    def _replace_run_fields(
        self, run: RunRecord, *, state: RunState | None = None, **updates: object
    ) -> RunRecord:
        """Re-validate ``run`` with ``state``/``updates`` applied, enforcing
        M2's legal state edges (only relevant when ``state`` changes it).

        ``model_copy(update=...)`` deliberately skips validation, which
        would let a transition silently bypass `RunRecord`'s own
        invariants (e.g. a durable state written with no ``next_action``).
        Round-tripping through `model_validate` re-runs them on every
        transition; the edge-table check below catches the invariant
        `RunRecord` cannot check by itself -- whether *this* state was
        even reachable from the run's current one. ``state`` is a named
        parameter (rather than folded into ``**updates``) purely so its
        type is known here, for the edge-table lookup and error message.
        """
        if state is not None and state != run.state:
            legal = _LEGAL_RUN_STATE_EDGES.get(run.state, frozenset())
            if state not in legal:
                raise InvalidRunTransitionError(
                    f"run {run.run_id} cannot move from state {run.state.value!r} to "
                    f"{state.value!r} (M2 legal edges from here: "
                    f"{sorted(edge.value for edge in legal)})."
                )
        merged: dict[str, object] = {**run.model_dump(mode="python"), **updates}
        if state is not None:
            merged["state"] = state
        return RunRecord.model_validate(merged)

    # -- bundle lifecycle -----------------------------------------------------

    def create_bundle(self) -> BundleManifest:
        """Mint ``bundle_id``/``document_id`` and record their edge (M1, D3).

        Uses the exclusive-write primitive for the initial manifest, so two
        concurrent ``create_bundle()`` calls against the same root cannot
        silently clobber one another into disagreeing about the bundle's
        own identity; the loser gets `BundleAlreadyExistsError`.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        for subdirectory in _SUBDIRECTORIES:
            (self._root / subdirectory).mkdir(parents=True, exist_ok=True)

        manifest = BundleManifest(
            bundle_id=mint_id("bundle"),
            document_id=mint_id("doc"),
            created_at=_utc_now(),
        )
        self._write_json_exclusive(
            self._manifest_path(), manifest, conflict_error=BundleAlreadyExistsError
        )
        return manifest

    def load_manifest(self) -> BundleManifest:
        try:
            text = self._manifest_path().read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise NotABundleError(
                f"{self._root} is not a bundle (no manifest.json)."
            ) from exc
        return BundleManifest.model_validate_json(text)

    def register_source(
        self, *, association: SourceAssociation, evidence: str
    ) -> SourceMembershipRecord:
        """Record a source membership (M3) and append it to the manifest."""
        with self._locked():
            manifest = self.load_manifest()
            membership = SourceMembershipRecord(
                source_id=mint_id("source"),
                bundle_id=manifest.bundle_id,
                association=association,
                evidence=evidence,
            )
            updated = manifest.model_copy(
                update={
                    "source_memberships": manifest.source_memberships + (membership,)
                }
            )
            self._write_json_atomic(self._manifest_path(), updated)
            return membership

    # -- artefacts --------------------------------------------------------

    def _iter_artefacts(self) -> Iterator[ArtefactRecord]:
        artefacts_dir = self._root / "artefacts"
        if not artefacts_dir.exists():
            return
        for path in sorted(artefacts_dir.glob("*.json")):
            yield ArtefactRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def _find_existing_artefact(
        self,
        *,
        sha256: str,
        source_id: SourceId,
        acquisition_locator: str,
        kind: str,
        producer: str,
        derived_from: tuple[ArtefactId, ...],
    ) -> ArtefactRecord | None:
        for record in self._iter_artefacts():
            if not (
                record.sha256 == sha256
                and record.source_id == source_id
                and record.acquisition_locator == acquisition_locator
            ):
                continue
            if (
                record.kind != kind
                or record.producer != producer
                or record.derived_from != derived_from
            ):
                raise ArtefactMetadataConflictError(
                    f"an artefact for sha256={sha256} source={source_id} "
                    f"locator={acquisition_locator!r} already exists "
                    f"(kind={record.kind!r}, producer={record.producer!r}, "
                    f"derived_from={record.derived_from!r}) with different metadata "
                    f"than requested (kind={kind!r}, producer={producer!r}, "
                    f"derived_from={derived_from!r})."
                )
            return record
        return None

    def load_artefact(self, artefact_id: ArtefactId) -> ArtefactRecord:
        path = self._artefact_path(artefact_id)
        if not path.exists():
            raise UnknownArtefactError(f"artefact {artefact_id} does not exist.")
        return ArtefactRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def load_artefact_bytes(self, artefact_id: ArtefactId) -> bytes:
        """The immutable bytes behind one artefact, hash-verified on read.

        Every downstream transform (normalisation reading ``asr.json``,
        rendering reading a note snapshot) goes through this rather than
        touching ``blobs/`` directly, so a corrupted or hand-edited blob
        fails closed at one place instead of being silently parsed as
        evidence.
        """
        record = self.load_artefact(artefact_id)
        content = self._blob_path(record.sha256).read_bytes()
        actual = _sha256_hex(content)
        if actual != record.sha256:
            raise ArtefactBlobHashMismatchError(
                f"artefact {artefact_id}'s blob hashes to {actual}, not the "
                f"{record.sha256} its record declares -- the blob has been "
                "modified or corrupted."
            )
        return content

    def ingest_artefact(
        self,
        *,
        source_id: SourceId,
        content: bytes,
        kind: str,
        producer: str,
        acquisition_locator: str,
        derived_from: tuple[ArtefactId, ...] = (),
    ) -> ArtefactRecord:
        """Content-addressed ingest (M16).

        Idempotent on the full acquisition identity — content hash, source,
        *and* locator: re-ingesting identical bytes from the same source via
        the same locator (a repeated command) is a no-op returning the
        existing record, *provided* the metadata (kind/producer/
        derived_from) also matches — a metadata mismatch on the same triple
        is a caller bug and is rejected outright rather than silently
        served from the mismatched existing record. The same bytes from a
        different source, or via a different locator on the same source,
        are evidence of a distinct acquisition and get a new artefact
        record (M1 §3.2), even though the underlying bytes dedupe at blob
        storage. The head never moves as a side effect of this (M12).

        The dedup decision (does a record for this identity already exist,
        and does its metadata match) is a read-modify-write, not a pure
        append -- deciding "no existing record" and then minting one must
        be one atomic step across processes, or concurrent ingests of the
        same identity mint duplicate records (M16 requires exactly one).
        It runs under :meth:`_locked`. The content-addressed blob write
        itself stays outside the lock: two processes racing to write the
        same ``sha256`` are writing byte-identical content to the same
        path, so the race is harmless (whichever ``replace`` lands last,
        the bytes are the same either way).
        """
        manifest = self.load_manifest()
        known_sources = {
            membership.source_id for membership in manifest.source_memberships
        }
        if source_id not in known_sources:
            raise UnknownSourceError(
                f"source {source_id} has no membership record on this bundle; "
                "call register_source first."
            )
        missing_inputs = [
            artefact_id
            for artefact_id in derived_from
            if not self._artefact_path(artefact_id).exists()
        ]
        if missing_inputs:
            raise UnknownArtefactError(
                f"derived_from references unresolved artefact(s): {missing_inputs}"
            )

        sha256 = _sha256_hex(content)
        blob_path = self._blob_path(sha256)
        if not blob_path.exists():
            self._write_blob(blob_path, content)

        with self._locked():
            existing = self._find_existing_artefact(
                sha256=sha256,
                source_id=source_id,
                acquisition_locator=acquisition_locator,
                kind=kind,
                producer=producer,
                derived_from=derived_from,
            )
            if existing is not None:
                return existing

            record = ArtefactRecord(
                artefact_id=mint_id("artefact"),
                bundle_id=manifest.bundle_id,
                source_id=source_id,
                acquisition_locator=acquisition_locator,
                sha256=sha256,
                blob_ref=f"blobs/{sha256}",
                kind=kind,
                producer=producer,
                derived_from=derived_from,
                created_at=_utc_now(),
            )
            self._write_json_exclusive(
                self._artefact_path(record.artefact_id),
                record,
                conflict_error=RecordIdCollisionError,
            )
            return record

    def _write_blob(self, blob_path: Path, content: bytes) -> None:
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, tmp_path = tempfile.mkstemp(
            dir=blob_path.parent, prefix=f".{blob_path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            self._replace(tmp_path, str(blob_path))
            self._fsync_directory(blob_path.parent)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

    def _candidate_artefact_ids(self) -> tuple[ArtefactId, ...]:
        return tuple(record.artefact_id for record in self._iter_artefacts())

    # -- components -------------------------------------------------------

    def _iter_components(self) -> Iterator[ComponentRecord]:
        """Every stored component, or a typed, file-naming error.

        A single unparseable file must not "poison" every future
        :meth:`add_component` dedup scan with an opaque, path-less
        ``pydantic.ValidationError`` -- wrapping it here, naming the exact
        file, keeps the failure diagnosable (MINOR 13).
        """
        components_dir = self._root / "components"
        if not components_dir.exists():
            return
        for path in sorted(components_dir.glob("*.json")):
            try:
                yield _COMPONENT_RECORD_ADAPTER.validate_json(
                    path.read_text(encoding="utf-8")
                )
            except ValidationError as exc:
                raise InvalidComponentFileError(
                    f"component file {path} does not parse as a known component: {exc}"
                ) from exc

    def iter_components(self) -> Iterator[ComponentRecord]:
        """Every stored component, whether or not a revision references it.

        The public counterpart to the private dedup scan: ``transform
        assemble`` has to find the components an adapter produced at
        ingest time (M3: ingestion creates *candidates*), and those are
        by definition not yet in any revision's closure -- so the document
        projection cannot see them and there would otherwise be no way to
        bring them in.
        """
        yield from self._iter_components()

    def _find_existing_component_by_hash(
        self, content_hash: str
    ) -> ComponentRecord | None:
        """Dedup lookup by the *persisted* ``content_hash`` field.

        Compares against each stored record's own already-computed hash
        rather than re-hashing every existing component's full canonical
        JSON on every call -- an O(n) directory scan is unavoidable
        without a secondary hash index, but re-hashing every candidate on
        every add is pure waste this avoids (MINOR 13).
        """
        for record in self._iter_components():
            if record.content_hash == content_hash:
                return record
        return None

    def load_component(self, component_id: ComponentId) -> ComponentRecord:
        """Load a stored component, typed by its kind (M1), with its
        content_hash verified against its own recomputed content (MINOR C).

        Raises :class:`UnknownComponentKindError` -- not a bare
        ``ValidationError`` -- when the persisted ``component_kind`` does
        not match any member of the closed v1 set (:class:`.components.
        ComponentKind`): fail-closed rather than guessing at a shape for
        data this code does not recognise. Raises
        :class:`InvalidComponentFileError`, naming the file, when the
        stored ``content_hash`` does not match the hash recomputed from
        the record's own content -- a hand-edited or otherwise tampered
        component file must never be trusted just because it happens to
        parse. This is the read path every closure resolution goes
        through (:meth:`_resolve_component_closure`), so every consumer
        of a document projection inherits this guarantee.
        """
        path = self._component_path(component_id)
        if not path.exists():
            raise UnknownComponentError(f"component {component_id} does not exist.")
        try:
            record = _COMPONENT_RECORD_ADAPTER.validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise UnknownComponentKindError(
                f"component {component_id} does not match any known component kind "
                f"(closed set: {[kind.value for kind in ComponentKind]})."
            ) from exc
        expected_hash = _component_content_hash(record)
        if record.content_hash != expected_hash:
            raise InvalidComponentFileError(
                f"component file {path} has a content_hash that does not match its "
                f"own content (stored {record.content_hash!r}, recomputed "
                f"{expected_hash!r}) -- the file may have been hand-edited or "
                "corrupted."
            )
        return record

    def add_component(self, body: ComponentBody) -> ComponentRecord:
        """Content-identified component storage (M1).

        ``body`` carries no ``component_id``/``created_at`` -- those are
        minted here, never by the caller (M1: "IDs are minted by the store
        at record creation"). For a notes body, every section's
        :data:`.ids.SegmentId` is *also* minted here for the same reason
        (MAJOR 6) -- a caller supplies section content only
        (:class:`.components.NotesSectionBody`), never an ID. Adding a
        body whose content already matches a stored component is
        idempotent and returns the existing record untouched (with its
        original section IDs), mirroring :meth:`ingest_artefact`'s dedup
        shape; the dedup decision is a read-modify-write and runs under
        :meth:`_locked` for the same reason that one does. The content
        hash is computed once and reused for both the dedup lookup and
        (on a miss) the stored record's own ``content_hash`` field.
        """
        with self._locked():
            content_hash = _component_content_hash(body)
            existing = self._find_existing_component_by_hash(content_hash)
            if existing is not None:
                return existing

            record = assemble_component_record(
                body,
                component_id=mint_id("component"),
                content_hash=content_hash,
                created_at=_utc_now(),
                mint_segment_id=lambda: mint_id("seg"),
            )
            self._write_json_exclusive(
                self._component_path(record.component_id),
                record,
                conflict_error=RecordIdCollisionError,
            )
            return record

    # -- revisions ----------------------------------------------------------

    def load_revision(self, revision_id: RevisionId) -> RevisionRecord:
        path = self._revision_path(revision_id)
        if not path.exists():
            raise UnknownRevisionError(f"revision {revision_id} does not exist.")
        return RevisionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def append_revision(
        self,
        *,
        operation: OperationRef,
        parent_revision_ids: tuple[RevisionId, ...] = (),
        artefact_ids: tuple[ArtefactId, ...] = (),
        component_ids: tuple[ComponentId, ...] = (),
        superseded_component_ids: tuple[ComponentId, ...] = (),
    ) -> RevisionRecord:
        """Append one revision to the DAG (M1). Never rewrites an existing one.

        ``superseded_component_ids`` (M21) records this revision's
        corrections: components it removes from the closure it and its
        descendants compute. Only cheap, immediate checks run here
        (parents exist) -- whether the supersession claim is actually
        well-formed against the ancestor closure is validated lazily, at
        closure-resolution time (:meth:`_validate_structural_closure`),
        matching how artefact/component *ref* resolution already works:
        ``append_revision`` never re-walks the whole ancestor DAG itself.
        """
        manifest = self.load_manifest()
        missing_parents = [
            parent_id
            for parent_id in parent_revision_ids
            if not self._revision_path(parent_id).exists()
        ]
        if missing_parents:
            raise UnknownRevisionError(
                f"parent revision(s) not found: {missing_parents}"
            )

        record = RevisionRecord(
            revision_id=mint_id("rev"),
            bundle_id=manifest.bundle_id,
            parent_revision_ids=parent_revision_ids,
            component_ids=component_ids,
            superseded_component_ids=superseded_component_ids,
            artefact_ids=artefact_ids,
            operation=operation,
            created_at=_utc_now(),
        )
        self._write_json_exclusive(
            self._revision_path(record.revision_id),
            record,
            conflict_error=RecordIdCollisionError,
        )
        return record

    def _validate_structural_closure(
        self, revision_id: RevisionId
    ) -> tuple[
        RevisionRecord,
        dict[RevisionId, RevisionRecord],
        dict[ComponentId, ComponentRecord],
    ]:
        """M16/M21: revision exists, every ancestor resolves, every artefact
        and component ref resolves *within the closure*, and every
        supersession claim is well-formed.

        A revision's component graph is ``union(component_ids of self +
        ancestors) - union(superseded_component_ids of self + ancestors)``
        (M21) -- components accumulate across the DAG the same way
        ``artefact_ids`` do, except a later revision may also *retract* an
        earlier one's component by superseding its ID (correcting a
        one-cardinality component without bricking the bundle). Each live
        component is loaded (fail-closed on an unknown kind, M1) and its
        own declared input artefact/component refs
        (:func:`.components.component_input_refs`) must themselves already
        be members of this same closure -- not merely exist somewhere else
        in the bundle. This is the coherent-snapshot rule (F1): a
        component embedded in one document can never silently pull in
        evidence that a different revision assembled.

        Returns the target revision, its full ancestor closure (keyed by
        revision ID, including the target itself), and every resolved,
        still-live component (keyed by component ID).
        """
        target = self.load_revision(revision_id)
        revision_closure: dict[RevisionId, RevisionRecord] = {revision_id: target}
        frontier = [revision_id]
        while frontier:
            current = revision_closure[frontier.pop()]
            for parent_id in current.parent_revision_ids:
                if parent_id in revision_closure:
                    continue
                revision_closure[parent_id] = self.load_revision(parent_id)
                frontier.append(parent_id)

        closure_artefact_ids: set[ArtefactId] = set()
        for revision in revision_closure.values():
            missing_artefacts = [
                artefact_id
                for artefact_id in revision.artefact_ids
                if not self._artefact_path(artefact_id).exists()
            ]
            if missing_artefacts:
                raise UnresolvedClosureError(
                    f"revision {revision.revision_id} references unresolved artefact "
                    f"ID(s): {missing_artefacts}"
                )
            closure_artefact_ids.update(revision.artefact_ids)

        closure_component_ids = self._resolve_live_component_ids(
            revision_id, revision_closure
        )

        resolved_components: dict[ComponentId, ComponentRecord] = {}
        for component_id in closure_component_ids:
            self._resolve_component_closure(
                component_id,
                closure_artefact_ids=closure_artefact_ids,
                closure_component_ids=closure_component_ids,
                resolved=resolved_components,
            )
        return target, revision_closure, resolved_components

    def _ancestor_revision_ids(
        self,
        revision_id: RevisionId,
        revision_closure: Mapping[RevisionId, RevisionRecord],
    ) -> set[RevisionId]:
        """``revision_id``'s transitive ancestors within ``revision_closure``,
        excluding ``revision_id`` itself."""
        ancestors: set[RevisionId] = set()
        frontier = [revision_id]
        while frontier:
            current = revision_closure[frontier.pop()]
            for parent_id in current.parent_revision_ids:
                if parent_id not in ancestors:
                    ancestors.add(parent_id)
                    frontier.append(parent_id)
        return ancestors

    def _ancestors_flat_state(
        self,
        revision_id: RevisionId,
        revision_closure: Mapping[RevisionId, RevisionRecord],
    ) -> tuple[set[ComponentId], set[ComponentId]]:
        """The flat union of ``component_ids``/``superseded_component_ids``
        across ``revision_id``'s own transitive ancestors (M21's
        "ancestors' closure" inputs, scoped to exactly this revision's
        lineage -- never a sibling branch's, and never the whole target's
        closure)."""
        components: set[ComponentId] = set()
        superseded: set[ComponentId] = set()
        for ancestor_id in self._ancestor_revision_ids(revision_id, revision_closure):
            ancestor = revision_closure[ancestor_id]
            components.update(ancestor.component_ids)
            superseded.update(ancestor.superseded_component_ids)
        return components, superseded

    def _resolve_live_component_ids(
        self,
        target_id: RevisionId,
        revision_closure: Mapping[RevisionId, RevisionRecord],
    ) -> set[ComponentId]:
        """M21's flat formula, literally: ``union(component_ids) -
        union(superseded_component_ids)`` over the whole ancestor set --
        plus the per-revision validation M21 requires, each check scoped
        to that revision's own ancestors (not a running set shared across
        sibling branches, and not the whole closure indiscriminately).

        Three passes:

        1. **The flat closure itself.** One pass over every revision in
           ``revision_closure`` accumulates ``all_components`` and
           ``all_superseded``; the result is their set difference. This is
           the literal M21 formula -- order-independent, and correct for a
           diamond where two sibling branches both supersede the *same*
           ID (a legal, unremarkable case under this formula: the ID is
           superseded once, from either branch's perspective, and both
           branches' own new components survive).
        2. **Resurrection guard.** No revision may *carry* (in its own
           ``component_ids``) an ID that any of *its own ancestors*
           superseded -- re-deriving retracted content (a very ordinary
           path: ``add_component``'s content-dedup returns the exact same
           ``component_id`` for byte-identical content) must not silently
           bring it back. Scoped per-revision to that revision's own
           ancestors (:meth:`_ancestors_flat_state`) rather than the
           target's whole closure indiscriminately -- the revision that
           *originally* introduced a component predates any supersession
           of it and must never be flagged for carrying its own original
           contribution, and a sibling branch that never superseded
           anything must never be flagged for a *different* branch's
           supersession.
        3. **Per-revision supersession validity.** Unchanged from before,
           but likewise scoped to each revision's own ancestors: it may
           not supersede an ID absent from its ancestors' closure
           (components minus superseded, restricted to that lineage), and
           it may not both carry and supersede the same ID itself.

        A true multi-parent *merge* revision is handled correctly by all
        three passes (verified by the diamond tests); D3 still defers
        richer branching semantics generally, not this.
        """
        all_components: set[ComponentId] = set()
        all_superseded: set[ComponentId] = set()
        superseded_by: dict[ComponentId, RevisionId] = {}
        for revision in revision_closure.values():
            all_components.update(revision.component_ids)
            for component_id in revision.superseded_component_ids:
                all_superseded.add(component_id)
                superseded_by.setdefault(component_id, revision.revision_id)

        for revision in revision_closure.values():
            _ancestors_components, ancestors_superseded = self._ancestors_flat_state(
                revision.revision_id, revision_closure
            )

            resurrected = set(revision.component_ids) & ancestors_superseded
            if resurrected:
                raise InvalidSupersessionError(
                    f"revision {revision.revision_id} carries component ID(s) "
                    "already superseded by its own ancestor(s): "
                    f"{ {cid: superseded_by[cid] for cid in sorted(resurrected)} }"
                )

            own_superseded = set(revision.superseded_component_ids)
            if not own_superseded:
                continue
            ancestors_closure = _ancestors_components - ancestors_superseded
            unresolved_supersessions = own_superseded - ancestors_closure
            if unresolved_supersessions:
                raise InvalidSupersessionError(
                    f"revision {revision.revision_id} supersedes component ID(s) "
                    f"absent from its ancestors' closure: "
                    f"{sorted(unresolved_supersessions)}"
                )
            self_superseding = own_superseded & set(revision.component_ids)
            if self_superseding:
                raise InvalidSupersessionError(
                    f"revision {revision.revision_id} both carries and supersedes "
                    f"component ID(s): {sorted(self_superseding)}"
                )

        return all_components - all_superseded

    def _resolve_component_closure(
        self,
        component_id: ComponentId,
        *,
        closure_artefact_ids: set[ArtefactId],
        closure_component_ids: set[ComponentId],
        resolved: dict[ComponentId, ComponentRecord],
    ) -> None:
        if component_id in resolved:
            return
        if not self._component_path(component_id).exists():
            # Existence is a closure-resolution concern (like a missing
            # artefact ref above): UnresolvedClosureError, not the more
            # specific UnknownComponentError load_component raises for its
            # own direct callers. An unrecognised *kind* on a component
            # that does exist is still a distinct, fail-closed error --
            # load_component below raises UnknownComponentKindError for it.
            raise UnresolvedClosureError(
                f"revision closure references unresolved component ID: {component_id}"
            )
        record = self.load_component(component_id)
        resolved[component_id] = record

        refs = component_input_refs(record)
        missing_artefact_refs = [
            artefact_id
            for artefact_id in refs.artefact_ids
            if artefact_id not in closure_artefact_ids
        ]
        if missing_artefact_refs:
            raise UnresolvedClosureError(
                f"component {component_id} references artefact ID(s) outside the "
                f"revision's closure: {missing_artefact_refs}"
            )
        missing_component_refs = [
            input_id
            for input_id in refs.component_ids
            if input_id not in closure_component_ids
        ]
        if missing_component_refs:
            raise UnresolvedClosureError(
                f"component {component_id} references component ID(s) outside the "
                f"revision's closure: {missing_component_refs}"
            )
        for input_id in refs.component_ids:
            self._resolve_component_closure(
                input_id,
                closure_artefact_ids=closure_artefact_ids,
                closure_component_ids=closure_component_ids,
                resolved=resolved,
            )

    def resolve_revision_closure(self, revision_id: RevisionId) -> RevisionClosure:
        """Public entry point for :meth:`_validate_structural_closure` (M16).

        The document projection (``document.py``) is the intended caller:
        it needs the same "walk the ancestor DAG, resolve every component,
        fail closed on anything dangling" logic that ``update_head`` uses
        to gate the head pointer, so both share this one implementation
        rather than two copies drifting apart.
        """
        target, ancestors, components = self._validate_structural_closure(revision_id)
        return RevisionClosure(
            target=target, ancestors=ancestors, components=components
        )

    def update_head(self, *, run_id: RunId, revision_id: RevisionId) -> BundleManifest:
        """Move the head, transactionally, iff `run_id` holds the lease (M2, M18).

        Only moves to a revision whose structural closure validates (M16),
        and only forward: the current head (when non-null) must be in the
        target's ancestor closure, so the head can never move backwards or
        sideways to an unrelated root and orphan lineage (M18 — the
        lease-holding run moves the head to the revision *it just
        appended*, which is by definition built on the head it started
        from). Never as a side effect of ingestion (M12).
        """
        with self._locked():
            lease = self.load_lease()
            if lease is None or lease.run_id != run_id:
                holder = lease.run_id if lease is not None else "no active run"
                raise NotLeaseHolderError(
                    f"run {run_id} does not hold the bundle's lease (held by {holder}); "
                    "only the lease-holding run may move the head."
                )

            target, revision_closure, _components = self._validate_structural_closure(
                revision_id
            )
            self._validate_capabilities(target)

            manifest = self.load_manifest()
            if (
                manifest.head_revision_id is not None
                and manifest.head_revision_id not in revision_closure
            ):
                raise HeadNotAncestorError(
                    f"current head {manifest.head_revision_id} is not an ancestor of "
                    f"revision {target.revision_id}; the head only moves forward to a "
                    "revision built on top of it (M18)."
                )

            updated = manifest.model_copy(
                update={"head_revision_id": target.revision_id}
            )
            self._write_json_atomic(self._manifest_path(), updated)
            return updated

    # -- derived-output records: reviews, renders, applies (M8/M13/M16/M17) --

    def add_review(self, record: ReviewRecord) -> ReviewRecord:
        """M16: append one immutable ``reviews/<review_id>.json`` record.

        Exclusive-write, so a second application of the same
        ``review_id`` can never overwrite the first's result -- the
        idempotency path (:meth:`find_review`) is expected to have caught
        it long before, and this is the structural backstop if it did not.
        """
        self._write_json_exclusive(
            self._review_path(record.review_id),
            record,
            conflict_error=RecordIdCollisionError,
        )
        return record

    def load_review(self, review_id: ReviewId) -> ReviewRecord:
        path = self._review_path(review_id)
        if not path.exists():
            raise UnknownReviewError(f"review {review_id} does not exist.")
        return ReviewRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def iter_reviews(self) -> Iterator[ReviewRecord]:
        """Every applied review, oldest first. M8's application registry:
        both the idempotent-re-apply check and the "a second, differing
        review set against the same input revision is rejected" rule read
        this, rather than each deriving applied-ness from revision
        archaeology.
        """
        directory = self._root / "reviews"
        if not directory.is_dir():
            return
        records = [
            ReviewRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("review_*.json"))
        ]
        yield from sorted(records, key=lambda record: record.created_at)

    def find_review(self, review_id: ReviewId) -> ReviewRecord | None:
        path = self._review_path(review_id)
        if not path.exists():
            return None
        return ReviewRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def add_render(self, record: RenderRecord, *, output: bytes) -> RenderRecord:
        """M17: store a render's bytes (content-addressed) and its record.

        The bytes go to ``blobs/<sha256>`` -- the same content-addressed
        store artefacts use -- so two byte-identical renders of the same
        revision cost one blob, which is also how the determinism check
        (double render, compare ``output_sha256``) stays cheap. The record
        is written second: a crash between the two leaves an unreferenced
        blob, never a record pointing at bytes that are not there.
        """
        actual = _sha256_hex(output)
        if actual != record.output_sha256:
            raise RenderOutputHashMismatchError(
                f"render {record.render_id} declares output_sha256={record.output_sha256}"
                f" but the supplied bytes hash to {actual}."
            )
        with self._locked():
            self._write_blob(self._blob_path(actual), output)
            self._write_json_exclusive(
                self._render_path(record.render_id),
                record,
                conflict_error=RecordIdCollisionError,
            )
        return record

    def load_render(self, render_id: RenderId) -> RenderRecord:
        path = self._render_path(render_id)
        if not path.exists():
            raise UnknownRenderError(f"render {render_id} does not exist.")
        return RenderRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def load_render_output(self, render_id: RenderId) -> bytes:
        """The exact bytes a render produced, verified against the hash its
        own record declared -- a tampered blob is refused, never returned."""
        record = self.load_render(render_id)
        content = self._blob_path(record.output_sha256).read_bytes()
        actual = _sha256_hex(content)
        if actual != record.output_sha256:
            raise RenderOutputHashMismatchError(
                f"render {render_id}'s stored blob hashes to {actual}, not the "
                f"{record.output_sha256} its record declares."
            )
        return content

    def add_apply(self, record: ApplyRecord) -> ApplyRecord:
        """M13: append one immutable ``applies/<apply_id>.json`` record.

        Every outcome is recorded, including refusals and partial writes --
        an apply that did not happen is evidence too, and the record is
        written whether or not the target was ever touched.
        """
        self._write_json_exclusive(
            self._apply_path(record.apply_id),
            record,
            conflict_error=RecordIdCollisionError,
        )
        return record

    def iter_applies(self) -> Iterator[ApplyRecord]:
        directory = self._root / "applies"
        if not directory.is_dir():
            return
        records = [
            ApplyRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("apply_*.json"))
        ]
        yield from sorted(records, key=lambda record: record.created_at)

    def document_head(self) -> RevisionRecord | NoDocumentYet:
        """M16: the loaded head revision, or the explicit no-document-yet state."""
        manifest = self.load_manifest()
        if manifest.head_revision_id is None:
            return NoDocumentYet(
                bundle_id=manifest.bundle_id,
                candidate_artefact_ids=self._candidate_artefact_ids(),
            )
        return self.load_revision(manifest.head_revision_id)

    # -- runs and leases ------------------------------------------------------

    def load_run(self, run_id: RunId) -> RunRecord:
        path = self._run_path(run_id)
        if not path.exists():
            raise UnknownRunError(f"run {run_id} does not exist.")
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def load_lease(self) -> Lease | None:
        path = self._lease_path()
        if not path.exists():
            return None
        return Lease.model_validate_json(path.read_text(encoding="utf-8"))

    def create_run(
        self,
        *,
        next_action: OperationRef,
        resumes_run_id: RunId | None = None,
        takeover_of_run_id: RunId | None = None,
    ) -> RunRecord:
        """Mint a run record in state ``created`` (M2). Does not acquire the lease.

        ``resumes_run_id``/``takeover_of_run_id``, when given, must name a
        real, existing run — an audit trail that points at nothing is
        worse than no audit trail.
        """
        if resumes_run_id is not None:
            self.load_run(resumes_run_id)
        if takeover_of_run_id is not None:
            self.load_run(takeover_of_run_id)

        with self._locked():
            manifest = self.load_manifest()
            record = RunRecord(
                run_id=mint_id("run"),
                bundle_id=manifest.bundle_id,
                state=RunState.CREATED,
                next_action=next_action,
                resumes_run_id=resumes_run_id,
                takeover_of_run_id=takeover_of_run_id,
                created_at=_utc_now(),
            )
            self._write_json_exclusive(
                self._run_path(record.run_id),
                record,
                conflict_error=RecordIdCollisionError,
            )
            updated_manifest = manifest.model_copy(
                update={"run_ids": manifest.run_ids + (record.run_id,)}
            )
            self._write_json_atomic(self._manifest_path(), updated_manifest)
            return record

    def acquire_lease(
        self, *, run_id: RunId, pid: int, take_over: bool = False
    ) -> Lease:
        """Acquire the bundle's single run lease (M2).

        The run must exist and be in an acquirable state — ``created``,
        one of the durable states, or ``running`` *with no lease currently
        held* — checked *before* the lease is ever touched, so a bad
        ``run_id`` or a ``completed`` run never leaves a stuck lease
        behind. The ``running``-with-no-lease case exists because
        ``running`` is otherwise not acquirable, but a run stuck showing
        it with no lease at all is provably an orphan (holding the lease
        *is* what ``running`` means operationally) — the residual crash
        window inside this very method, between the run-record write and
        the lease write below, produces exactly this state, and it must
        stay recoverable rather than immortally stranded.

        If a lease is already held:

        - and its run's own record is **not** in state ``running`` — that
          is provably a crash remnant (M2: any durable-state entry
          releases the lease, so this can only be a leftover from a run
          that crashed between writing its durable state and unlinking the
          lease). It is cleared automatically and acquisition proceeds —
          no ``take_over`` needed, this is not a genuine contest.
        - and its run **is** ``running`` and its PID is alive — refused,
          naming the holder, regardless of ``take_over``.
        - and its run **is** ``running`` and its PID is dead — the genuine
          take-over case (M2). Refused unless ``take_over=True`` *and*
          `run_id`'s own record declares ``takeover_of_run_id`` equal to
          the stale holder's run ID (the audit trail this take-over
          leaves behind).

        The whole method runs under :meth:`_locked`, so this decision and
        the writes that follow it are one atomic step across processes;
        within that, the run record is written to ``running`` before the
        lease file (a crash in between leaves an orphaned run record but
        *no* lease, which is trivially self-recovering via the
        ``running``-with-no-lease acquirability rule above, rather than a
        lease nobody can clear).
        """
        with self._locked():
            run = self.load_run(run_id)
            current = self.load_lease()

            running_orphan = run.state == RunState.RUNNING and current is None
            if run.state not in _ACQUIRABLE_RUN_STATES and not running_orphan:
                raise RunNotAcquirableError(
                    f"run {run_id} is in state {run.state.value!r}; only "
                    f"{[state.value for state in _ACQUIRABLE_RUN_STATES]} (or "
                    "'running' with no lease currently held, a crash-orphaned run) "
                    "may acquire the lease."
                )
            if take_over and run.takeover_of_run_id is None:
                raise TakeOverRefusedError(
                    f"run {run_id} does not declare takeover_of_run_id; create_run "
                    "must record which run it is taking over before acquiring with "
                    "take_over=True."
                )

            if current is not None:
                holder = self.load_run(current.run_id)
                if holder.state != RunState.RUNNING:
                    # M2: durable-state entry releases the lease. A lease
                    # still pointing at a non-running run is a crash
                    # remnant, not a live contest -- self-heal.
                    self._lease_path().unlink()
                elif not take_over:
                    raise LeaseHeldError(
                        f"bundle is already active under run {current.run_id} "
                        f"(pid {current.pid}); pass take_over=True to steal a dead "
                        "lease."
                    )
                elif run.takeover_of_run_id != current.run_id:
                    raise TakeOverRefusedError(
                        f"run {run_id} declares takeover_of_run_id="
                        f"{run.takeover_of_run_id!r}, but the held lease belongs to "
                        f"run {current.run_id}; refusing a take-over that does not "
                        "name the actual stale holder."
                    )
                elif _pid_is_alive(current.pid):
                    raise TakeOverRefusedError(
                        f"run {current.run_id} (pid {current.pid}) is still alive; "
                        "refusing to steal a live lease."
                    )
                else:
                    self._lease_path().unlink()

            started_at = _utc_now()
            updated_run = self._replace_run_fields(
                run, state=RunState.RUNNING, pid=pid, started_at=started_at
            )
            self._write_json_atomic(self._run_path(run_id), updated_run)

            lease = Lease(run_id=run_id, pid=pid, started_at=started_at)
            self._write_json_exclusive(
                self._lease_path(), lease, conflict_error=LeaseHeldError
            )
            return lease

    def release_lease(
        self,
        *,
        run_id: RunId,
        new_state: RunState,
        next_action: OperationRef | None = None,
    ) -> RunRecord:
        """Enter a durable or terminal state and release the lease (M2).

        ``new_state`` must be one of M2's durable/terminal states, checked
        up front by an explicit whitelist: `_replace_run_fields`'s edge
        table alone is not enough here, because it only checks an edge
        when the target state *differs* from the run's current one --
        releasing into the run's own current state (``running`` ->
        ``running``) trivially satisfies "no transition happened" and
        would sail through un-checked, silently releasing the lease
        without the run ever entering a durable state.

        Only the current lease holder may release it -- and "holder" means
        the actual process that acquired it: a mismatched but *live* PID
        is refused even if it names the right ``run_id``, so a second
        process that merely learned the run ID cannot release a lease out
        from under the process actively using it. A *dead* PID's lease may
        still be released by a different (recovering) process.

        ``next_action`` is a precondition checked up front — every durable
        state must carry one (so a resume replays it exactly), and
        ``completed`` must not carry one (it is terminal) — raising a
        typed error naming the rule, rather than letting a bare
        ``ValidationError`` from `RunRecord`'s own invariant leak out
        uncontextualised.
        """
        if new_state not in (*DURABLE_RUN_STATES, RunState.COMPLETED):
            raise InvalidRunTransitionError(
                "release_lease only accepts a durable or terminal state, got "
                f"{new_state.value!r}."
            )
        if new_state != RunState.COMPLETED and next_action is None:
            raise NextActionPreconditionError(
                f"release_lease(new_state={new_state.value!r}) requires next_action: "
                "M2 durable states must carry one so a later resume replays it exactly."
            )
        if new_state == RunState.COMPLETED and next_action is not None:
            raise NextActionPreconditionError(
                "release_lease(new_state='completed') must not carry a next_action: "
                "'completed' is terminal (M2)."
            )

        with self._locked():
            current = self.load_lease()
            if current is None or current.run_id != run_id:
                holder = current.run_id if current is not None else "no active run"
                raise NotLeaseHolderError(
                    f"run {run_id} does not hold the bundle's lease (held by {holder})."
                )
            if current.pid != os.getpid() and _pid_is_alive(current.pid):
                raise NotLeaseHolderError(
                    f"run {run_id}'s lease is held by live process {current.pid}; "
                    f"only that process (or a later process taking over a dead one) "
                    f"may release it (this process is {os.getpid()})."
                )

            run = self.load_run(run_id)
            updated_run = self._replace_run_fields(
                run, state=new_state, next_action=next_action
            )
            self._write_json_atomic(self._run_path(run_id), updated_run)
            self._lease_path().unlink()
            return updated_run
