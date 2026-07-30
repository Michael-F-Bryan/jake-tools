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
  each primitive (between the temp write and the rename/link).
- **Cross-process concurrency** (two processes interleave between a read
  and the write it informs) is handled by :meth:`BundleStore._locked`, a
  single advisory ``flock`` per bundle held for the duration of one
  read-modify-write critical section. Every operation that reads
  ``manifest.json`` (or the lease) and then writes back a decision based on
  that read — ``register_source``, ``create_run``'s manifest update,
  ``update_head``, ``acquire_lease``, ``release_lease`` — holds this lock
  for its whole critical section. Pure append-only writes (fresh artefacts,
  revisions) do not need it: exclusive creation is already race-safe by
  construction.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..errors import TranscriptError
from .ids import ArtefactId, ComponentId, RevisionId, RunId, SourceId, mint_id
from .records import (
    DURABLE_RUN_STATES,
    ArtefactRecord,
    BundleManifest,
    Lease,
    NoDocumentYet,
    OperationRef,
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
    "runs",
    "attempts",
    "reviews",
    "renders",
    "applies",
)

_LOCK_FILENAME = ".store.lock"

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

_ARTEFACT_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(ArtefactId)
_REVISION_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(RevisionId)
_RUN_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(RunId)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    """Default ``validate_capabilities`` seam (M16).

    Structural closure — revision/ancestor/artefact existence — is always
    checked by :meth:`BundleStore.update_head`. This seam is where semantic
    capability validation plugs in once the capability registry (M4)
    exists; until then it is a deliberate no-op, not a stand-in
    implementation of the registry.
    """


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
        validate_capabilities: CapabilityValidator = _no_op_capability_validator,
    ) -> None:
        self._root = root
        self._replace = replace
        self._link = link
        self._validate_capabilities = validate_capabilities

    @property
    def root(self) -> Path:
        return self._root

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

    # -- locking and generic atomic IO ---------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialize one read-modify-write critical section across processes.

        The per-file write primitives below protect against a *crash*
        mid-write; they say nothing about two processes interleaving
        between a read and the write it informs (e.g. two processes both
        reading "no lease held" and both proceeding to acquire). This is a
        single advisory ``flock`` per bundle for exactly that.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        with open(self._root / _LOCK_FILENAME, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
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

        blob_path = self._blob_path(sha256)
        if not blob_path.exists():
            self._write_blob(blob_path, content)

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
    ) -> RevisionRecord:
        """Append one revision to the DAG (M1). Never rewrites an existing one."""
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
    ) -> tuple[RevisionRecord, dict[RevisionId, RevisionRecord]]:
        """M16: revision exists, every ancestor resolves, every artefact ref resolves.

        Component refs are not resolvable yet — no component storage
        exists in this phase (nothing mints components until a later
        task) — so a revision carrying any is rejected outright rather
        than silently accepted or half-checked; full capability validation
        (of which component resolution is a part) is the injected
        ``validate_capabilities`` seam's job once the registry exists.

        Returns the target revision plus its full ancestor closure (keyed
        by revision ID, including the target itself), so callers can also
        check ancestry membership without re-walking the DAG.
        """
        target = self.load_revision(revision_id)
        closure: dict[RevisionId, RevisionRecord] = {revision_id: target}
        frontier = [revision_id]
        while frontier:
            current = closure[frontier.pop()]
            for parent_id in current.parent_revision_ids:
                if parent_id in closure:
                    continue
                closure[parent_id] = self.load_revision(parent_id)
                frontier.append(parent_id)

        for revision in closure.values():
            if revision.component_ids:
                raise UnresolvedClosureError(
                    f"revision {revision.revision_id} references component ID(s) "
                    f"{revision.component_ids!r}, but component storage does not exist "
                    "yet (component ref resolution arrives with the capability registry)."
                )
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
        return target, closure

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

            target, closure = self._validate_structural_closure(revision_id)
            self._validate_capabilities(target)

            manifest = self.load_manifest()
            if (
                manifest.head_revision_id is not None
                and manifest.head_revision_id not in closure
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

        The run must exist and be in an acquirable state — ``created`` or
        one of the durable states — checked *before* the lease is ever
        touched, so a bad ``run_id`` or a ``completed`` run never leaves a
        stuck lease behind. If a lease is already held:

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
        *no* lease, which is trivially self-recovering, rather than a
        lease nobody can clear).
        """
        with self._locked():
            run = self.load_run(run_id)
            if run.state not in _ACQUIRABLE_RUN_STATES:
                raise RunNotAcquirableError(
                    f"run {run_id} is in state {run.state.value!r}; only "
                    f"{[state.value for state in _ACQUIRABLE_RUN_STATES]} may "
                    "acquire the lease."
                )
            if take_over and run.takeover_of_run_id is None:
                raise TakeOverRefusedError(
                    f"run {run_id} does not declare takeover_of_run_id; create_run "
                    "must record which run it is taking over before acquiring with "
                    "take_over=True."
                )

            current = self.load_lease()
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

        Only the current lease holder may release it. ``next_action`` is a
        precondition checked up front — every durable state must carry one
        (so a resume replays it exactly), and ``completed`` must not carry
        one (it is terminal) — raising a typed error naming the rule,
        rather than letting a bare ``ValidationError`` from `RunRecord`'s
        own invariant leak out uncontextualised. Which *states* are legal
        targets at all is enforced generically by `_replace_run_fields`'s
        edge table.
        """
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

            run = self.load_run(run_id)
            updated_run = self._replace_run_fields(
                run, state=new_state, next_action=next_action
            )
            self._write_json_atomic(self._run_path(run_id), updated_run)
            self._lease_path().unlink()
            return updated_run
