"""Filesystem-backed bundle store (M16).

``BundleStore`` owns one bundle's directory end to end: minting every ID a
bundle ever sees, the content-addressed blob store, the append-only
artefact/revision/run/attempt records, the transactional ``manifest.json``
head pointer, and the single-active-run lease. Every write goes through one
of two primitives:

- :meth:`BundleStore._write_json_exclusive` — ``O_CREAT | O_EXCL``, for
  records that are minted once and never rewritten (artefacts, revisions,
  freshly-created runs). The OS refuses the write outright if the path
  already exists, so "append-only" is enforced by construction rather than
  a check-then-write race.
- :meth:`BundleStore._write_json_atomic` — temp file + ``replace``, for the
  two files that legitimately change after creation: ``manifest.json`` and
  a run's own record as it moves through its state machine. The ``replace``
  callable is an injectable seam (defaults to ``os.replace``) so tests can
  simulate a crash between the temp write and the rename.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

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


class UnknownRevisionError(BundleStoreError):
    pass


class UnknownRunError(BundleStoreError):
    pass


class UnknownSourceError(BundleStoreError):
    pass


class RecordIdCollisionError(BundleStoreError):
    """A freshly minted ID collided with an existing record file.

    uuid7 collisions are not a realistic operational concern; this exists
    so the append-only guarantee is enforced (and diagnosable) rather than
    silently overwriting a record if it ever did happen.
    """


class UnresolvedClosureError(BundleStoreError):
    """A revision's dependency closure does not fully resolve (M16)."""


class LeaseHeldError(BundleStoreError):
    """A second run tried to acquire a lease already held by a live run."""


class NotLeaseHolderError(BundleStoreError):
    """A run tried to move the head or release a lease it does not hold."""


class TakeOverRefusedError(BundleStoreError):
    """A take-over was requested but its precondition (M2) did not hold."""


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
        validate_capabilities: CapabilityValidator = _no_op_capability_validator,
    ) -> None:
        self._root = root
        self._replace = replace
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

    def _artefact_path(self, artefact_id: ArtefactId) -> Path:
        return self._root / "artefacts" / f"{artefact_id}.json"

    def _revision_path(self, revision_id: RevisionId) -> Path:
        return self._root / "revisions" / f"{revision_id}.json"

    def _run_path(self, run_id: RunId) -> Path:
        return self._root / "runs" / f"{run_id}.json"

    # -- generic atomic IO --------------------------------------------------

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
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

    def _write_json_exclusive(
        self, path: Path, payload: BaseModel, *, conflict_error: type[BundleStoreError]
    ) -> None:
        """Create ``path`` exclusively; refuse if it already exists.

        Uses ``O_CREAT | O_EXCL`` directly against the target path (no temp
        file) so "does this record already exist" is answered atomically by
        the OS, not by a check-then-write race.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise conflict_error(f"{path} already exists.") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def _replace_run_fields(self, run: RunRecord, **updates: object) -> RunRecord:
        """Re-validate ``run`` with ``updates`` applied.

        ``model_copy(update=...)`` deliberately skips validation, which
        would let a transition silently bypass `RunRecord`'s state-machine
        invariants (e.g. a durable state written with no ``next_action`).
        Round-tripping through `model_validate` re-runs them on every
        transition.
        """
        return RunRecord.model_validate({**run.model_dump(mode="python"), **updates})

    # -- bundle lifecycle -----------------------------------------------------

    def create_bundle(self) -> BundleManifest:
        """Mint ``bundle_id``/``document_id`` and record their edge (M1, D3)."""
        if self._manifest_path().exists():
            raise BundleAlreadyExistsError(
                f"a bundle manifest already exists at {self._manifest_path()}."
            )
        self._root.mkdir(parents=True, exist_ok=True)
        for subdirectory in _SUBDIRECTORIES:
            (self._root / subdirectory).mkdir(parents=True, exist_ok=True)

        manifest = BundleManifest(
            bundle_id=mint_id("bundle"),
            document_id=mint_id("doc"),
            created_at=_utc_now(),
        )
        self._write_json_atomic(self._manifest_path(), manifest)
        return manifest

    def load_manifest(self) -> BundleManifest:
        return BundleManifest.model_validate_json(
            self._manifest_path().read_text(encoding="utf-8")
        )

    def register_source(
        self, *, association: SourceAssociation, evidence: str
    ) -> SourceMembershipRecord:
        """Record a source membership (M3) and append it to the manifest."""
        manifest = self.load_manifest()
        membership = SourceMembershipRecord(
            source_id=mint_id("source"),
            bundle_id=manifest.bundle_id,
            association=association,
            evidence=evidence,
        )
        updated = manifest.model_copy(
            update={"source_memberships": manifest.source_memberships + (membership,)}
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
        self, *, sha256: str, source_id: SourceId, acquisition_locator: str
    ) -> ArtefactRecord | None:
        for record in self._iter_artefacts():
            if (
                record.sha256 == sha256
                and record.source_id == source_id
                and record.acquisition_locator == acquisition_locator
            ):
                return record
        return None

    def load_artefact(self, artefact_id: ArtefactId) -> ArtefactRecord:
        path = self._artefact_path(artefact_id)
        if not path.exists():
            raise UnknownRevisionError(f"artefact {artefact_id} does not exist.")
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
        existing record; the same bytes from a different source, or via a
        different locator on the same source, are evidence of a distinct
        acquisition and get a new artefact record (M1 §3.2), even though
        the underlying bytes dedupe at blob storage. The head never moves as
        a side effect of this (M12).
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
            raise UnknownRevisionError(
                f"derived_from references unresolved artefact(s): {missing_inputs}"
            )

        sha256 = _sha256_hex(content)
        existing = self._find_existing_artefact(
            sha256=sha256, source_id=source_id, acquisition_locator=acquisition_locator
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

    def _validate_structural_closure(self, revision_id: RevisionId) -> RevisionRecord:
        """M16: revision exists, every ancestor resolves, every artefact ref resolves.

        Component refs are not resolvable yet — no component storage
        exists in this phase (nothing mints components until a later
        task) — so a revision carrying any is rejected outright rather
        than silently accepted or half-checked; full capability validation
        (of which component resolution is a part) is the injected
        ``validate_capabilities`` seam's job once the registry exists.
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
        return target

    def update_head(self, *, run_id: RunId, revision_id: RevisionId) -> BundleManifest:
        """Move the head, transactionally, iff `run_id` holds the lease (M2, M18).

        Only moves to a revision whose structural closure validates (M16);
        never as a side effect of ingestion (M12).
        """
        lease = self.load_lease()
        if lease is None or lease.run_id != run_id:
            holder = lease.run_id if lease is not None else "no active run"
            raise NotLeaseHolderError(
                f"run {run_id} does not hold the bundle's lease (held by {holder}); "
                "only the lease-holding run may move the head."
            )

        target = self._validate_structural_closure(revision_id)
        self._validate_capabilities(target)

        manifest = self.load_manifest()
        updated = manifest.model_copy(update={"head_revision_id": target.revision_id})
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
        """Mint a run record in state ``created`` (M2). Does not acquire the lease."""
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
            self._run_path(record.run_id), record, conflict_error=RecordIdCollisionError
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

        Refuses if another run already holds it, naming the holder, unless
        ``take_over=True`` *and* the held lease is for a dead PID whose run
        is still in state ``running`` — the only case M2 allows a take-over
        (an ordinary resume of a released, durable-state run does not need
        ``take_over``, since ``release_lease`` already removed the lease).
        """
        current = self.load_lease()
        if current is not None:
            if not take_over:
                raise LeaseHeldError(
                    f"bundle is already active under run {current.run_id} "
                    f"(pid {current.pid}); pass take_over=True to steal a dead lease."
                )
            holder = self.load_run(current.run_id)
            if holder.state != RunState.RUNNING:
                raise TakeOverRefusedError(
                    f"run {current.run_id} is in state {holder.state.value!r}, not "
                    "'running'; there is nothing to take over."
                )
            if _pid_is_alive(current.pid):
                raise TakeOverRefusedError(
                    f"run {current.run_id} (pid {current.pid}) is still alive; "
                    "refusing to steal a live lease."
                )

        lease = Lease(run_id=run_id, pid=pid, started_at=_utc_now())
        self._write_json_atomic(self._lease_path(), lease)

        run = self.load_run(run_id)
        updated_run = self._replace_run_fields(
            run, state=RunState.RUNNING, pid=pid, started_at=lease.started_at
        )
        self._write_json_atomic(self._run_path(run_id), updated_run)
        return lease

    def release_lease(
        self,
        *,
        run_id: RunId,
        new_state: RunState,
        next_action: OperationRef | None = None,
    ) -> RunRecord:
        """Enter a durable or terminal state and release the lease (M2).

        Only the current lease holder may release it; the caller must
        supply ``new_state`` in {``review_required``, ``refused``,
        ``failed``, ``completed``} — the only states `release_lease`
        transitions into.
        """
        if new_state not in (*DURABLE_RUN_STATES, RunState.COMPLETED):
            raise ValueError(
                f"release_lease only accepts a durable or terminal state, got {new_state!r}."
            )
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
