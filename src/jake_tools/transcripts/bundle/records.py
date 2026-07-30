"""Immutable record types for the bundle store (M1, M2, M3, M16).

Every record here is a frozen Pydantic model. Nothing in this module
performs IO or minting — :class:`.store.BundleStore` owns both; this module
only shapes the data those operations read and write, so the invariants
(non-terminal runs carry a ``next_action``, IDs carry the right prefix)
hold even for records built directly in tests, not just ones that went
through the store.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .ids import (
    ApplyId,
    ArtefactId,
    AttemptId,
    BundleId,
    ComponentId,
    DocumentId,
    RenderId,
    ReviewId,
    RevisionId,
    RunId,
    SourceId,
)

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class SourceAssociation(StrEnum):
    """How a source entered the bundle (M3) — never inferred.

    ``provider-id`` carries an external provider ID (Teams event ID,
    Google event ID, YouTube video ID) as evidence; ``note-embed`` carries
    an embed link plus a note-snapshot artefact ID; ``operator-assertion``
    records the CLI invocation itself — naming inputs on the command line
    *is* the assertion (M3).
    """

    OPERATOR_ASSERTION = "operator-assertion"
    PROVIDER_ID = "provider-id"
    NOTE_EMBED = "note-embed"


class SourceMembershipRecord(BaseModel):
    """M3: the explicit, non-inferred link between a bundle and a source.

    Never associated by title, date, attendee overlap, or filename
    similarity (M3) — ``evidence`` is the only carrier of *why* this source
    belongs to this bundle.
    """

    model_config = ConfigDict(frozen=True)

    source_id: SourceId
    bundle_id: BundleId
    association: SourceAssociation
    evidence: str = Field(min_length=1)


class ArtefactRecord(BaseModel):
    """M1/M16: one acquisition of immutable bytes.

    Identity is ``artefact_id``, minted fresh per acquisition — never the
    content hash. The same bytes acquired from a different source are a
    different artefact record (M1 §3.2); deduplication of the underlying
    bytes happens only at blob storage (``blobs/<sha256>``), never at
    artefact identity.
    """

    model_config = ConfigDict(frozen=True)

    artefact_id: ArtefactId
    bundle_id: BundleId
    source_id: SourceId
    acquisition_locator: str = Field(min_length=1)
    sha256: Sha256Hex
    blob_ref: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    producer: str = Field(min_length=1)
    derived_from: tuple[ArtefactId, ...] = ()
    created_at: datetime


class OperationRef(BaseModel):
    """The operation behind a revision (M1) or a run's ``next_action`` (M2).

    Carries exactly what M2 requires a resumable run to replay: an
    operation kind, the exact input IDs it consumed, and a config hash —
    so ``resume`` executes this record rather than re-planning from
    whatever happens to be on disk.
    """

    model_config = ConfigDict(frozen=True)

    kind: str = Field(min_length=1)
    input_ids: tuple[str, ...] = ()
    config_hash: str | None = None
    rationale: str = ""


class RevisionRecord(BaseModel):
    """M1/M16: one append-only node in the bundle's revision DAG.

    ``component_ids``/``artefact_ids`` are checked for resolution by
    ``BundleStore.update_head`` (M16 structural closure): a revision's
    component graph is the union of every ancestor's ``component_ids``.

    ``superseded_component_ids`` is M21's correction mechanism: a revision
    that carries a replacement component and supersedes the old one's ID
    removes that ID from the closure computed for it and every descendant
    (``union(component_ids) - union(superseded_component_ids)`` over the
    revision plus its ancestors) without ever rewriting or deleting the
    superseded component's own immutable file -- history is never
    destroyed, only excluded from later closures. ``BundleStore``
    validates that a revision never supersedes an ID absent from its own
    ancestors' closure, nor one it also carries itself (M21).
    """

    model_config = ConfigDict(frozen=True)

    revision_id: RevisionId
    bundle_id: BundleId
    parent_revision_ids: tuple[RevisionId, ...] = ()
    component_ids: tuple[ComponentId, ...] = ()
    superseded_component_ids: tuple[ComponentId, ...] = ()
    artefact_ids: tuple[ArtefactId, ...] = ()
    operation: OperationRef
    created_at: datetime


class RunState(StrEnum):
    """M2 run state machine: CREATED -> RUNNING -> one durable/terminal state.

    ``review_required``/``refused``/``failed`` are durable: entering any of
    them releases the lease, and a later run may resume from them.
    ``completed`` is terminal.
    """

    CREATED = "created"
    RUNNING = "running"
    REVIEW_REQUIRED = "review_required"
    REFUSED = "refused"
    FAILED = "failed"
    COMPLETED = "completed"


DURABLE_RUN_STATES = (RunState.REVIEW_REQUIRED, RunState.REFUSED, RunState.FAILED)


class RunRecord(BaseModel):
    """M2: one workflow run over a bundle.

    A run only ever rewrites its own record — never another run's (M2:
    "a run never edits another run's record"); ``BundleStore`` enforces
    this by construction, since every mutator takes the acting ``run_id``
    and writes only that run's file.

    ``resumes_run_id`` records an ordinary resume of a released,
    durable-state run; ``takeover_of_run_id`` records the crash-recovery
    case — a lease held by a dead PID in state ``running`` (M2) — kept as a
    separate field so a reader never has to infer which case produced this
    run from state alone. At most one may be set.
    """

    model_config = ConfigDict(frozen=True)

    run_id: RunId
    bundle_id: BundleId
    state: RunState
    next_action: OperationRef | None
    resumes_run_id: RunId | None = None
    takeover_of_run_id: RunId | None = None
    pid: int | None = None
    created_at: datetime
    started_at: datetime | None = None

    @model_validator(mode="after")
    def _check_state_machine_invariants(self) -> Self:
        if self.state == RunState.COMPLETED:
            if self.next_action is not None:
                raise ValueError(
                    "RunRecord state 'completed' is terminal and must not carry a next_action."
                )
        elif self.next_action is None:
            raise ValueError(
                f"RunRecord state {self.state.value!r} is non-terminal and must carry a "
                "next_action (M2)."
            )
        if self.resumes_run_id is not None and self.takeover_of_run_id is not None:
            raise ValueError("a run cannot both resume and take over another run.")
        return self


class Lease(BaseModel):
    """The ``runs/ACTIVE`` file (M2): run ID, PID, and start time."""

    model_config = ConfigDict(frozen=True)

    run_id: RunId
    pid: int
    started_at: datetime


class AttemptStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AttemptRecord(BaseModel):
    """M16 stub: identity, status, and retained artefacts for one operation
    attempt. Attempts are retained until the bundle is deleted and never
    promoted to a capability (M16); M11's resume keys off them by matching
    input+config hashes. Full attempt semantics (error classes, timing)
    arrive with the worker contract (M11) — out of scope here.
    """

    model_config = ConfigDict(frozen=True)

    attempt_id: AttemptId
    bundle_id: BundleId
    status: AttemptStatus
    retained_artefact_ids: tuple[ArtefactId, ...] = ()
    created_at: datetime


class ReviewRecord(BaseModel):
    """M8/M16: the audit trail for one applied speaker review.

    The *decisions* live in the document (a
    :class:`~.components.SpeakerReviewComponent` inside the applied
    revision's closure), because M17 forbids a renderer from reading
    anything outside the revision it is bound to. This record is the
    other half: who reviewed, which pack bytes they returned, and which
    revision the application produced. It doubles as M8's application
    registry -- re-applying the same ``review_id`` finds this record and
    returns ``result_revision_id`` instead of appending a second,
    duplicate revision.
    """

    model_config = ConfigDict(frozen=True)

    review_id: ReviewId
    bundle_id: BundleId
    input_revision_id: RevisionId
    result_revision_id: RevisionId
    decision_component_id: ComponentId
    pack_schema_version: str = Field(min_length=1)
    pack_sha256: Sha256Hex
    reviewer: str = Field(min_length=1)
    created_at: datetime


class RenderRecord(BaseModel):
    """M17: a render's *complete* identity.

    Every field here participates in "rendering the same revision with
    the same identity fields is byte-deterministic": the bound revision,
    the profile and its version, the renderer implementation version, the
    template hash, and every parameter (locale, timezone, whether the
    transcript body is included). ``input_capability_keys`` records the
    capability closure the render actually consumed, and ``output_sha256``
    names the bytes -- stored as a content-addressed blob, since a render
    is a derived output and never a capability.
    """

    model_config = ConfigDict(frozen=True)

    render_id: RenderId
    bundle_id: BundleId
    revision_id: RevisionId
    profile: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    renderer_version: str = Field(min_length=1)
    template_sha256: Sha256Hex
    parameters: dict[str, str] = Field(default_factory=dict)
    destination_snapshot_artefact_id: ArtefactId | None = None
    input_capability_keys: tuple[str, ...] = ()
    output_sha256: Sha256Hex
    created_at: datetime


class ApplyState(StrEnum):
    """M13's four apply outcomes. ``written-unverified`` is deliberately
    distinct from ``verified``: bytes reached the target but the read-back
    check did not confirm them, which is a different operational
    situation from a clean success and must never be reported as one.
    """

    NOT_WRITTEN = "not-written"
    PARTIALLY_WRITTEN = "partially-written"
    WRITTEN_UNVERIFIED = "written-unverified"
    VERIFIED = "verified"


class ApplyRecord(BaseModel):
    """M13: one attempt to write a render into its destination note."""

    model_config = ConfigDict(frozen=True)

    apply_id: ApplyId
    bundle_id: BundleId
    render_id: RenderId
    revision_id: RevisionId
    target_path: str = Field(min_length=1)
    precondition_sha256: Sha256Hex
    post_write_sha256: Sha256Hex | None = None
    state: ApplyState
    allow_stale_render: bool = False
    adopted_edited_region: bool = False
    migrated_legacy_headings: bool = False
    detail: str = ""
    created_at: datetime


class BundleManifest(BaseModel):
    """``manifest.json`` (M16): small and navigational.

    The document edge (D3, M1) is the pair of ``bundle_id``/``document_id``
    fields recorded together on this one record — ``document_id`` is
    minted separately and never derived from ``bundle_id`` by string
    arithmetic. This is the only file in a bundle that changes after
    creation; every write goes through ``BundleStore``'s atomic
    temp-file-plus-rename helper, and only ``update_head`` additionally
    requires lease ownership and structural closure validation (M16, M2).
    """

    model_config = ConfigDict(frozen=True)

    bundle_id: BundleId
    document_id: DocumentId
    head_revision_id: RevisionId | None = None
    source_memberships: tuple[SourceMembershipRecord, ...] = ()
    run_ids: tuple[RunId, ...] = ()
    created_at: datetime


class NoDocumentYet(BaseModel):
    """M16: the explicit state for a bundle whose head is still null.

    Returned instead of a crash or a fabricated empty document. Names the
    ingested artefacts waiting for an assembly revision (M18) to bring
    them into the document.
    """

    model_config = ConfigDict(frozen=True)

    bundle_id: BundleId
    candidate_artefact_ids: tuple[ArtefactId, ...]
