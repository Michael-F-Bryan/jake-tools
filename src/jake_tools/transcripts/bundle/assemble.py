"""M18: assembly -- the only operation that moves ingested candidates
into the document.

Ingesting a source or adapting it into components (``adapters.py``)
creates *candidates* only (M3, M12): the bundle head never moves as a
side effect of either. ``assemble()`` is the missing operation the first
Phase 0 draft assumed existed -- it appends one *assembly revision*
naming exactly which artefacts this revision draws in, each with a
non-empty M18 disposition set (``{evidence-only, media,
transcript-candidate, selected-transcript, notes, destination}``), plus
the assembly's rationale, and then moves the bundle head to that
revision via the caller's already-held run lease (M2/M16).

Where the disposition/rationale record lives
----------------------------------------------
``RevisionRecord`` (``store.py``/``records.py``, read-only to this
slice) has no per-artefact disposition field -- only a flat
``artefact_ids`` tuple. Rather than needing a ``records.py`` change,
the richer per-artefact bookkeeping lives in a dedicated
``AssemblyManifestComponent`` (``components.py``, owned by this slice):
:func:`assemble` builds one from ``selections``, adds it via
``store.add_component``, and includes its ID alongside the caller's own
adapter-produced component IDs (turn sets, label sets, participants,
notes, ...) in the appended revision's ``component_ids``.
``RevisionRecord.artefact_ids`` still carries the flat selected-artefact
list the store's own structural closure validation (M16) needs.

Crash-window discipline
------------------------
``assemble()`` is two store writes, not one -- enumerated here so a
resuming run (M2) knows exactly what state it might find:

1. **Before ``add_component`` (the manifest component) returns**:
   nothing durable yet, or the manifest component now exists unreferenced
   by any revision. ``add_component`` is itself content-addressed and
   idempotent (M16): retrying with the same ``selections``/``rationale``
   returns the *same* component record, never a duplicate.
2. **After the manifest component exists, before ``append_revision``
   returns**: the revision does not yet exist. Retry-safe -- nothing
   downstream can observe a component that no revision references yet.
3. **After ``append_revision`` returns, before ``update_head`` runs**:
   the assembly revision exists on disk but is not (yet) the bundle
   head -- a harmless orphan. ``append_revision`` mints a fresh
   ``revision_id`` on every call (no content-based dedup exists for
   revisions, unlike components/artefacts), so a naive retry from
   scratch after a crash here would append a *second*, functionally
   equivalent revision rather than reusing the first -- wasted disk, but
   never a correctness problem: only whichever revision ``update_head``
   actually lands on ever becomes observable via
   ``document_head``/``project_head``.
4. **During/after ``update_head``**: inherited crash safety from
   ``BundleStore._write_json_atomic`` (temp file + atomic rename) --
   the manifest is either still pointing at the old head or fully moved
   to the new one, never torn.

No step here writes outside primitives the store already guarantees are
individually crash-safe (M16); this module adds no new IO primitive of
its own.
"""

from __future__ import annotations

from ..errors import TranscriptError
from .components import ArtefactSelection, AssemblyManifestComponentBody
from .ids import ComponentId, RunId
from .records import NoDocumentYet, OperationRef, RevisionRecord
from .store import BundleStore, UnknownArtefactError


class AssembleError(TranscriptError):
    """Base class for every error :func:`assemble` raises."""


class NoSelectionsError(AssembleError):
    """``assemble()`` was called with zero artefact selections (M18)."""


class UnresolvableSelectionError(AssembleError):
    """An assembly selection names an artefact that does not exist in
    this bundle -- named clearly here rather than surfacing as a bare
    :class:`~.store.UnknownArtefactError` with no assembly context.
    """


def assemble(
    store: BundleStore,
    *,
    run_id: RunId,
    selections: tuple[ArtefactSelection, ...],
    rationale: str,
    component_ids: tuple[ComponentId, ...] = (),
) -> RevisionRecord:
    """Append one assembly revision and move the bundle head to it (M18).

    ``selections`` names every artefact this revision brings into the
    document, each with its own non-empty :class:`~.components.
    Disposition` set (enforced by :class:`~.components.ArtefactSelection`
    itself -- a type-level guarantee, not a check performed here).
    ``component_ids`` are the adapter-produced components (turn sets,
    label sets, participant sets, notes, ...) this assembly also carries;
    the assembly's own :class:`~.components.AssemblyManifestComponent` is
    added automatically and folded in alongside them.

    Builds on whatever the bundle's *current* head is at call time
    (``store.document_head()``): the empty-parent case makes this the
    document's root revision (M18: "the initial assembly revision is the
    document's root revision"); a non-null head makes this assembly a
    child of it, which is how a later re-assembly (D5's competing-
    transcript selection, or simply adding a second source) is expressed.
    Reading the head unlocked before acting on it is safe under M2's
    single-active-run-lease invariant: only the lease holder may ever
    move the head, and ``update_head`` independently re-checks that the
    read head is still an ancestor of the target revision before moving
    -- so a wrong or stale read here is caught, not trusted blindly.

    Raises :class:`NoSelectionsError` if ``selections`` is empty, or
    :class:`UnresolvableSelectionError` naming the first artefact ID that
    does not resolve in this bundle -- both checked up front, before any
    write, so a bad call never reaches the store's own (less specific)
    closure-validation error.
    """
    if not selections:
        raise NoSelectionsError(
            "assemble() requires at least one artefact selection (M18)."
        )
    for selection in selections:
        try:
            store.load_artefact(selection.artefact_id)
        except UnknownArtefactError as exc:
            raise UnresolvableSelectionError(
                f"assembly selection names artefact {selection.artefact_id!r}, "
                "which does not exist in this bundle."
            ) from exc

    manifest_body = AssemblyManifestComponentBody(
        selections=selections, rationale=rationale
    )
    manifest_component = store.add_component(manifest_body)

    head = store.document_head()
    parent_revision_ids = () if isinstance(head, NoDocumentYet) else (head.revision_id,)
    selected_artefact_ids = tuple(selection.artefact_id for selection in selections)

    revision = store.append_revision(
        operation=OperationRef(
            kind="assemble",
            input_ids=selected_artefact_ids,
            rationale=rationale,
        ),
        parent_revision_ids=parent_revision_ids,
        artefact_ids=selected_artefact_ids,
        component_ids=(manifest_component.component_id, *component_ids),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)
    return revision
