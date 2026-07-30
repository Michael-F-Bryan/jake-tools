from __future__ import annotations

import os
from pathlib import Path

import pytest
from fixtures_bundle import registered_source_and_artefact

from jake_tools.transcripts.bundle.assemble import (
    NoSelectionsError,
    UnresolvableSelectionError,
    assemble,
)
from jake_tools.transcripts.bundle.components import (
    ArtefactSelection,
    AssemblyManifestComponent,
    AssemblyManifestComponentBody,
    Disposition,
    NotesComponent,
    NotesComponentBody,
    NotesKind,
    NotesSectionBody,
)
from jake_tools.transcripts.bundle.document import project_head, project_revision
from jake_tools.transcripts.bundle.records import NoDocumentYet, OperationRef
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.store import BundleStore, UnknownArtefactError


def _store_with_run(tmp_path: Path) -> tuple[BundleStore, str]:
    """A fresh bundle plus a run already holding the lease -- the shape
    every ``assemble()`` caller (a recipe, or the M2 run lifecycle) is
    expected to arrive with; assemble() itself has no run/lease opinions.
    """
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    return store, run.run_id


def test_assemble_rejects_zero_selections(tmp_path: Path) -> None:
    store, run_id = _store_with_run(tmp_path)

    with pytest.raises(NoSelectionsError):
        assemble(store, run_id=run_id, selections=(), rationale="test")


def test_assemble_rejects_an_unresolvable_artefact(tmp_path: Path) -> None:
    store, run_id = _store_with_run(tmp_path)
    bogus = "artefact_00000000-0000-7000-8000-000000000000"

    with pytest.raises(UnresolvableSelectionError):
        assemble(
            store,
            run_id=run_id,
            selections=(
                ArtefactSelection(artefact_id=bogus, dispositions=(Disposition.NOTES,)),
            ),
            rationale="test",
        )


def test_assemble_leaves_the_artefact_untouched_on_rejection(tmp_path: Path) -> None:
    """UnknownArtefactError from the store is re-raised as the assembly-
    specific error, not swallowed or converted into a different failure
    mode -- and no revision/component is left behind by the rejected call."""
    store, run_id = _store_with_run(tmp_path)
    bogus = "artefact_00000000-0000-7000-8000-000000000000"

    with pytest.raises(UnresolvableSelectionError) as excinfo:
        assemble(
            store,
            run_id=run_id,
            selections=(
                ArtefactSelection(artefact_id=bogus, dispositions=(Disposition.NOTES,)),
            ),
            rationale="test",
        )
    assert isinstance(excinfo.value.__cause__, UnknownArtefactError)
    assert isinstance(store.document_head(), NoDocumentYet)


def test_assemble_appends_a_root_revision_and_moves_head(tmp_path: Path) -> None:
    store, run_id = _store_with_run(tmp_path)
    artefact = registered_source_and_artefact(store, content=b"hello world")

    revision = assemble(
        store,
        run_id=run_id,
        selections=(
            ArtefactSelection(
                artefact_id=artefact.artefact_id,
                dispositions=(Disposition.SELECTED_TRANSCRIPT,),
            ),
        ),
        rationale="only candidate available",
    )

    assert revision.parent_revision_ids == ()
    manifest = store.load_manifest()
    assert manifest.head_revision_id == revision.revision_id
    assert revision.artefact_ids == (artefact.artefact_id,)


def test_assemble_builds_on_the_current_head(tmp_path: Path) -> None:
    """A second assembly (e.g. adding a second source, or D5's competing-
    transcript re-selection) is a child of whatever the head already is
    -- never a second, disconnected root."""
    store, run_id = _store_with_run(tmp_path)
    first_artefact = registered_source_and_artefact(store, content=b"first")
    second_artefact = registered_source_and_artefact(store, content=b"second")

    first_revision = assemble(
        store,
        run_id=run_id,
        selections=(
            ArtefactSelection(
                artefact_id=first_artefact.artefact_id,
                dispositions=(Disposition.SELECTED_TRANSCRIPT,),
            ),
        ),
        rationale="first assembly",
    )
    second_revision = assemble(
        store,
        run_id=run_id,
        selections=(
            ArtefactSelection(
                artefact_id=second_artefact.artefact_id,
                dispositions=(Disposition.NOTES,),
            ),
        ),
        rationale="second assembly, adds another source",
    )

    assert second_revision.parent_revision_ids == (first_revision.revision_id,)
    manifest = store.load_manifest()
    assert manifest.head_revision_id == second_revision.revision_id


def test_assemble_records_the_manifest_component_with_dispositions(
    tmp_path: Path,
) -> None:
    store, run_id = _store_with_run(tmp_path)
    artefact = registered_source_and_artefact(store, content=b"vtt bytes")

    revision = assemble(
        store,
        run_id=run_id,
        selections=(
            ArtefactSelection(
                artefact_id=artefact.artefact_id,
                dispositions=(Disposition.MEDIA, Disposition.TRANSCRIPT_CANDIDATE),
            ),
        ),
        rationale="teams vtt is the only transcript evidence",
    )

    closure = store.resolve_revision_closure(revision.revision_id)
    manifests = [
        component
        for component in closure.components.values()
        if isinstance(component, AssemblyManifestComponent)
    ]
    assert len(manifests) == 1
    assert manifests[0].rationale == "teams vtt is the only transcript evidence"
    assert manifests[0].selections[0].artefact_id == artefact.artefact_id
    assert manifests[0].selections[0].dispositions == (
        Disposition.MEDIA,
        Disposition.TRANSCRIPT_CANDIDATE,
    )


def test_assemble_folds_in_caller_supplied_component_ids(tmp_path: Path) -> None:
    store, run_id = _store_with_run(tmp_path)
    artefact = registered_source_and_artefact(store, content=b"notes markdown")
    notes = store.add_component(
        NotesComponentBody(
            notes_kind=NotesKind.PROVIDER_SUMMARY,
            source_artefact_id=artefact.artefact_id,
            authored=False,
            sections=(NotesSectionBody(title="Summary", text="text"),),
        )
    )
    assert isinstance(notes, NotesComponent)

    revision = assemble(
        store,
        run_id=run_id,
        selections=(
            ArtefactSelection(
                artefact_id=artefact.artefact_id, dispositions=(Disposition.NOTES,)
            ),
        ),
        rationale="gemini notes only",
        component_ids=(notes.component_id,),
    )

    document = project_revision(store, revision.revision_id)
    assert document.capability(CapabilityKey.NOTES_PROVIDER).status == (
        CapabilityStatus.PRESENT_VALIDATED
    )


def test_assemble_result_projects_head_successfully(tmp_path: Path) -> None:
    store, run_id = _store_with_run(tmp_path)
    artefact = registered_source_and_artefact(store, content=b"a transcript")

    assemble(
        store,
        run_id=run_id,
        selections=(
            ArtefactSelection(
                artefact_id=artefact.artefact_id,
                dispositions=(Disposition.SELECTED_TRANSCRIPT,),
            ),
        ),
        rationale="only candidate available",
    )

    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)


# -- crash-window discipline --------------------------------------------------


def test_assemble_is_safe_to_call_again_after_a_simulated_append_only_crash(
    tmp_path: Path,
) -> None:
    """Simulates the crash window between append_revision returning and
    update_head running (assemble.py's docstring, window 3): a revision
    is durably on disk but never became head. A resumed run calling
    assemble() again (the only thing a real resume *can* do -- there is
    no partial-assemble resume state to detect and continue) must not
    corrupt the bundle -- it produces a second, independent, and this
    time head-advancing revision; the orphan stays a harmless, otherwise
    valid revision that a diagnostic could still inspect."""
    store, run_id = _store_with_run(tmp_path)
    artefact = registered_source_and_artefact(store, content=b"orphan candidate")

    orphan_manifest = store.add_component(
        AssemblyManifestComponentBody(
            selections=(
                ArtefactSelection(
                    artefact_id=artefact.artefact_id,
                    dispositions=(Disposition.SELECTED_TRANSCRIPT,),
                ),
            ),
            rationale="simulated crash before update_head",
        )
    )
    orphan_revision = store.append_revision(
        operation=OperationRef(kind="assemble", input_ids=(artefact.artefact_id,)),
        artefact_ids=(artefact.artefact_id,),
        component_ids=(orphan_manifest.component_id,),
    )
    # Crash simulated here -- update_head was never called for orphan_revision.
    assert isinstance(store.document_head(), NoDocumentYet)

    # Resume: the run retries assemble() from scratch.
    real_revision = assemble(
        store,
        run_id=run_id,
        selections=(
            ArtefactSelection(
                artefact_id=artefact.artefact_id,
                dispositions=(Disposition.SELECTED_TRANSCRIPT,),
            ),
        ),
        rationale="resumed assembly",
    )

    manifest = store.load_manifest()
    assert manifest.head_revision_id == real_revision.revision_id
    assert manifest.head_revision_id != orphan_revision.revision_id
    # The orphan is still a structurally valid revision -- inert, not corrupt.
    orphan_closure = store.resolve_revision_closure(orphan_revision.revision_id)
    assert orphan_closure.target.revision_id == orphan_revision.revision_id


def test_assemble_called_twice_with_identical_inputs_appends_two_revisions(
    tmp_path: Path,
) -> None:
    """Unlike add_component/ingest_artefact, append_revision has no
    content-based dedup (store.py, read-only) -- so assemble() is *not*
    idempotent in that sense by design. Calling it twice with the exact
    same selections is a legitimate two-revision sequence (e.g. a
    deliberate re-confirmation), not silently collapsed into one."""
    store, run_id = _store_with_run(tmp_path)
    artefact = registered_source_and_artefact(store, content=b"reconfirmed")
    selections = (
        ArtefactSelection(
            artefact_id=artefact.artefact_id,
            dispositions=(Disposition.SELECTED_TRANSCRIPT,),
        ),
    )

    first = assemble(
        store, run_id=run_id, selections=selections, rationale="first pass"
    )
    second = assemble(
        store, run_id=run_id, selections=selections, rationale="re-confirmed"
    )

    assert first.revision_id != second.revision_id
    assert second.parent_revision_ids == (first.revision_id,)
    assert store.load_manifest().head_revision_id == second.revision_id
