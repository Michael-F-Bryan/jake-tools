"""Shared, tiny test helpers for the Phase 2 bundle-store test suite.

Not a ``test_*.py`` module -- pytest does not collect it as a test file;
it exists purely to be imported by the ones that do, so the "register a
source and ingest one artefact" boilerplate (identical across the
assemble, adapter, and corpus-canary tests) has exactly one home.
"""

from __future__ import annotations

from jake_tools.transcripts.bundle.records import ArtefactRecord, SourceAssociation
from jake_tools.transcripts.bundle.store import BundleStore


def registered_source_and_artefact(
    store: BundleStore,
    *,
    content: bytes,
    kind: str = "text",
    producer: str = "test",
    acquisition_locator: str = "test:inline",
) -> ArtefactRecord:
    """Register an operator-asserted source membership and ingest one
    artefact under it -- the minimum a test needs before it can exercise
    anything that consumes an ``artefact_id`` (adapters, ``assemble()``).
    """
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence="test fixture",
    )
    return store.ingest_artefact(
        source_id=membership.source_id,
        content=content,
        kind=kind,
        producer=producer,
        acquisition_locator=acquisition_locator,
    )
