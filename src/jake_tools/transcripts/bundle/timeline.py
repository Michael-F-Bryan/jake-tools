"""M6: the combined-timeline transform.

Given a bundle's assembled media members, in note-embed order (default)
or explicit operator order, this builds one :class:`~.components.
TimelineCombinedComponent`: piecewise mappings ``{artefact, [0, dur) ->
[offset, offset+dur)}`` plus an explicit inter-recording gap record
between every consecutive pair (zero-length in v1's builder -- recordings
are placed back to back; only a real declared gap duration would make
one non-zero, and no fixture in this slice needs one). Overlapping
recordings are structurally impossible to construct through
:func:`build_combined_timeline` (it only ever adds durations), but the
underlying :class:`~.components.TimelineCombinedComponentBody` still
refuses one at construction (the actual M6 enforcement mechanism,
exercised directly by the property tests against a hand-built instance).

:func:`transform_timeline` is the store-level orchestration entry point,
mirroring :func:`.assemble.assemble`'s shape: read the current head's
media members, build the component, add it, append a revision, move the
head via the caller's already-held run lease (M2/M16). It requires an
already-assembled document (``NoDocumentYet`` -> refusal) -- the timeline
transform runs *after* ``assemble()``, never before.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..errors import TranscriptError
from .components import (
    ComponentRecord,
    MediaRecordingComponent,
    RecordingReferenceSetComponent,
    TimelineCombinedComponentBody,
    TimelineGapRecord,
    TimelineMappingSegment,
)
from .document import NoDocumentYet, project_head
from .ids import ArtefactId, ComponentId, RunId
from .records import OperationRef, RevisionRecord
from .store import BundleStore


class TimelineError(TranscriptError):
    """Base class for every error this module raises."""


class NoDocumentToTransformError(TimelineError):
    """The bundle has no assembled document yet (M18: assemble first)."""


class NoTimelineMediaError(TimelineError):
    """No media member resolved to build a combined timeline from."""


class AmbiguousNoteEmbedOrderError(TimelineError):
    """More than one note in the closure declares recording references,
    so "the" note-embed default order is ambiguous; pass ``order``
    explicitly."""


class TimelineRangeError(TimelineError):
    """A source<->combined mapping query fell outside every segment."""


# -- pure mapping functions (source <-> combined), seeded property-tested --


@dataclass(frozen=True)
class MediaMember:
    artefact_id: ArtefactId
    duration_ms: int


def build_combined_timeline(
    members: Sequence[MediaMember],
) -> TimelineCombinedComponentBody:
    """M6: place ``members`` back to back in the given order, starting at
    combined-domain 0, with an explicit zero-length gap record between
    every consecutive pair. Raises :class:`NoTimelineMediaError` if
    ``members`` is empty; a single member still produces a component (no
    gaps, one segment) -- gaps are *inter*-recording, so N members always
    produce exactly N-1 of them.
    """
    if not members:
        raise NoTimelineMediaError(
            "build_combined_timeline() requires at least one media member."
        )
    segments: list[TimelineMappingSegment] = []
    gaps: list[TimelineGapRecord] = []
    offset = 0
    previous: MediaMember | None = None
    for member in members:
        segments.append(
            TimelineMappingSegment(
                artefact_id=member.artefact_id,
                source_start_ms=0,
                source_end_ms=member.duration_ms,
                combined_start_ms=offset,
                combined_end_ms=offset + member.duration_ms,
            )
        )
        if previous is not None:
            gaps.append(
                TimelineGapRecord(
                    after_artefact_id=previous.artefact_id,
                    before_artefact_id=member.artefact_id,
                    combined_start_ms=offset,
                    combined_end_ms=offset,
                )
            )
        offset += member.duration_ms
        previous = member
    return TimelineCombinedComponentBody(segments=tuple(segments), gaps=tuple(gaps))


def map_source_to_combined(
    segments: Sequence[TimelineMappingSegment],
    *,
    artefact_id: ArtefactId,
    source_ms: int,
) -> int:
    """M6: map one instant in ``artefact_id``'s own source domain to the
    combined domain. Half-open per segment (``source_ms == source_end_ms``
    is out of range, matching every other half-open span in this
    codebase); refuses -- never clamps or guesses -- when no segment for
    this artefact covers ``source_ms``.
    """
    for segment in segments:
        if segment.artefact_id != artefact_id:
            continue
        if segment.source_start_ms <= source_ms < segment.source_end_ms:
            return segment.combined_start_ms + (source_ms - segment.source_start_ms)
    raise TimelineRangeError(
        f"source_ms={source_ms} is out of range for artefact {artefact_id} in this "
        "combined timeline."
    )


def map_combined_to_source(
    segments: Sequence[TimelineMappingSegment], *, combined_ms: int
) -> tuple[ArtefactId, int]:
    """The inverse of :func:`map_source_to_combined`: which artefact and
    source-domain instant a combined-domain instant belongs to. Refuses
    on a gap or beyond the last segment -- a gap is not "owned" by either
    neighbouring artefact (M6: gaps are their own explicit record, not
    silently attributed to one side)."""
    for segment in segments:
        if segment.combined_start_ms <= combined_ms < segment.combined_end_ms:
            return (
                segment.artefact_id,
                segment.source_start_ms + (combined_ms - segment.combined_start_ms),
            )
    raise TimelineRangeError(
        f"combined_ms={combined_ms} does not fall inside any mapped segment "
        "(it may be a gap, or beyond the timeline's own range)."
    )


# -- orchestration: resolve member order, build, append, move head ----------


def _resolve_member_order(
    document_components: Mapping[ComponentId, ComponentRecord],
    *,
    order: tuple[ArtefactId, ...] | None,
) -> tuple[ArtefactId, ...]:
    if order is not None:
        return order

    reference_sets = [
        component
        for component in document_components.values()
        if isinstance(component, RecordingReferenceSetComponent)
    ]
    if len(reference_sets) > 1:
        raise AmbiguousNoteEmbedOrderError(
            f"{len(reference_sets)} recording-reference-set components are present; "
            "pass order= explicitly to disambiguate the default note-embed order."
        )
    if not reference_sets:
        return ()

    media_by_path = {
        component.media_path: component
        for component in document_components.values()
        if isinstance(component, MediaRecordingComponent)
    }
    resolved: list[ArtefactId] = []
    for reference in reference_sets[0].references:
        media = media_by_path.get(reference.resolved_path)
        if media is not None:
            resolved.append(media.source_artefact_id)
    return tuple(resolved)


def transform_timeline(
    store: BundleStore,
    *,
    run_id: RunId,
    order: tuple[ArtefactId, ...] | None = None,
) -> RevisionRecord:
    """M6: append a combined-timeline revision on top of the bundle's
    current head.

    ``order`` explicitly overrides member order (operator-specified,
    M6's "explicit operator order"); when omitted, order is read from the
    bundle's single :class:`~.components.RecordingReferenceSetComponent`
    (note-embed order, M6's default), filtered to references that
    resolved to an actually-ingested :class:`~.components.
    MediaRecordingComponent` -- a reference-only embed never satisfies
    ``media.recording`` (M3) and contributes nothing to the timeline
    either. Raises :class:`NoDocumentToTransformError` if nothing has
    been assembled yet, or :class:`NoTimelineMediaError` if member
    resolution (default or explicit) yields nothing to build from.
    """
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise NoDocumentToTransformError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )

    resolved_order = _resolve_member_order(document.components, order=order)
    if not resolved_order:
        raise NoTimelineMediaError(
            "no media member resolved for the combined timeline (no explicit "
            "order given, and no note-embed reference resolved to ingested "
            "media)."
        )

    media_by_artefact_id = {
        component.source_artefact_id: component
        for component in document.components.values()
        if isinstance(component, MediaRecordingComponent)
    }
    missing = [
        artefact_id
        for artefact_id in resolved_order
        if artefact_id not in media_by_artefact_id
    ]
    if missing:
        raise NoTimelineMediaError(
            f"artefact ID(s) named in order have no ingested MediaRecordingComponent "
            f"in this document: {missing}"
        )

    members = tuple(
        MediaMember(
            artefact_id=artefact_id,
            duration_ms=media_by_artefact_id[artefact_id].duration_ms,
        )
        for artefact_id in resolved_order
    )
    body = build_combined_timeline(members)
    component = store.add_component(body)

    revision = store.append_revision(
        operation=OperationRef(
            kind="timeline",
            input_ids=resolved_order,
            rationale="combined timeline over media members (M6)",
        ),
        parent_revision_ids=(document.revision_id,),
        component_ids=(component.component_id,),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)
    return revision
