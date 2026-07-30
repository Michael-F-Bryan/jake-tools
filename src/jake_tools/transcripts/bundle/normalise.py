"""M7: the normalisation transform -- raw inference evidence becomes the
canonical timed turn set.

Raw ASR and diarisation artefacts stay faithful to the model (M11):
duplicate and zero-length tokens are legal evidence there and are never
rewritten. *Canonical* turns are strict, and this is the one place that
conversion happens, with a lineage record for every raw segment it drops.

What one pass produces, all in a single revision:

- a :class:`~.components.TimedTurnSetComponent` in the combined
  timeline's own coordinate domain (M6), covering every recording at
  once -- ``transcript.timed`` is a one-cardinality key, so a
  multi-recording bundle has exactly one canonical turn set, not one per
  recording;
- a :class:`~.components.MachineAttributionSetComponent`: the voice
  clusters diarisation found and which canonical turn each belongs to
  (M4 ``speakers.machine-clustered``). Clusters are never participant
  identities (F13) -- nothing in that component names a person;
- a :class:`~.components.NormalisationLedgerComponent`: M7's raw-segment
  and canonical-turn coverage ledgers, plus one
  ``dropped_as_duplicate``/``dropped_as_empty`` record per removed raw
  token.

Re-running supersedes all three (M21), so a second pass corrects rather
than duplicating -- and because ``turn_id``s are minted fresh on each
pass, an applied review's binding to the *old* inventory is correctly
invalidated rather than silently carried onto different turns (M8).

The token -> turn rule
----------------------
Each raw token is assigned to the diarisation segment it overlaps most
(ties resolve to the *preceding* segment, M7), and every token assigned
to one segment becomes one turn. Grouping by *segment* -- not by cluster,
and not by adjacency in the time-ordered token stream -- is what keeps two
people talking across each other as two overlapping turns (D4) instead of
four interleaved fragments, and what keeps two consecutive utterances from
one speaker separate until an explicit same-speaker adjacent merge (M9
polish) decides otherwise. Tokens no segment covers are grouped by time
proximity and reported as unattributed: a turn with no voice evidence is
honest, and M8's ladder resolves it at rung 8 rather than by guessing a
neighbour's speaker.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..errors import TranscriptError
from .components import (
    AsrResultComponent,
    CoverageLedger,
    DiarisationResultComponent,
    DroppedSegmentRecord,
    DropReason,
    MachineAttributionSetComponent,
    MachineAttributionSetComponentBody,
    NormalisationLedgerComponent,
    NormalisationLedgerComponentBody,
    SpeakerCluster,
    TimedTurn,
    TimedTurnSetComponent,
    TimedTurnSetComponentBody,
    TimelineCombinedComponent,
    TurnClusterAssignment,
)
from .document import NoDocumentYet, TranscriptDocumentV1, project_head
from .ids import (
    ArtefactId,
    ClusterId,
    ComponentId,
    RunId,
    SegmentId,
    TurnId,
    combined_domain,
    mint_id,
)
from .records import OperationRef, RevisionRecord
from .store import BundleStore
from .worker_contract import (
    WireAsrStageResult,
    WireDiarisationSegment,
    WireDiarisationStageResult,
)

#: Bumped whenever the token->turn rule below changes in a way that would
#: produce a different canonical set from identical inputs. Participates
#: in the ledger's ``config_hash``, so M12 can tell a stale normalisation
#: from a current one.
NORMALISER_VERSION = "v1"


class NormaliseError(TranscriptError):
    """Base class for every error this module raises."""


class NoDocumentToNormaliseError(NormaliseError):
    """The bundle has no assembled document yet (M18: assemble first)."""


class NoCombinedTimelineError(NormaliseError):
    """Normalisation maps every recording's raw output into one shared
    combined domain, so it needs the M6 timeline that defines it -- run
    ``transform timeline`` first. Refused rather than falling back to
    per-recording source coordinates, which would silently produce a
    canonical set whose timestamps mean different things per turn.
    """


class NoInferenceEvidenceError(NormaliseError):
    """No completed ASR result exists in the document to normalise."""


class UnmappedRecordingError(NormaliseError):
    """A recording with ASR output has no segment in the combined
    timeline, so its turns have no combined-domain position. Refused
    rather than guessed -- an unmapped recording means the timeline was
    built before this recording was assembled, and rebuilding it is the
    fix."""


@dataclass(frozen=True)
class RawToken:
    """One raw ASR token plus the immutable source-segment ID minted for
    it at parse time (M7). Every raw token gets one, including the ones
    this pass goes on to drop -- a lineage record naming a segment ID that
    was never minted would not be auditable."""

    source_segment_id: SegmentId
    source_artefact_id: ArtefactId
    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class NormaliseOutcome:
    """``attribution`` is ``None`` only when every recording's diarisation
    stage failed: the canonical turn set is still real and useful (M11's
    partial-failure rule), but there are no voice clusters, so
    ``speakers.machine-clustered`` stays absent and every turn resolves at
    M8 rung 8 until a human reviews it."""

    revision: RevisionRecord
    turn_set: TimedTurnSetComponent
    attribution: MachineAttributionSetComponent | None
    ledger: NormalisationLedgerComponent
    dropped_duplicate_count: int
    dropped_empty_count: int
    unattributed_turn_count: int


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _config_hash() -> str:
    return _sha256_hex(
        json.dumps(
            {
                "normaliser_version": NORMALISER_VERSION,
                "unattributed_gap_ms": UNATTRIBUTED_GAP_MS,
            },
            sort_keys=True,
        ).encode()
    )


def assign_tokens_to_segments(
    tokens: Sequence[RawToken], segments: Sequence[WireDiarisationSegment]
) -> tuple[int | None, ...]:
    """M7's maximal-overlap rule, as a pure function over one recording.

    Returns one entry per token: the index of the diarisation segment it
    overlaps most, or ``None`` when no segment overlaps it at all.
    ``segments`` must already be in ascending ``(start_ms, end_ms)`` order
    -- that is what makes "ties resolve to the preceding segment" a
    property of the data rather than of iteration luck, since the first
    segment achieving the maximum wins.
    """
    assignments: list[int | None] = []
    for token in tokens:
        best_index: int | None = None
        best_overlap = 0
        for index, segment in enumerate(segments):
            overlap = min(token.end_ms, segment.end_ms) - max(
                token.start_ms, segment.start_ms
            )
            if overlap > best_overlap:
                best_overlap, best_index = overlap, index
        assignments.append(best_index)
    return tuple(assignments)


def _clean_raw_tokens(
    tokens: Sequence[RawToken],
) -> tuple[tuple[RawToken, ...], tuple[DroppedSegmentRecord, ...]]:
    """Canonicalise one recording's raw token stream (M7).

    Two removals, both with lineage: zero-length tokens
    (``dropped_as_empty`` -- legal raw evidence per M11, never a canonical
    turn per M6) and exact repeats of an already-retained ``(start, end,
    text)`` (``dropped_as_duplicate``, naming the segment retained in its
    place). Nothing else is removed here: normalisation cleans structure,
    never content.

    Ordering is a stable sort on ``(start_ms, end_ms)`` applied *after*
    the drops, so the retained token for a duplicate group is the one the
    model emitted first.
    """
    retained: list[RawToken] = []
    dropped: list[DroppedSegmentRecord] = []
    first_seen: dict[tuple[int, int, str], SegmentId] = {}
    for token in tokens:
        if token.end_ms <= token.start_ms:
            dropped.append(
                DroppedSegmentRecord(
                    source_segment_id=token.source_segment_id,
                    source_artefact_id=token.source_artefact_id,
                    reason=DropReason.DROPPED_AS_EMPTY,
                    detail=(
                        f"zero-length raw token [{token.start_ms}, {token.end_ms}) "
                        f"{token.text!r}"
                    ),
                )
            )
            continue
        key = (token.start_ms, token.end_ms, token.text)
        previous = first_seen.get(key)
        if previous is not None:
            dropped.append(
                DroppedSegmentRecord(
                    source_segment_id=token.source_segment_id,
                    source_artefact_id=token.source_artefact_id,
                    reason=DropReason.DROPPED_AS_DUPLICATE,
                    retained_source_segment_id=previous,
                    detail=(
                        f"identical raw token [{token.start_ms}, {token.end_ms}) "
                        f"{token.text!r} emitted more than once"
                    ),
                )
            )
            continue
        first_seen[key] = token.source_segment_id
        retained.append(token)
    ordered = tuple(sorted(retained, key=lambda t: (t.start_ms, t.end_ms)))
    return ordered, tuple(dropped)


@dataclass(frozen=True)
class _DraftTurn:
    """One grouped run of tokens, still in its own recording's source
    coordinates -- shifted into the combined domain only once the whole
    document's turns exist, so the shift is applied in exactly one place.
    """

    source_artefact_id: ArtefactId
    source_segment_id: SegmentId
    cluster_id: ClusterId | None
    speaker_label: str
    text: str
    start_ms: int
    end_ms: int


#: The label a canonical turn carries when no diarisation segment covers
#: it. A raw cluster label would be a lie (there is no cluster) and an
#: empty string is not constructible (``TimedTurn.speaker_label`` requires
#: content), so the honest machine-level label is that there is no voice
#: evidence -- M8's ladder resolves it at rung 8.
UNATTRIBUTED_SPEAKER_LABEL = "unattributed"


#: How far apart two *unattributed* tokens may be and still belong to one
#: turn. Only reached where diarisation covered nothing, so there is no
#: voice evidence to group by; without a bound, one recording's stray
#: tokens would collapse into a single turn spanning the whole meeting.
#: Participates in the ledger's config hash, like every other tunable.
UNATTRIBUTED_GAP_MS = 2_000


def _build_draft(
    group: Sequence[RawToken],
    *,
    cluster_id: ClusterId | None,
    speaker_label: str,
) -> _DraftTurn | None:
    text = "".join(token.text for token in group).strip()
    start_ms = group[0].start_ms
    end_ms = max(token.end_ms for token in group)
    if not text or end_ms <= start_ms:
        return None
    return _DraftTurn(
        source_artefact_id=group[0].source_artefact_id,
        source_segment_id=group[0].source_segment_id,
        cluster_id=cluster_id,
        speaker_label=speaker_label,
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _group_tokens_into_turns(
    tokens: Sequence[RawToken],
    assignments: Sequence[int | None],
    segments: Sequence[WireDiarisationSegment],
    cluster_ids_by_label: Mapping[str, ClusterId],
) -> tuple[_DraftTurn, ...]:
    """Group tokens into turns, one turn per diarisation segment.

    The group key is the *segment index*, not the cluster and not
    adjacency in the time-ordered token stream. Both alternatives are
    wrong on real audio:

    - grouping by cluster would merge two consecutive utterances from one
      speaker, and an adjacent same-speaker merge is an explicit M9
      polish decision with its own lineage, never an accident here;
    - grouping by adjacency shatters a turn whenever speech overlaps --
      two people talking across each other interleave in the sorted token
      stream, and the corpus's simultaneous 00:58 utterances are exactly
      that case. Overlapping turns are representable (D4), so the right
      answer is two whole turns that overlap in time, not four fragments.

    Tokens no segment covers have no voice evidence to group by, so they
    are grouped by time proximity (:data:`UNATTRIBUTED_GAP_MS`) and
    reported as unattributed.
    """
    by_segment: dict[int, list[RawToken]] = {}
    unassigned: list[RawToken] = []
    for token, segment_index in zip(tokens, assignments, strict=True):
        if segment_index is None:
            unassigned.append(token)
        else:
            by_segment.setdefault(segment_index, []).append(token)

    drafts: list[_DraftTurn] = []
    for segment_index, group in by_segment.items():
        raw_label = segments[segment_index].speaker_label
        draft = _build_draft(
            group,
            cluster_id=cluster_ids_by_label[raw_label],
            speaker_label=raw_label,
        )
        if draft is not None:
            drafts.append(draft)

    run: list[RawToken] = []
    for token in unassigned:
        if run and token.start_ms - run[-1].end_ms > UNATTRIBUTED_GAP_MS:
            draft = _build_draft(
                run, cluster_id=None, speaker_label=UNATTRIBUTED_SPEAKER_LABEL
            )
            if draft is not None:
                drafts.append(draft)
            run = []
        run.append(token)
    if run:
        draft = _build_draft(
            run, cluster_id=None, speaker_label=UNATTRIBUTED_SPEAKER_LABEL
        )
        if draft is not None:
            drafts.append(draft)
    return tuple(sorted(drafts, key=lambda draft: (draft.start_ms, draft.end_ms)))


def _combined_offsets(
    timeline: TimelineCombinedComponent,
) -> Mapping[ArtefactId, int]:
    """Each recording's pure-shift offset into the combined domain (M6).

    v1's timeline builder emits exactly one segment per recording, each
    mapping ``[0, duration)``, so the shift is a single integer per
    artefact rather than a piecewise lookup per turn.
    """
    return {
        segment.artefact_id: segment.combined_start_ms - segment.source_start_ms
        for segment in timeline.segments
    }


def _load_asr(store: BundleStore, component: AsrResultComponent) -> WireAsrStageResult:
    return WireAsrStageResult.model_validate_json(
        store.load_artefact_bytes(component.result_artefact_id)
    )


def _load_diarisation(
    store: BundleStore, component: DiarisationResultComponent
) -> WireDiarisationStageResult:
    return WireDiarisationStageResult.model_validate_json(
        store.load_artefact_bytes(component.result_artefact_id)
    )


def _superseded_component_ids(
    document: TranscriptDocumentV1,
) -> tuple[ComponentId, ...]:
    """M21: every component a previous normalisation pass produced.

    Superseding them (rather than adding alongside) is what keeps
    ``transcript.timed`` a one-cardinality key across re-runs -- two
    canonical turn sets in one closure is an ambiguity the registry
    correctly refuses, so a second pass must retract the first.

    A provider-adapter turn set (Teams VTT) would also be caught here.
    That is deliberate and correct under D5: if a bundle somehow held both
    an imported provider transcript and local ASR, normalising is the
    explicit act of selecting the local one, and the superseded provider
    set remains immutable on disk as evidence.
    """
    return tuple(
        component.component_id
        for component in document.components.values()
        if isinstance(
            component,
            TimedTurnSetComponent
            | MachineAttributionSetComponent
            | NormalisationLedgerComponent,
        )
    )


def normalise_transcript(store: BundleStore, *, run_id: RunId) -> NormaliseOutcome:
    """M7: build the canonical timed turn set from the head's raw evidence.

    Requires an assembled document with a combined timeline (M6) and at
    least one completed ASR result (M11). Diarisation is *optional* per
    recording: a recording whose diarisation stage failed still
    contributes canonical turns, all unattributed -- M11's partial-failure
    rule means one failed stage never costs the other's evidence, and an
    honest "no voice evidence for these turns" is exactly what M8 rung 8
    is for.
    """
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise NoDocumentToNormaliseError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )

    timelines = document.components_of(TimelineCombinedComponent)
    if len(timelines) != 1:
        raise NoCombinedTimelineError(
            f"{len(timelines)} combined-timeline components in this document; "
            "normalisation needs exactly one (run `transform timeline` first)."
        )
    timeline = timelines[0]
    offsets = _combined_offsets(timeline)

    asr_components = sorted(
        document.components_of(AsrResultComponent),
        key=lambda component: component.media_artefact_id,
    )
    if not asr_components:
        raise NoInferenceEvidenceError(
            "no completed ASR result exists in this document (M11: run `transform "
            "transcribe` first)."
        )
    diarisation_by_media = {
        component.media_artefact_id: component
        for component in document.components_of(DiarisationResultComponent)
    }

    unmapped = sorted(
        component.media_artefact_id
        for component in asr_components
        if component.media_artefact_id not in offsets
    )
    if unmapped:
        raise UnmappedRecordingError(
            f"recording(s) with ASR output have no combined-timeline segment: "
            f"{unmapped} -- rebuild the timeline so every transcribed recording "
            "has a combined-domain position."
        )

    all_turns: list[TimedTurn] = []
    all_dropped: list[DroppedSegmentRecord] = []
    clusters: list[SpeakerCluster] = []
    assignments_out: list[TurnClusterAssignment] = []
    unattributed_turn_ids: list[TurnId] = []
    raw_token_total = 0

    for asr_component in asr_components:
        media_artefact_id = asr_component.media_artefact_id
        asr = _load_asr(store, asr_component)
        raw_tokens = tuple(
            RawToken(
                source_segment_id=mint_id("seg"),
                source_artefact_id=media_artefact_id,
                start_ms=token.start_ms,
                end_ms=token.end_ms,
                text=token.text,
            )
            for token in (asr.output.tokens if asr.output is not None else ())
        )
        raw_token_total += len(raw_tokens)
        tokens, dropped = _clean_raw_tokens(raw_tokens)
        all_dropped.extend(dropped)
        if not tokens:
            continue

        diarisation_component = diarisation_by_media.get(media_artefact_id)
        segments: tuple[WireDiarisationSegment, ...] = ()
        cluster_ids_by_label: dict[str, ClusterId] = {}
        if diarisation_component is not None:
            diarisation = _load_diarisation(store, diarisation_component)
            segments = tuple(
                sorted(
                    diarisation.output.segments if diarisation.output else (),
                    key=lambda segment: (segment.start_ms, segment.end_ms),
                )
            )
            for raw_label in dict.fromkeys(
                segment.speaker_label for segment in segments
            ):
                own = [
                    segment
                    for segment in segments
                    if segment.speaker_label == raw_label
                ]
                cluster_id = mint_id("cluster")
                cluster_ids_by_label[raw_label] = cluster_id
                clusters.append(
                    SpeakerCluster(
                        cluster_id=cluster_id,
                        raw_label=raw_label,
                        media_artefact_id=media_artefact_id,
                        diarisation_artefact_id=(
                            diarisation_component.result_artefact_id
                        ),
                        segment_count=len(own),
                        total_ms=sum(
                            segment.end_ms - segment.start_ms for segment in own
                        ),
                    )
                )

        token_assignments = assign_tokens_to_segments(tokens, segments)
        offset = offsets[media_artefact_id]
        for draft in _group_tokens_into_turns(
            tokens, token_assignments, segments, cluster_ids_by_label
        ):
            turn_id = mint_id("turn")
            all_turns.append(
                TimedTurn(
                    turn_id=turn_id,
                    source_segment_id=draft.source_segment_id,
                    source_artefact_id=draft.source_artefact_id,
                    speaker_label=draft.speaker_label,
                    text=draft.text,
                    start_ms=draft.start_ms + offset,
                    end_ms=draft.end_ms + offset,
                )
            )
            if draft.cluster_id is None:
                unattributed_turn_ids.append(turn_id)
            else:
                assignments_out.append(
                    TurnClusterAssignment(
                        turn_id=turn_id,
                        cluster_id=draft.cluster_id,
                        overlap_ms=draft.end_ms - draft.start_ms,
                    )
                )

    if not all_turns:
        raise NoInferenceEvidenceError(
            "the ASR evidence in this document yielded no canonical turns (every "
            "raw token was empty, duplicated, or textless)."
        )

    ordered_turns = tuple(
        sorted(
            all_turns,
            key=lambda turn: (turn.start_ms, turn.end_ms, turn.source_segment_id),
        )
    )
    turn_set_body = TimedTurnSetComponentBody(
        source_artefact_ids=tuple(
            sorted({turn.source_artefact_id for turn in ordered_turns})
        ),
        coordinate_domain=combined_domain(timeline.component_id),
        turns=ordered_turns,
    )
    ledger_body = NormalisationLedgerComponentBody(
        source_artefact_ids=turn_set_body.source_artefact_ids,
        config_hash=_config_hash(),
        raw_source_segments=CoverageLedger(
            total=raw_token_total,
            accounted=raw_token_total - len(all_dropped),
            dropped=len(all_dropped),
        ),
        canonical_turns=CoverageLedger(
            total=len(ordered_turns), accounted=len(ordered_turns), dropped=0
        ),
        dropped=tuple(all_dropped),
    )

    turn_set = store.add_component(turn_set_body)
    assert isinstance(turn_set, TimedTurnSetComponent)
    ledger = store.add_component(ledger_body)
    assert isinstance(ledger, NormalisationLedgerComponent)
    # A recording whose diarisation stage failed contributes no cluster,
    # so a document where *every* recording lost diarisation has none at
    # all. MachineAttributionSetComponentBody requires at least one
    # cluster (an attribution set with nothing to attribute to is not
    # evidence), so that case correctly produces no component -- and
    # `speakers.machine-clustered` stays absent rather than being faked
    # with an empty one.
    attribution: MachineAttributionSetComponent | None = None
    if clusters:
        attribution_body = MachineAttributionSetComponentBody(
            clusters=tuple(sorted(clusters, key=lambda cluster: cluster.cluster_id)),
            assignments=tuple(sorted(assignments_out, key=lambda item: item.turn_id)),
            unattributed_turn_ids=tuple(sorted(unattributed_turn_ids)),
        )
        added = store.add_component(attribution_body)
        assert isinstance(added, MachineAttributionSetComponent)
        attribution = added

    new_component_ids = [turn_set.component_id, ledger.component_id]
    if attribution is not None:
        new_component_ids.append(attribution.component_id)
    revision = store.append_revision(
        operation=OperationRef(
            kind="normalise",
            input_ids=tuple(
                component.result_artefact_id for component in asr_components
            ),
            config_hash=ledger_body.config_hash,
            rationale=(
                "canonical timed turns from raw ASR/diarisation evidence, in the "
                "combined timeline's coordinate domain (M7)"
            ),
        ),
        parent_revision_ids=(document.revision_id,),
        component_ids=tuple(new_component_ids),
        superseded_component_ids=_superseded_component_ids(document),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)

    return NormaliseOutcome(
        revision=revision,
        turn_set=turn_set,
        attribution=attribution,
        ledger=ledger,
        dropped_duplicate_count=sum(
            1
            for record in all_dropped
            if record.reason == DropReason.DROPPED_AS_DUPLICATE
        ),
        dropped_empty_count=sum(
            1 for record in all_dropped if record.reason == DropReason.DROPPED_AS_EMPTY
        ),
        unattributed_turn_count=len(unattributed_turn_ids),
    )
