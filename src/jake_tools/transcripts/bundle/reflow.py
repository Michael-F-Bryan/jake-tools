"""Deterministic turn reflow before model-backed correction and polish.

Reflow repairs representation damage, not speech. It joins turns only when a
reviewed speaker identity makes the continuation explicit, or when an
unattributed cue is an objective word suffix such as ``lic`` + ``ense``. An
overlapping interjection remains a separate turn; the interrupted speaker's
continuation moves back beside the utterance it completes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .assignment import SpeakerContext, canonical_turn_set
from .components import (
    ChapterSetComponent,
    EditorialDerivationMode,
    EditorialNode,
    EditorialOperation,
    EditorialOperationKind,
    EditorialOperationLedgerComponentBody,
    EditorialSpeakerState,
    EditorialTranscriptComponentBody,
    MinutesComponent,
    TextEditEntry,
    TextEditLedgerComponent,
    TextEditLedgerComponentBody,
    TextEditMode,
    TextEditOperation,
    TimedTurn,
    TimedTurnSetComponent,
    TimedTurnSetComponentBody,
)
from .document import NoDocumentYet, project_head
from .ids import RunId, mint_id
from .records import OperationRef, RevisionRecord
from .store import BundleStore

_SUFFIX_RE = re.compile(
    r"^(?:ance|ence|ense|able|ible|ation|ition|ment|ness|ing|ers?|ed|ly|es|s|it)\b",
    re.IGNORECASE,
)
_LAST_WORD_RE = re.compile(r"([A-Za-z']+)$")
_FIRST_WORD_RE = re.compile(r"^([A-Za-z']+)")


@dataclass(frozen=True)
class ReflowResult:
    turns: tuple[TimedTurn, ...]
    entries: tuple[TextEditEntry, ...]
    merged_turn_count: int


@dataclass(frozen=True)
class ReflowOutcome:
    revision: RevisionRecord | None
    turn_set: TimedTurnSetComponent
    ledger: TextEditLedgerComponent | None
    merged_turn_count: int


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _participant_id(turn: TimedTurn, context: SpeakerContext) -> str | None:
    assignment = context.assignment_for(turn)
    if not assignment.is_reviewed:
        return None
    return assignment.participant_id


def _same_reviewed_speaker(
    first: TimedTurn, second: TimedTurn, context: SpeakerContext
) -> bool:
    first_id = _participant_id(first, context)
    return first_id is not None and first_id == _participant_id(second, context)


def _looks_like_word_suffix(first: TimedTurn, second: TimedTurn) -> bool:
    if first.text.endswith((".", "?", "!", ",", ";", ":")):
        return False
    last = _LAST_WORD_RE.search(first.text)
    following = _FIRST_WORD_RE.match(second.text)
    if last is None or following is None:
        return False
    return bool(_SUFFIX_RE.match(following.group(1))) and len(last.group(1)) <= 6


def _same_source(first: TimedTurn, second: TimedTurn) -> bool:
    return first.source_artefact_id == second.source_artefact_id


def _within_gap(first: TimedTurn, second: TimedTurn, max_gap_ms: int) -> bool:
    return second.start_ms - first.end_ms <= max_gap_ms


def _join_text(turns: tuple[TimedTurn, ...], *, lexical_boundary: bool) -> str:
    text = turns[0].text.rstrip()
    previous = turns[0]
    for turn in turns[1:]:
        following = turn.text.lstrip()
        if lexical_boundary and _looks_like_word_suffix(previous, turn):
            text += following
        else:
            text += f" {following}"
        previous = turn
    return text.strip()


def _merge_entry(
    turns: tuple[TimedTurn, ...], output: TimedTurn, *, evidence_ref: str
) -> TextEditEntry:
    return TextEditEntry(
        operation=TextEditOperation.MERGE,
        input_turn_ids=tuple(turn.turn_id for turn in turns),
        output_turn_ids=(output.turn_id,),
        old_text_sha256=_sha256_text("\n".join(turn.text for turn in turns)),
        new_text_sha256=_sha256_text(output.text),
        evidence_ref=evidence_ref,
    )


def _identity_entry(turn: TimedTurn) -> TextEditEntry:
    digest = _sha256_text(turn.text)
    return TextEditEntry(
        operation=TextEditOperation.IDENTITY,
        input_turn_ids=(turn.turn_id,),
        output_turn_ids=(turn.turn_id,),
        old_text_sha256=digest,
        new_text_sha256=digest,
    )


def reflow_turns(
    turns: tuple[TimedTurn, ...],
    context: SpeakerContext,
    *,
    max_gap_ms: int = 3000,
    lexical_gap_ms: int = 1000,
) -> ReflowResult:
    """Join safe continuations while preserving a complete M9/M7 ledger."""
    if max_gap_ms < 0 or lexical_gap_ms < 0:
        raise ValueError("reflow gap limits must be zero or greater")

    groups: dict[int, tuple[tuple[int, ...], str]] = {}
    consumed: set[int] = set()

    for index, turn in enumerate(turns):
        if index in consumed:
            continue

        group = [index]
        evidence = "same-reviewed-speaker"
        following_index = index + 1
        while following_index < len(turns) and following_index not in consumed:
            previous = turns[group[-1]]
            following = turns[following_index]
            if not _same_source(previous, following):
                break
            if _same_reviewed_speaker(previous, following, context) and _within_gap(
                previous, following, max_gap_ms
            ):
                group.append(following_index)
                following_index += 1
                continue
            if (
                _participant_id(previous, context) is not None
                and _participant_id(following, context) is None
                and _within_gap(previous, following, lexical_gap_ms)
                and _looks_like_word_suffix(previous, following)
            ):
                group.append(following_index)
                evidence = "lexical-boundary-repair"
                following_index += 1
                continue
            break

        if len(group) == 1 and index + 2 < len(turns):
            interjection = turns[index + 1]
            continuation = turns[index + 2]
            if (
                _same_source(turn, continuation)
                and _same_reviewed_speaker(turn, continuation, context)
                and _within_gap(turn, continuation, max_gap_ms)
                and continuation.start_ms < interjection.end_ms
            ):
                group.append(index + 2)
                evidence = "overlapping-same-speaker-continuation"

        if len(group) > 1:
            group_tuple = tuple(group)
            groups[index] = (group_tuple, evidence)
            consumed.update(group_tuple[1:])

    output_turns: list[TimedTurn] = []
    entries: list[TextEditEntry] = []
    for index, turn in enumerate(turns):
        if index in consumed:
            continue
        planned = groups.get(index)
        if planned is None:
            output_turns.append(turn)
            entries.append(_identity_entry(turn))
            continue

        indices, evidence = planned
        source_turns = tuple(turns[source_index] for source_index in indices)
        merged = turn.model_copy(
            update={
                "end_ms": max(source.end_ms for source in source_turns),
                "text": _join_text(
                    source_turns,
                    lexical_boundary=evidence == "lexical-boundary-repair",
                ),
            }
        )
        output_turns.append(merged)
        entries.append(_merge_entry(source_turns, merged, evidence_ref=evidence))

    return ReflowResult(
        turns=tuple(output_turns),
        entries=tuple(entries),
        merged_turn_count=sum(len(indices) - 1 for indices, _ in groups.values()),
    )


def reflow_until_stable(
    turns: tuple[TimedTurn, ...],
    context: SpeakerContext,
    *,
    max_gap_ms: int = 3000,
    lexical_gap_ms: int = 1000,
) -> tuple[ReflowResult, ...]:
    """Run deterministic reflow passes until no new adjacency remains."""
    passes: list[ReflowResult] = []
    current = turns
    for _ in range(len(turns)):
        result = reflow_turns(
            current,
            context,
            max_gap_ms=max_gap_ms,
            lexical_gap_ms=lexical_gap_ms,
        )
        if result.merged_turn_count == 0:
            return tuple(passes)
        if len(result.turns) >= len(current):
            raise RuntimeError("reflow reported merges without reducing the turn count")
        passes.append(result)
        current = result.turns
    raise RuntimeError("reflow did not converge within the input turn count")


@dataclass(frozen=True)
class EditorialReflowResult:
    body: EditorialTranscriptComponentBody
    ledger: EditorialOperationLedgerComponentBody
    merged_node_count: int


def editorial_reflow_config_hash(
    *, max_gap_ms: int = 3000, lexical_gap_ms: int = 1000
) -> str:
    return _sha256_text(
        f"editorial-reflow-v1:max-gap-ms={max_gap_ms}:lexical-gap-ms={lexical_gap_ms}"
    )


def _same_named_editorial_speaker(first: EditorialNode, second: EditorialNode) -> bool:
    return (
        first.attribution.state == EditorialSpeakerState.NAMED
        and second.attribution.state == EditorialSpeakerState.NAMED
        and first.attribution.participant_ids == second.attribution.participant_ids
    )


def _looks_like_editorial_word_suffix(
    first: EditorialNode, second: EditorialNode
) -> bool:
    if first.text.endswith((".", "?", "!", ",", ";", ":")):
        return False
    last = _LAST_WORD_RE.search(first.text)
    following = _FIRST_WORD_RE.match(second.text)
    if last is None or following is None:
        return False
    return bool(_SUFFIX_RE.match(following.group(1))) and len(last.group(1)) <= 6


def _editorial_group_is_reviewed_overlap(
    body: EditorialTranscriptComponentBody,
    nodes: tuple[EditorialNode, ...],
) -> bool:
    group_id = nodes[0].overlap_group_id
    if group_id is None or any(node.overlap_group_id != group_id for node in nodes):
        return False
    group = next(
        (
            candidate
            for candidate in body.overlap_groups
            if candidate.overlap_group_id == group_id
        ),
        None,
    )
    if group is None:
        return False
    source_turn_ids = {span.turn_id for node in nodes for span in node.source_spans}
    return source_turn_ids.issubset(set(group.turn_ids))


def reflow_editorial_nodes(
    body: EditorialTranscriptComponentBody,
    *,
    max_gap_ms: int = 3000,
    lexical_gap_ms: int = 1000,
) -> EditorialReflowResult:
    """Port the bounded deterministic reflow rules onto editorial nodes."""
    if max_gap_ms < 0 or lexical_gap_ms < 0:
        raise ValueError("reflow gap limits must be zero or greater")
    nodes = body.nodes
    groups: dict[int, tuple[tuple[int, ...], str]] = {}
    consumed: set[int] = set()
    for index, node in enumerate(nodes):
        if index in consumed:
            continue
        group = [index]
        evidence = "same-reviewed-speaker"
        following_index = index + 1
        while following_index < len(nodes) and following_index not in consumed:
            previous = nodes[group[-1]]
            following = nodes[following_index]
            gap_ms = following.display_start_ms - previous.display_end_ms
            if (
                _same_named_editorial_speaker(previous, following)
                and gap_ms <= max_gap_ms
            ):
                group.append(following_index)
                following_index += 1
                continue
            if (
                previous.attribution.state == EditorialSpeakerState.NAMED
                and following.attribution.state == EditorialSpeakerState.UNCLEAR
                and gap_ms <= lexical_gap_ms
                and _looks_like_editorial_word_suffix(previous, following)
            ):
                group.append(following_index)
                evidence = "lexical-boundary-repair"
                following_index += 1
                continue
            break
        if len(group) == 1 and index + 2 < len(nodes):
            interjection = nodes[index + 1]
            continuation = nodes[index + 2]
            overlap_nodes = (node, interjection, continuation)
            if (
                _same_named_editorial_speaker(node, continuation)
                and continuation.display_start_ms < interjection.display_end_ms
                and continuation.display_start_ms - node.display_end_ms <= max_gap_ms
                and _editorial_group_is_reviewed_overlap(body, overlap_nodes)
            ):
                group.append(index + 2)
                evidence = "reviewed-overlap-same-speaker-continuation"
        if len(group) > 1:
            indices = tuple(group)
            groups[index] = (indices, evidence)
            consumed.update(indices[1:])

    config_hash = editorial_reflow_config_hash(
        max_gap_ms=max_gap_ms, lexical_gap_ms=lexical_gap_ms
    )
    output_nodes: list[EditorialNode] = []
    operations: list[EditorialOperation] = []
    merged_node_count = 0
    for index, node in enumerate(nodes):
        if index in consumed:
            continue
        planned = groups.get(index)
        if planned is None:
            output_nodes.append(node)
            operations.append(
                EditorialOperation(
                    operation_id=f"identity-reflow:{node.node_id}",
                    operation=EditorialOperationKind.IDENTITY,
                    input_node_ids=(node.node_id,),
                    output_node_ids=(node.node_id,),
                    source_spans=node.source_spans,
                    reason="deterministic reflow left node unchanged",
                )
            )
            continue
        indices, evidence = planned
        source_nodes = tuple(nodes[source_index] for source_index in indices)
        lexical = evidence == "lexical-boundary-repair"
        text = source_nodes[0].text.rstrip()
        for source in source_nodes[1:]:
            following = source.text.lstrip()
            text += following if lexical else f" {following}"
        output_id = mint_id("editorial")
        operation_id = f"reflow:{output_id}"
        overlap_group_id = (
            source_nodes[0].overlap_group_id
            if evidence.startswith("reviewed-overlap")
            else None
        )
        merged = EditorialNode(
            node_id=output_id,
            display_order=len(output_nodes),
            text=text.strip(),
            source_spans=tuple(
                span for source in source_nodes for span in source.source_spans
            ),
            constituent_intervals=tuple(
                interval
                for source in source_nodes
                for interval in source.constituent_intervals
            ),
            overlap_group_id=overlap_group_id,
            display_start_ms=min(source.display_start_ms for source in source_nodes),
            display_end_ms=max(source.display_end_ms for source in source_nodes),
            attribution=source_nodes[0].attribution,
            derivation_mode=(
                EditorialDerivationMode.RECONSTRUCTED
                if lexical
                else EditorialDerivationMode.NORMALISED
            ),
            operation_ancestry=tuple(
                ancestry
                for source in source_nodes
                for ancestry in source.operation_ancestry
            )
            + (operation_id,),
            creator="jake-tools:deterministic-editorial-reflow-v1",
            config_hash=config_hash,
        )
        output_nodes.append(merged)
        operations.append(
            EditorialOperation(
                operation_id=operation_id,
                operation=(
                    EditorialOperationKind.OVERLAP_REORDER
                    if overlap_group_id is not None
                    else EditorialOperationKind.MERGE
                ),
                input_node_ids=tuple(source.node_id for source in source_nodes),
                output_node_ids=(output_id,),
                source_spans=merged.source_spans,
                reason=evidence,
                overlap_group_id=overlap_group_id,
            )
        )
        merged_node_count += len(source_nodes) - 1

    reindexed_nodes = tuple(
        node.model_copy(update={"display_order": index})
        for index, node in enumerate(output_nodes)
    )
    output_body = body.model_copy(update={"nodes": reindexed_nodes})
    ledger = EditorialOperationLedgerComponentBody(
        canonical_turn_set_component_id=body.canonical_turn_set_component_id,
        speaker_review_component_id=body.speaker_review_component_id,
        creator="jake-tools:deterministic-editorial-reflow-v1",
        config_hash=config_hash,
        operations=tuple(operations),
    )
    return EditorialReflowResult(
        body=output_body,
        ledger=ledger,
        merged_node_count=merged_node_count,
    )


def transform_reflow(
    store: BundleStore,
    *,
    run_id: RunId,
    max_gap_ms: int = 3000,
    lexical_gap_ms: int = 1000,
) -> ReflowOutcome:
    """Append durable reflow revisions until the canonical turns are stable."""
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise ValueError("this bundle has no assembled document to reflow")
    turn_set = canonical_turn_set(document.components)
    if turn_set is None:
        raise ValueError("this document has no single canonical timed turn set")

    results = reflow_until_stable(
        turn_set.turns,
        SpeakerContext.from_components(document.components),
        max_gap_ms=max_gap_ms,
        lexical_gap_ms=lexical_gap_ms,
    )
    if not results:
        return ReflowOutcome(
            revision=None,
            turn_set=turn_set,
            ledger=None,
            merged_turn_count=0,
        )

    config_hash = _sha256_text(
        f"reflow-v1:max-gap-ms={max_gap_ms}:lexical-gap-ms={lexical_gap_ms}"
    )
    revision: RevisionRecord | None = None
    ledger: TextEditLedgerComponent | None = None
    merged_turn_count = 0
    for pass_number, result in enumerate(results, start=1):
        new_turn_set = store.add_component(
            TimedTurnSetComponentBody(
                source_artefact_ids=turn_set.source_artefact_ids,
                coordinate_domain=turn_set.coordinate_domain,
                turns=result.turns,
            )
        )
        assert isinstance(new_turn_set, TimedTurnSetComponent)
        added_ledger = store.add_component(
            TextEditLedgerComponentBody(
                mode=TextEditMode.REFLOW,
                input_turn_set_component_id=turn_set.component_id,
                output_turn_set_component_id=new_turn_set.component_id,
                editor="jake-tools:deterministic-reflow-v1",
                config_hash=config_hash,
                entries=result.entries,
            )
        )
        assert isinstance(added_ledger, TextEditLedgerComponent)
        ledger = added_ledger
        revision = store.append_revision(
            operation=OperationRef(
                kind="reflow",
                input_ids=(turn_set.component_id,),
                config_hash=config_hash,
                rationale=(
                    "join reviewed speaker continuations before text correction "
                    f"(pass {pass_number})"
                ),
            ),
            parent_revision_ids=(document.revision_id,),
            component_ids=(new_turn_set.component_id, ledger.component_id),
            superseded_component_ids=(
                turn_set.component_id,
                *(
                    component.component_id
                    for component in document.components.values()
                    if isinstance(component, (ChapterSetComponent, MinutesComponent))
                ),
            ),
        )
        store.update_head(run_id=run_id, revision_id=revision.revision_id)
        merged_turn_count += result.merged_turn_count
        turn_set = new_turn_set
        projected = project_head(store)
        assert not isinstance(projected, NoDocumentYet)
        document = projected

    assert revision is not None
    assert ledger is not None
    return ReflowOutcome(
        revision=revision,
        turn_set=turn_set,
        ledger=ledger,
        merged_turn_count=merged_turn_count,
    )
