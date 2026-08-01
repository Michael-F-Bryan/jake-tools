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
from .ids import RunId
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


def transform_reflow(
    store: BundleStore,
    *,
    run_id: RunId,
    max_gap_ms: int = 3000,
    lexical_gap_ms: int = 1000,
) -> ReflowOutcome:
    """Append one durable reflow revision, or do nothing when already stable."""
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise ValueError("this bundle has no assembled document to reflow")
    turn_set = canonical_turn_set(document.components)
    if turn_set is None:
        raise ValueError("this document has no single canonical timed turn set")

    result = reflow_turns(
        turn_set.turns,
        SpeakerContext.from_components(document.components),
        max_gap_ms=max_gap_ms,
        lexical_gap_ms=lexical_gap_ms,
    )
    if result.merged_turn_count == 0:
        return ReflowOutcome(
            revision=None,
            turn_set=turn_set,
            ledger=None,
            merged_turn_count=0,
        )

    new_turn_set = store.add_component(
        TimedTurnSetComponentBody(
            source_artefact_ids=turn_set.source_artefact_ids,
            coordinate_domain=turn_set.coordinate_domain,
            turns=result.turns,
        )
    )
    assert isinstance(new_turn_set, TimedTurnSetComponent)
    config_hash = _sha256_text(
        f"reflow-v1:max-gap-ms={max_gap_ms}:lexical-gap-ms={lexical_gap_ms}"
    )
    ledger = store.add_component(
        TextEditLedgerComponentBody(
            mode=TextEditMode.REFLOW,
            input_turn_set_component_id=turn_set.component_id,
            output_turn_set_component_id=new_turn_set.component_id,
            editor="jake-tools:deterministic-reflow-v1",
            config_hash=config_hash,
            entries=result.entries,
        )
    )
    assert isinstance(ledger, TextEditLedgerComponent)
    revision = store.append_revision(
        operation=OperationRef(
            kind="reflow",
            input_ids=(turn_set.component_id,),
            config_hash=config_hash,
            rationale="join reviewed speaker continuations before text correction",
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
    return ReflowOutcome(
        revision=revision,
        turn_set=new_turn_set,
        ledger=ledger,
        merged_turn_count=result.merged_turn_count,
    )
