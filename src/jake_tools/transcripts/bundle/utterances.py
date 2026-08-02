"""Reviewed semantic utterance plans for safe transcript repartitioning."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import TranscriptError
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
from .records import OperationRef, RevisionRecord, RunState
from .store import BundleStore

_SCHEMA = "v1"


class UtterancePlanError(TranscriptError):
    pass


class UtteranceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_turn_ids: tuple[str, ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ref: str = ""
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_nonblank_review_text(self) -> UtteranceGroup:
        if not self.text.strip():
            raise ValueError("utterance text cannot be blank.")
        if not self.rationale.strip():
            raise ValueError("utterance rationale cannot be blank.")
        return self


class UtterancePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = _SCHEMA
    bundle_id: str
    input_revision_id: str
    input_turn_set_component_id: str
    input_turn_set_content_hash: str
    reviewer: str = ""
    groups: tuple[UtteranceGroup, ...]


class UtteranceOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    revision: RevisionRecord
    turn_set: TimedTurnSetComponent
    ledger: TextEditLedgerComponent
    merged_turn_count: int


def export_utterance_plan(store: BundleStore, *, destination: Path) -> Path:
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise UtterancePlanError("this bundle has no assembled document.")
    turns = canonical_turn_set(document.components)
    if turns is None:
        raise UtterancePlanError("this document has no canonical transcript.")
    plan = UtterancePlan(
        bundle_id=document.bundle_id,
        input_revision_id=document.revision_id,
        input_turn_set_component_id=turns.component_id,
        input_turn_set_content_hash=turns.content_hash,
        groups=tuple(
            UtteranceGroup(
                input_turn_ids=(turn.turn_id,),
                text=turn.text,
                rationale="retain semantic utterance",
            )
            for turn in turns.turns
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return destination


def apply_utterance_plan(store: BundleStore, *, plan_path: Path) -> UtteranceOutcome:
    raw = plan_path.read_bytes()
    try:
        plan = UtterancePlan.model_validate_json(raw)
    except ValueError as exc:
        raise UtterancePlanError(f"invalid utterance plan: {exc}") from exc
    if plan.schema_version != _SCHEMA or not plan.reviewer.strip():
        raise UtterancePlanError("utterance plan requires schema v1 and a reviewer.")
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise UtterancePlanError("this bundle has no assembled document.")
    turn_set = canonical_turn_set(document.components)
    if turn_set is None:
        raise UtterancePlanError("this document has no canonical transcript.")
    if (
        plan.bundle_id,
        plan.input_revision_id,
        plan.input_turn_set_component_id,
        plan.input_turn_set_content_hash,
    ) != (
        document.bundle_id,
        document.revision_id,
        turn_set.component_id,
        turn_set.content_hash,
    ):
        raise UtterancePlanError(
            "utterance plan is stale or belongs to other input bytes."
        )

    turns = turn_set.turns
    by_id = {turn.turn_id: turn for turn in turns}
    flattened = tuple(
        turn_id for group in plan.groups for turn_id in group.input_turn_ids
    )
    expected = tuple(turn.turn_id for turn in turns)
    if flattened != expected:
        raise UtterancePlanError(
            "utterance groups must be an exact ordered partition of all input turns."
        )
    context = SpeakerContext.from_components(document.components)
    output_turns: list[TimedTurn] = []
    entries: list[TextEditEntry] = []
    for group in plan.groups:
        source = [by_id[turn_id] for turn_id in group.input_turn_ids]
        if not source:
            raise UtterancePlanError("utterance groups cannot be empty.")
        if len({turn.source_artefact_id for turn in source}) != 1:
            raise UtterancePlanError(
                "an utterance group cannot cross source recordings."
            )
        participants = {
            assignment.participant_id
            for turn in source
            if (assignment := context.assignment_for(turn)).is_reviewed
            and assignment.participant_id is not None
        }
        if len(participants) > 1:
            raise UtterancePlanError(
                f"cross-named-speaker merge refused for {group.input_turn_ids}."
            )
        has_unclear = any(
            not context.assignment_for(turn).is_reviewed
            or context.assignment_for(turn).participant_id is None
            for turn in source
        )
        if len(source) > 1 and has_unclear and not group.evidence_ref.strip():
            raise UtterancePlanError(
                f"named-plus-unclear merge {group.input_turn_ids} requires evidence_ref."
            )
        if _content_key(group.text) != _content_key(
            "".join(turn.text for turn in source)
        ):
            raise UtterancePlanError(
                f"group {group.input_turn_ids} changes lexical content; use a correction pack."
            )
        first = source[0]
        output = first.model_copy(
            update={
                "end_ms": max(turn.end_ms for turn in source),
                "text": group.text.strip(),
            }
        )
        output_turns.append(output)
        operation = (
            TextEditOperation.MERGE
            if len(source) > 1
            else (
                TextEditOperation.IDENTITY
                if output.text == first.text
                else TextEditOperation.TEXT_EDIT
            )
        )
        entries.append(
            TextEditEntry(
                operation=operation,
                input_turn_ids=group.input_turn_ids,
                output_turn_ids=(output.turn_id,),
                old_text_sha256=hashlib.sha256(
                    "\n".join(t.text for t in source).encode()
                ).hexdigest(),
                new_text_sha256=hashlib.sha256(output.text.encode()).hexdigest(),
                evidence_ref=group.evidence_ref or group.rationale,
            )
        )

    output_set = store.add_component(
        TimedTurnSetComponentBody(
            source_artefact_ids=turn_set.source_artefact_ids,
            coordinate_domain=turn_set.coordinate_domain,
            turns=tuple(output_turns),
        )
    )
    assert isinstance(output_set, TimedTurnSetComponent)
    digest = hashlib.sha256(raw).hexdigest()
    ledger = store.add_component(
        TextEditLedgerComponentBody(
            mode=TextEditMode.REFLOW,
            input_turn_set_component_id=turn_set.component_id,
            output_turn_set_component_id=output_set.component_id,
            editor=f"jake-tools:reviewed-utterance-plan-v1:{plan.reviewer.strip()}",
            config_hash=digest,
            entries=tuple(entries),
        )
    )
    assert isinstance(ledger, TextEditLedgerComponent)
    run = store.create_run(next_action=OperationRef(kind="utterance-plan"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    try:
        revision = store.append_revision(
            operation=OperationRef(
                kind="utterance-plan",
                input_ids=(turn_set.component_id,),
                config_hash=digest,
                rationale=f"apply reviewed utterance plan by {plan.reviewer.strip()}",
            ),
            parent_revision_ids=(document.revision_id,),
            component_ids=(output_set.component_id, ledger.component_id),
            superseded_component_ids=(
                turn_set.component_id,
                *(
                    c.component_id
                    for c in document.components.values()
                    if isinstance(c, (ChapterSetComponent, MinutesComponent))
                ),
            ),
        )
        store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
        store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)
    except Exception:
        store.release_lease(run_id=run.run_id, new_state=RunState.FAILED)
        raise
    return UtteranceOutcome(
        revision=revision,
        turn_set=output_set,
        ledger=ledger,
        merged_turn_count=len(turns) - len(output_turns),
    )


def _content_key(text: str) -> str:
    return "".join(re.findall(r"[\w']+", text.casefold())).replace("_", "")
