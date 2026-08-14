"""Guarded, auditable transcript correction packs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import TranscriptError
from .assignment import canonical_turn_set
from .components import (
    ChapterSetComponent,
    EditorialOperationLedgerComponent,
    EditorialTranscriptComponent,
    MinutesComponent,
    TextEditEntry,
    TextEditLedgerComponent,
    TextEditLedgerComponentBody,
    TextEditMode,
    TextEditOperation,
    TimedTurnSetComponent,
    TimedTurnSetComponentBody,
)
from .document import NoDocumentYet, project_head
from .records import OperationRef, RevisionRecord, RunState
from .store import BundleStore

_SCHEMA = "v1"


class CorrectionPackError(TranscriptError):
    pass


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    turn_id: str
    old_text: str
    new_text: str = Field(min_length=1)
    evidence_ref: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_material_edit(self) -> Correction:
        if not self.new_text.strip():
            raise ValueError("replacement text cannot be blank.")
        if self.new_text == self.old_text:
            raise ValueError("replacement text must differ from expected text.")
        if not self.evidence_ref.strip() or not self.rationale.strip():
            raise ValueError("evidence_ref and rationale cannot be blank.")
        return self


class CorrectionPack(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = _SCHEMA
    bundle_id: str
    revision_id: str
    turn_set_component_id: str
    turn_set_content_hash: str
    reviewer: str = ""
    corrections: tuple[Correction, ...] = ()


class CorrectionOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    revision: RevisionRecord
    turn_set: TimedTurnSetComponent
    ledger: TextEditLedgerComponent
    changed_turn_count: int


def export_correction_pack(store: BundleStore, *, destination: Path) -> Path:
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise CorrectionPackError("this bundle has no assembled document.")
    turns = canonical_turn_set(document.components)
    if turns is None:
        raise CorrectionPackError("this document has no canonical transcript.")
    pack = CorrectionPack(
        bundle_id=document.bundle_id,
        revision_id=document.revision_id,
        turn_set_component_id=turns.component_id,
        turn_set_content_hash=turns.content_hash,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    return destination


def apply_correction_pack(store: BundleStore, *, pack_path: Path) -> CorrectionOutcome:
    raw = pack_path.read_bytes()
    try:
        pack = CorrectionPack.model_validate_json(raw)
    except ValueError as exc:
        raise CorrectionPackError(f"invalid correction pack: {exc}") from exc
    if (
        pack.schema_version != _SCHEMA
        or not pack.corrections
        or not pack.reviewer.strip()
    ):
        raise CorrectionPackError(
            "correction pack must use schema v1, name a reviewer, and contain corrections."
        )
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise CorrectionPackError("this bundle has no assembled document.")
    turns = canonical_turn_set(document.components)
    if turns is None:
        raise CorrectionPackError("this document has no canonical transcript.")
    if (
        pack.bundle_id,
        pack.revision_id,
        pack.turn_set_component_id,
        pack.turn_set_content_hash,
    ) != (
        document.bundle_id,
        document.revision_id,
        turns.component_id,
        turns.content_hash,
    ):
        raise CorrectionPackError(
            "correction pack is stale or belongs to another bundle."
        )
    corrections = {item.turn_id: item for item in pack.corrections}
    if len(corrections) != len(pack.corrections):
        raise CorrectionPackError("correction pack repeats a turn ID.")
    known = {turn.turn_id: turn for turn in turns.turns}
    missing = sorted(set(corrections) - set(known))
    if missing:
        raise CorrectionPackError(
            f"correction pack references unknown turns: {missing}."
        )
    for turn_id, item in corrections.items():
        if known[turn_id].text != item.old_text:
            raise CorrectionPackError(f"old-text guard failed for {turn_id}.")

    output_turns = tuple(
        turn.model_copy(update={"text": corrections[turn.turn_id].new_text.strip()})
        if turn.turn_id in corrections
        else turn
        for turn in turns.turns
    )
    output = store.add_component(
        TimedTurnSetComponentBody(
            source_artefact_ids=turns.source_artefact_ids,
            coordinate_domain=turns.coordinate_domain,
            turns=output_turns,
        )
    )
    assert isinstance(output, TimedTurnSetComponent)
    entries = []
    for before, after in zip(turns.turns, output_turns, strict=True):
        item = corrections.get(before.turn_id)
        entries.append(
            TextEditEntry(
                operation=(
                    TextEditOperation.TEXT_EDIT if item else TextEditOperation.IDENTITY
                ),
                input_turn_ids=(before.turn_id,),
                output_turn_ids=(after.turn_id,),
                old_text_sha256=hashlib.sha256(before.text.encode()).hexdigest(),
                new_text_sha256=hashlib.sha256(after.text.encode()).hexdigest(),
                evidence_ref=item.evidence_ref if item else "identity",
            )
        )
    digest = hashlib.sha256(raw).hexdigest()
    ledger = store.add_component(
        TextEditLedgerComponentBody(
            mode=TextEditMode.CORRECT,
            input_turn_set_component_id=turns.component_id,
            output_turn_set_component_id=output.component_id,
            editor="jake-tools:guarded-correction-pack-v1",
            config_hash=digest,
            entries=tuple(entries),
        )
    )
    assert isinstance(ledger, TextEditLedgerComponent)
    run = store.create_run(next_action=OperationRef(kind="correction-pack"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    try:
        revision = store.append_revision(
            operation=OperationRef(
                kind="correction-pack",
                input_ids=(turns.component_id,),
                config_hash=digest,
                rationale=(
                    f"apply guarded corrections reviewed by {pack.reviewer.strip()} "
                    f"from {pack_path.name}"
                ),
            ),
            parent_revision_ids=(document.revision_id,),
            component_ids=(output.component_id, ledger.component_id),
            superseded_component_ids=(
                turns.component_id,
                *(
                    c.component_id
                    for c in document.components.values()
                    if isinstance(
                        c,
                        (
                            ChapterSetComponent,
                            EditorialOperationLedgerComponent,
                            EditorialTranscriptComponent,
                            MinutesComponent,
                        ),
                    )
                ),
            ),
        )
        store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
        store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)
    except Exception:
        store.release_lease(run_id=run.run_id, new_state=RunState.FAILED)
        raise
    return CorrectionOutcome(
        revision=revision,
        turn_set=output,
        ledger=ledger,
        changed_turn_count=len(corrections),
    )
