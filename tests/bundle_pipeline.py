"""Build a bundle that already carries inference evidence, without models.

The Phase 3-B/3-C transforms all start from "a document with a combined
timeline and raw ASR/diarisation artefacts". Getting there through the
real path means running the pinned worker, which is a minutes-long,
model-gated operation -- so these helpers write the *same artefacts the
worker writes*, validated against the mirrored wire types
(``worker_contract``), and assemble them exactly as ``transcribe.py``
does.

Deliberately not a ``test_*.py`` module: pytest does not collect it, and
every test that needs a realistic starting document imports from here
rather than open-coding a nine-step setup.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jake_tools.transcripts.bundle.assemble import assemble
from jake_tools.transcripts.bundle.components import (
    ArtefactSelection,
    AsrResultComponent,
    AsrResultComponentBody,
    DiarisationResultComponent,
    DiarisationResultComponentBody,
    Disposition,
    MediaRecordingComponent,
    MediaRecordingComponentBody,
    NotesComponentBody,
    NotesKind,
    NotesSectionBody,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponentBody,
    ParticipantStatus,
    RecordingReference,
    RecordingReferenceSetComponentBody,
)
from jake_tools.transcripts.bundle.control import run_timeline_transform
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.records import (
    OperationRef,
    RunState,
    SourceAssociation,
)
from jake_tools.transcripts.bundle.store import BundleStore


@dataclass(frozen=True)
class SpokenToken:
    """One raw ASR token, in its recording's own source domain."""

    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True)
class SpokenSegment:
    """One diarisation segment, in its recording's own source domain."""

    start_ms: int
    end_ms: int
    speaker_label: str


@dataclass(frozen=True)
class Recording:
    name: str
    duration_ms: int
    tokens: tuple[SpokenToken, ...]
    segments: tuple[SpokenSegment, ...]


def tokens_from_utterances(
    utterances: Sequence[tuple[int, int, str]],
) -> tuple[SpokenToken, ...]:
    """Split ``(start_ms, end_ms, text)`` utterances into word-level tokens.

    The worker emits per-word tokens with their own leading whitespace
    (``"".join(token.text)`` rebuilds its transcript), so a fixture that
    handed normalisation whole sentences would exercise a token stream the
    real one never produces.
    """
    tokens: list[SpokenToken] = []
    for start_ms, end_ms, text in utterances:
        words = text.split()
        if not words:
            continue
        step = max(1, (end_ms - start_ms) // len(words))
        for index, word in enumerate(words):
            token_start = start_ms + index * step
            token_end = end_ms if index == len(words) - 1 else token_start + step
            tokens.append(
                SpokenToken(
                    start_ms=token_start,
                    end_ms=max(token_end, token_start + 1),
                    text=(" " if (tokens or index) else "") + word,
                )
            )
    return tuple(tokens)


def _asr_stage_result(tokens: Sequence[SpokenToken]) -> bytes:
    return json.dumps(
        {
            "status": "completed",
            "config_hash": "fixture-asr-config",
            "observations": {"wall_time_ms": 1},
            "model_provenance": {
                "identity": {"name": "fixture/parakeet", "version": "0"},
                "package_versions": {},
            },
            "output": {
                "text": "".join(token.text for token in tokens).strip(),
                "tokens": [
                    {
                        "start_ms": token.start_ms,
                        "end_ms": token.end_ms,
                        "text": token.text,
                        "confidence": 0.9,
                    }
                    for token in tokens
                ],
                "chunk_boundaries_ms": [],
            },
        },
        indent=2,
    ).encode("utf-8")


def _diarisation_stage_result(segments: Sequence[SpokenSegment]) -> bytes:
    return json.dumps(
        {
            "status": "completed",
            "config_hash": "fixture-diarisation-config",
            "observations": {"wall_time_ms": 1},
            "model_provenance": {
                "identity": {"name": "fixture/community-1", "version": "0"},
                "package_versions": {},
            },
            "output": {
                "segments": [
                    {
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "speaker_label": segment.speaker_label,
                    }
                    for segment in segments
                ],
            },
        },
        indent=2,
    ).encode("utf-8")


@dataclass(frozen=True)
class PreparedBundle:
    store: BundleStore
    note_artefact_id: str
    media_artefact_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]


def prepare_transcribed_bundle(
    root: Path,
    *,
    recordings: Sequence[Recording],
    attendees: Sequence[str] = ("Michael Bryan", "Avalon Mann"),
    note_text: str = "## Original notes\n\n- Goal: reduce the rate.\n",
    with_diarisation: bool = True,
) -> PreparedBundle:
    """A bundle at exactly the state ``transform normalise`` expects.

    Ingest -> adapt -> assemble -> timeline, plus ASR/diarisation result
    components that carry the artefacts the real worker would have
    written. The one thing deliberately *not* faked is the shape of those
    artefacts: they are the worker's own JSON, and normalisation parses
    them with the same mirrored wire types production uses.
    """
    store = BundleStore(root)
    store.create_bundle()
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION, evidence="test fixture"
    )
    source_id = membership.source_id

    note_artefact = store.ingest_artefact(
        source_id=source_id,
        content=note_text.encode("utf-8"),
        kind="obsidian-note",
        producer="test",
        acquisition_locator=str(root / "note.md"),
    )
    participants = tuple(
        ParticipantRecord(
            participant_id=mint_id("participant"),
            declaration_source=ParticipantDeclarationSource.NOTE_FRONTMATTER,
            declaration_evidence="note frontmatter Attendees",
            display_names=(attendee,),
            status=ParticipantStatus.DECLARED,
        )
        for attendee in attendees
    )
    participant_set = store.add_component(
        ParticipantSetComponentBody(participants=participants)
    )
    notes = store.add_component(
        NotesComponentBody(
            notes_kind=NotesKind.AUTHORED_PREP,
            source_artefact_id=note_artefact.artefact_id,
            authored=True,
            sections=(NotesSectionBody(title="Original notes", text=note_text),),
        )
    )

    media_artefact_ids: list[str] = []
    component_ids: list[str] = [participant_set.component_id, notes.component_id]
    references: list[RecordingReference] = []
    for recording in recordings:
        media_artefact = store.ingest_artefact(
            source_id=source_id,
            content=f"fake-audio:{recording.name}".encode(),
            kind="audio",
            producer="test",
            acquisition_locator=str(root / recording.name),
        )
        media_artefact_ids.append(media_artefact.artefact_id)
        media = store.add_component(
            MediaRecordingComponentBody(
                source_artefact_id=media_artefact.artefact_id,
                media_path=str(root / recording.name),
                duration_ms=recording.duration_ms,
                codec="opus",
                sample_rate_hz=48_000,
                channels=1,
            )
        )
        assert isinstance(media, MediaRecordingComponent)
        component_ids.append(media.component_id)
        references.append(
            RecordingReference(
                raw_link=f"![[{recording.name}]]",
                resolved_path=str(root / recording.name),
            )
        )

    reference_set = store.add_component(
        RecordingReferenceSetComponentBody(
            note_artefact_id=note_artefact.artefact_id, references=tuple(references)
        )
    )
    component_ids.append(reference_set.component_id)

    run = store.create_run(
        next_action=OperationRef(
            kind="assemble", input_ids=(note_artefact.artefact_id,)
        )
    )
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=(
            ArtefactSelection(
                artefact_id=note_artefact.artefact_id,
                dispositions=(Disposition.DESTINATION, Disposition.NOTES),
            ),
            *(
                ArtefactSelection(
                    artefact_id=artefact_id, dispositions=(Disposition.MEDIA,)
                )
                for artefact_id in media_artefact_ids
            ),
        ),
        rationale="note plus recordings (test fixture)",
        component_ids=tuple(component_ids),
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    run_timeline_transform(store)
    _attach_inference(
        store,
        source_id=source_id,
        recordings=recordings,
        media_artefact_ids=media_artefact_ids,
        with_diarisation=with_diarisation,
    )
    return PreparedBundle(
        store=store,
        note_artefact_id=note_artefact.artefact_id,
        media_artefact_ids=tuple(media_artefact_ids),
        participant_ids=tuple(
            participant.participant_id for participant in participants
        ),
    )


def _attach_inference(
    store: BundleStore,
    *,
    source_id: str,
    recordings: Sequence[Recording],
    media_artefact_ids: Sequence[str],
    with_diarisation: bool,
) -> None:
    """Ingest the worker's own output artefacts and promote them, exactly
    the way ``transcribe.py`` does -- one revision carrying every result
    component plus the artefacts they reference."""
    head = store.document_head()
    assert not isinstance(head, type(None))
    component_ids: list[str] = []
    artefact_ids: list[str] = []
    for recording, media_artefact_id in zip(
        recordings, media_artefact_ids, strict=True
    ):
        asr_bytes = _asr_stage_result(recording.tokens)
        asr_artefact = store.ingest_artefact(
            source_id=source_id,
            content=asr_bytes,
            kind="asr",
            producer="inference-worker",
            acquisition_locator=f"fixture:{recording.name}:asr.json",
            derived_from=(media_artefact_id,),
        )
        artefact_ids.append(asr_artefact.artefact_id)
        asr_component = store.add_component(
            AsrResultComponentBody(
                media_artefact_id=media_artefact_id,
                result_artefact_id=asr_artefact.artefact_id,
                attempt_id=mint_id("attempt"),
                request_fingerprint="0" * 64,
                worker_config_hash="fixture-asr-config",
                model_name="fixture/parakeet",
                model_version="0",
            )
        )
        assert isinstance(asr_component, AsrResultComponent)
        component_ids.append(asr_component.component_id)

        if not with_diarisation:
            continue
        diarisation_bytes = _diarisation_stage_result(recording.segments)
        diarisation_artefact = store.ingest_artefact(
            source_id=source_id,
            content=diarisation_bytes,
            kind="diarisation",
            producer="inference-worker",
            acquisition_locator=f"fixture:{recording.name}:diarisation.json",
            derived_from=(media_artefact_id,),
        )
        artefact_ids.append(diarisation_artefact.artefact_id)
        diarisation_component = store.add_component(
            DiarisationResultComponentBody(
                media_artefact_id=media_artefact_id,
                result_artefact_id=diarisation_artefact.artefact_id,
                attempt_id=mint_id("attempt"),
                request_fingerprint="1" * 64,
                worker_config_hash="fixture-diarisation-config",
                model_name="fixture/community-1",
                model_version="0",
            )
        )
        assert isinstance(diarisation_component, DiarisationResultComponent)
        component_ids.append(diarisation_component.component_id)

    manifest_head = store.document_head()
    assert hasattr(manifest_head, "revision_id")
    revision = store.append_revision(
        operation=OperationRef(kind="transcribe", rationale="test fixture inference"),
        parent_revision_ids=(manifest_head.revision_id,),  # pyright: ignore[reportAttributeAccessIssue]
        artefact_ids=tuple(artefact_ids),
        component_ids=tuple(component_ids),
    )
    run = store.create_run(
        next_action=OperationRef(kind="transcribe", rationale="test fixture inference")
    )
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    store.update_head(run_id=run.run_id, revision_id=revision.revision_id)
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)
