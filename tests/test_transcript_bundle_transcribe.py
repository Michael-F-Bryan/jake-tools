"""M11 transcribe-transform behaviour tests.

The fake worker at the subprocess seam constructs its output through the
REAL ``inference_worker.models`` (``inference_worker_shim.py``), so a
change to the actual frozen contract breaks this suite immediately
rather than silently drifting out of sync with
``bundle/worker_contract.py``'s own mirrored wire types. Covers: real
request/response JSON round-trip, hash verification, M11's partial-
failure promotion rule, resume-by-hash reuse (the zero-invocation case),
and ``--force``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fixtures_bundle import registered_source_and_artefact
from inference_worker_shim import models as real

from jake_tools.transcripts.bundle.assemble import assemble
from jake_tools.transcripts.bundle.components import (
    ArtefactSelection,
    Disposition,
    MediaRecordingComponentBody,
    ParticipantDeclarationSource,
    ParticipantRecord,
    ParticipantSetComponentBody,
    ParticipantStatus,
)
from jake_tools.transcripts.bundle.document import NoDocumentYet, project_head
from jake_tools.transcripts.bundle.ids import mint_id
from jake_tools.transcripts.bundle.records import OperationRef, RunState
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.store import BundleStore
from jake_tools.transcripts.bundle.transcribe import (
    NoDocumentToTranscribeError,
    NoTranscribableMediaError,
    WireInferenceRequest,
    WorkerInvocationError,
    WorkerResponseHashMismatchError,
    transcribe_media,
)


@dataclass
class FakeWorker:
    """A fake at the subprocess seam: parses the request jake-tools wrote
    exactly as the real worker's CLI would, and constructs its
    response/asr.json/diarisation.json through the REAL
    ``inference_worker.models`` (contract-drift-sensitive).

    ``requests`` retains each call's *parsed* request (not just the raw
    command/path) -- ``transcribe_media`` cleans up its own per-attempt
    temp directory before returning, so a test inspecting what was
    actually requested must capture it here, during the call, rather
    than re-reading a path that no longer exists afterwards.
    """

    asr_status: str = "completed"
    diarisation_status: str = "completed"
    calls: list[list[str]] = field(default_factory=list)
    requests: list[WireInferenceRequest] = field(default_factory=list)

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        request_path = Path(command[8])
        out_dir = Path(command[10])
        out_dir.mkdir(parents=True, exist_ok=True)
        wire_request = WireInferenceRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        self.requests.append(wire_request)
        real_request = real.InferenceRequest(
            request_id=wire_request.request_id,
            audio=real.AudioArtefact(
                path=wire_request.audio.path, sha256=wire_request.audio.sha256
            ),
            asr_model=real.ModelIdentity(
                name=wire_request.asr_model.name, version=wire_request.asr_model.version
            ),
            diarisation_model=real.ModelIdentity(
                name=wire_request.diarisation_model.name,
                version=wire_request.diarisation_model.version,
            ),
            runtime_provenance=real.RuntimeProvenance(
                python_version="3.12",
                ml_framework_versions={},
                worker_package_version="0.1.0",
                dependency_lockfile_sha256="0" * 64,
            ),
            speaker_constraints=real.SpeakerConstraints(
                min_speakers=wire_request.speaker_constraints.min_speakers,
                max_speakers=wire_request.speaker_constraints.max_speakers,
            ),
            stage_timeouts=real.StageTimeouts(prepare_s=60, asr_s=60, diarise_s=60),
        )
        observations = real.StageObservations(wall_time_ms=5)

        if self.asr_status == "completed":
            asr_result = real.AsrStageResult(
                status="completed",
                config_hash="asr-cfg",
                observations=observations,
                model_provenance=real.ModelProvenance(
                    identity=real.ModelIdentity(
                        name=real_request.asr_model.name, version="rev-asr"
                    ),
                    package_versions={},
                ),
                output=real.AsrOutput(text="hello world", tokens=[]),
            )
        else:
            asr_result = real.AsrStageResult(
                status="failed",
                config_hash="asr-cfg",
                observations=observations,
                error=real.StageError(
                    error_class="test-failure",
                    message="synthetic ASR failure",
                    retryable=True,
                ),
            )

        if self.diarisation_status == "completed":
            diarisation_result = real.DiarisationStageResult(
                status="completed",
                config_hash="diar-cfg",
                observations=observations,
                model_provenance=real.ModelProvenance(
                    identity=real.ModelIdentity(
                        name=real_request.diarisation_model.name, version="rev-diar"
                    ),
                    package_versions={},
                ),
                output=real.DiarisationOutput(segments=[]),
            )
        else:
            diarisation_result = real.DiarisationStageResult(
                status="failed",
                config_hash="diar-cfg",
                observations=observations,
                error=real.StageError(
                    error_class="test-failure",
                    message="synthetic diarisation failure",
                    retryable=True,
                ),
            )

        asr_path = out_dir / "asr.json"
        diarisation_path = out_dir / "diarisation.json"
        asr_path.write_text(asr_result.model_dump_json(indent=2), encoding="utf-8")
        diarisation_path.write_text(
            diarisation_result.model_dump_json(indent=2), encoding="utf-8"
        )

        response = real.InferenceResponse(
            request_id=real_request.request_id,
            audio=real_request.audio,
            prepare=real.PrepareStageResult(
                status="completed",
                config_hash="prep-cfg",
                source_sha256=real_request.audio.sha256,
                observations=observations,
                output=real.PreparedAudio(
                    path="/tmp/p.wav", sha256="1" * 64, duration_ms=1000
                ),
            ),
            asr=real.ArtefactRef(
                filename="asr.json",
                sha256=hashlib.sha256(asr_path.read_bytes()).hexdigest(),
                status=self.asr_status,
            ),
            diarisation=real.ArtefactRef(
                filename="diarisation.json",
                sha256=hashlib.sha256(diarisation_path.read_bytes()).hexdigest(),
                status=self.diarisation_status,
            ),
            runtime=real_request.runtime_provenance,
        )
        (out_dir / "response.json").write_text(
            response.model_dump_json(indent=2), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")


def _assembled_bundle_with_media(
    tmp_path: Path, *, participant_count: int | None = 2
) -> tuple[BundleStore, str]:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    audio_artefact = registered_source_and_artefact(
        store,
        content=b"fake audio bytes",
        kind="audio",
        acquisition_locator="/tmp/audio.wav",
    )
    media = store.add_component(
        MediaRecordingComponentBody(
            source_artefact_id=audio_artefact.artefact_id,
            media_path="/tmp/audio.wav",
            duration_ms=50_000,
            codec="opus",
            sample_rate_hz=48000,
            channels=1,
        )
    )
    component_ids = [media.component_id]
    if participant_count is not None:
        participants = store.add_component(
            ParticipantSetComponentBody(
                participants=tuple(
                    ParticipantRecord(
                        participant_id=mint_id("participant"),
                        declaration_source=ParticipantDeclarationSource.OPERATOR,
                        declaration_evidence="test",
                        display_names=(f"P{i}",),
                        status=ParticipantStatus.DECLARED,
                    )
                    for i in range(participant_count)
                )
            )
        )
        component_ids.append(participants.component_id)

    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=(
            ArtefactSelection(
                artefact_id=audio_artefact.artefact_id,
                dispositions=(Disposition.MEDIA,),
            ),
        ),
        rationale="test fixture",
        component_ids=tuple(component_ids),
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)
    return store, audio_artefact.artefact_id


def _new_run(store: BundleStore, *, kind: str = "transcribe") -> str:
    run = store.create_run(next_action=OperationRef(kind=kind))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    return run.run_id


# -- happy path: both stages complete, capabilities promote -----------------


def test_transcribe_media_promotes_both_capabilities_and_declares_speaker_max(
    tmp_path: Path,
) -> None:
    store, media_artefact_id = _assembled_bundle_with_media(
        tmp_path, participant_count=2
    )
    worker = FakeWorker()
    run_id = _new_run(store)

    outcome = transcribe_media(store, run_id=run_id, subprocess_runner=worker)

    assert outcome.worker_invocations == 1
    assert outcome.completed_asr_media_ids == (media_artefact_id,)
    assert outcome.completed_diarisation_media_ids == (media_artefact_id,)

    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)
    assert (
        document.capability(CapabilityKey.INFERENCE_ASR).status
        == CapabilityStatus.PRESENT_VALIDATED
    )
    assert (
        document.capability(CapabilityKey.INFERENCE_DIARISATION).status
        == CapabilityStatus.PRESENT_VALIDATED
    )

    # M11: "min 1, max = declared attendee count when known" -- 2 declared.
    assert worker.requests[0].speaker_constraints.min_speakers == 1
    assert worker.requests[0].speaker_constraints.max_speakers == 2


# -- M11 partial failure: completed ASR promotes even when diarisation fails


def test_transcribe_media_partial_failure_promotes_asr_but_not_diarisation(
    tmp_path: Path,
) -> None:
    store, media_artefact_id = _assembled_bundle_with_media(tmp_path)
    worker = FakeWorker(diarisation_status="failed")
    run_id = _new_run(store)

    outcome = transcribe_media(store, run_id=run_id, subprocess_runner=worker)

    assert outcome.completed_asr_media_ids == (media_artefact_id,)
    assert outcome.failed_diarisation_media_ids == (media_artefact_id,)
    assert outcome.revision is not None

    # The revision still became head -- a failed sibling stage never
    # blocks the whole run/document (M11).
    manifest = store.load_manifest()
    assert manifest.head_revision_id == outcome.revision.revision_id

    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)
    assert (
        document.capability(CapabilityKey.INFERENCE_ASR).status
        == CapabilityStatus.PRESENT_VALIDATED
    )
    # No DiarisationResultComponent was ever created for a failed stage
    # (see its docstring) -- the capability is honestly absent, not a
    # blocking `failed`.
    assert document.capability(CapabilityKey.INFERENCE_DIARISATION).status == (
        CapabilityStatus.ABSENT
    )


# -- resume-by-hash: a second call over unchanged inputs is a no-op --------


def test_transcribe_media_second_call_reuses_with_zero_invocations(
    tmp_path: Path,
) -> None:
    store, _media_artefact_id = _assembled_bundle_with_media(tmp_path)
    worker = FakeWorker()

    first_run_id = _new_run(store)
    first_outcome = transcribe_media(
        store, run_id=first_run_id, subprocess_runner=worker
    )
    assert first_outcome.worker_invocations == 1
    store.release_lease(run_id=first_run_id, new_state=RunState.COMPLETED)

    second_run_id = _new_run(store)
    second_outcome = transcribe_media(
        store, run_id=second_run_id, subprocess_runner=worker
    )

    assert second_outcome.worker_invocations == 0
    assert second_outcome.revision is None
    assert len(worker.calls) == 1  # the worker subprocess was never invoked again


def test_transcribe_media_force_reinvokes_despite_matching_fingerprint(
    tmp_path: Path,
) -> None:
    store, _media_artefact_id = _assembled_bundle_with_media(tmp_path)
    worker = FakeWorker()

    first_run_id = _new_run(store)
    transcribe_media(store, run_id=first_run_id, subprocess_runner=worker)
    store.release_lease(run_id=first_run_id, new_state=RunState.COMPLETED)

    second_run_id = _new_run(store)
    forced_outcome = transcribe_media(
        store, run_id=second_run_id, subprocess_runner=worker, force=True
    )

    assert forced_outcome.worker_invocations == 1
    assert len(worker.calls) == 2
    assert forced_outcome.revision is not None
    assert "force=True" in forced_outcome.revision.operation.rationale


# -- refusals ---------------------------------------------------------------


def test_transcribe_media_refuses_without_an_assembled_document(tmp_path: Path) -> None:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    run_id = _new_run(store)

    with pytest.raises(NoDocumentToTranscribeError):
        transcribe_media(store, run_id=run_id, subprocess_runner=FakeWorker())


def test_transcribe_media_refuses_with_no_media_in_the_document(tmp_path: Path) -> None:
    store = BundleStore(tmp_path / "bundle")
    store.create_bundle()
    artefact = registered_source_and_artefact(store, content=b"notes only")
    run = store.create_run(next_action=OperationRef(kind="assemble"))
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=(
            ArtefactSelection(
                artefact_id=artefact.artefact_id,
                dispositions=(Disposition.EVIDENCE_ONLY,),
            ),
        ),
        rationale="no media at all",
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    with pytest.raises(NoTranscribableMediaError):
        transcribe_media(store, run_id=_new_run(store), subprocess_runner=FakeWorker())


def test_transcribe_media_raises_when_the_worker_writes_no_response(
    tmp_path: Path,
) -> None:
    store, _media_artefact_id = _assembled_bundle_with_media(tmp_path)

    def _silent_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, returncode=1, stdout="", stderr="boom"
        )

    with pytest.raises(WorkerInvocationError):
        transcribe_media(
            store, run_id=_new_run(store), subprocess_runner=_silent_runner
        )


def test_transcribe_media_raises_on_a_hash_mismatch_in_the_response(
    tmp_path: Path,
) -> None:
    store, _media_artefact_id = _assembled_bundle_with_media(tmp_path)

    def _lying_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        request_path = Path(command[8])
        out_dir = Path(command[10])
        out_dir.mkdir(parents=True, exist_ok=True)
        wire_request = WireInferenceRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        # Echo a DIFFERENT audio hash than requested -- a genuine contract
        # violation this must refuse, never silently accept.
        response = {
            "request_id": wire_request.request_id,
            "audio": {"path": wire_request.audio.path, "sha256": "f" * 64},
            "prepare": {
                "status": "completed",
                "config_hash": "x",
                "source_sha256": wire_request.audio.sha256,
                "observations": {"wall_time_ms": 1},
                "output": {"path": "/tmp/p.wav", "sha256": "1" * 64, "duration_ms": 1},
            },
            "asr": {"filename": "asr.json", "sha256": "0" * 64, "status": "completed"},
            "diarisation": {
                "filename": "diarisation.json",
                "sha256": "0" * 64,
                "status": "completed",
            },
            "runtime": {
                "python_version": "3.12",
                "ml_framework_versions": {},
                "worker_package_version": "0.1.0",
                "dependency_lockfile_sha256": "0" * 64,
            },
        }
        import json

        (out_dir / "response.json").write_text(json.dumps(response), encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode=0, stdout="", stderr="")

    with pytest.raises(WorkerResponseHashMismatchError):
        transcribe_media(store, run_id=_new_run(store), subprocess_runner=_lying_runner)
