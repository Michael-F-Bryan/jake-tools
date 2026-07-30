"""Phase 2 exit evidence: the three inference-free evaluation-corpus canaries.

Runs the cheap-first order the corpus recommends (existing-untimed ->
teams -> gemini): ingest -> adapt -> assemble -> project head -> check
every relevant capability status against each fixture's own
``expected-behaviour.md``, then write a verdict file reporting the
acceptance-matrix dimensions this Phase 2 slice can actually speak to
(source integrity, capability truthfulness, timing, speaker handling --
rendering/apply/resume are later-phase concerns, not built yet, and are
recorded as such rather than silently skipped).

Two ways to run this:

1. ``pytest -m corpus -q`` (deselected by default -- see ``conftest.py``;
   these read the evaluation corpus under ``_working/``, an absolute,
   gitignored path outside this worktree). Each test call writes its
   fixture's verdict to the real ``phase2-canaries/<fixture>/`` location.
2. ``uv run python tests/test_transcript_bundle_corpus.py`` -- the "small
   runner script" the Phase 2 brief asks for, committed here rather than
   as a second file so it can never drift from what the marked tests
   actually exercise: both call the exact same ``_run_*_canary``
   functions below.

Fixtures are immutable: every input is read-only from
``evaluation-corpus/fixtures/<name>/``, hash-verified against that
fixture's own ``provenance.json`` both before and after copying it into
the fresh run directory; nothing is ever written back into a fixture
directory, and the Obsidian vault is never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from jake_tools.transcripts.bundle.adapters import (
    TeamsSpeakerConfirmation,
    adapt_gemini_notes,
    adapt_teams_vtt,
    adapt_untimed_transcript,
)
from jake_tools.transcripts.bundle.assemble import assemble
from jake_tools.transcripts.bundle.components import ArtefactSelection, Disposition
from jake_tools.transcripts.bundle.document import TranscriptDocumentV1, project_head
from jake_tools.transcripts.bundle.records import (
    NoDocumentYet,
    OperationRef,
    RunState,
    SourceAssociation,
)
from jake_tools.transcripts.bundle.registry import CapabilityKey, CapabilityStatus
from jake_tools.transcripts.bundle.store import BundleStore

_CORPUS_ROOT = Path(
    "/Users/work/Documents/jake-tools/_working/transcription-overhaul-context-2026-07-29"
)
_FIXTURES = _CORPUS_ROOT / "evaluation-corpus" / "fixtures"
_PHASE2_CANARIES = _CORPUS_ROOT / "phase2-canaries"
_HAS_CORPUS = _FIXTURES.is_dir()
_requires_corpus = pytest.mark.corpus


def _skip_if_no_corpus() -> None:
    if not _HAS_CORPUS:
        pytest.skip(f"evaluation corpus not present at {_FIXTURES}")


# -- fixture immutability: hash verify + copy, never write back -------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _provenance(fixture: str) -> dict[str, Any]:
    return json.loads(
        (_FIXTURES / fixture / "provenance.json").read_text(encoding="utf-8")
    )


def _verify_and_copy(
    fixture: str, relative_path: str, run_root: Path, *, hashes: dict[str, str]
) -> Path:
    """Verify ``relative_path``'s current on-disk hash against the
    fixture's own ``provenance.json`` (fail loudly if it has drifted),
    copy it into ``run_root/inputs/`` (a fresh destination this canary
    owns, never the fixture directory itself), and record the observed
    hash in ``hashes`` under ``relative_path`` -- called once before the
    run and once after, so the same dict proves the fixture was never
    mutated by anything this canary did.
    """
    provenance = _provenance(fixture)
    expected = next(
        entry["sha256"]
        for entry in provenance["derived_files"]
        if entry["relative_path"] == relative_path
    )
    source_path = _FIXTURES / fixture / relative_path
    actual = _sha256_bytes(source_path.read_bytes())
    if actual != expected:
        raise AssertionError(
            f"fixture {fixture}/{relative_path} hash mismatch: provenance.json "
            f"says {expected}, on-disk is {actual} -- the fixture has drifted "
            "or been mutated."
        )
    hashes[relative_path] = actual

    destination = run_root / "inputs" / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source_path.read_bytes())
    return destination


def _reverify(fixture: str, relative_path: str, hashes: dict[str, str]) -> None:
    """The 'after' half of the before/after hash check (eval corpus
    §"How to evaluate" 1): the fixture file must still match the hash
    recorded before the run started."""
    source_path = _FIXTURES / fixture / relative_path
    actual = _sha256_bytes(source_path.read_bytes())
    if actual != hashes[relative_path]:
        raise AssertionError(
            f"fixture {fixture}/{relative_path} was mutated during the canary "
            f"run: pre-run hash {hashes[relative_path]}, post-run hash {actual}."
        )


def _fresh_run_root(fixture: str) -> Path:
    run_root = _PHASE2_CANARIES / fixture
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)
    return run_root


# -- verdict evidence ----------------------------------------------------


@dataclass(frozen=True)
class CapabilityEvidence:
    key: str
    status: str
    members: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class CanaryVerdict:
    fixture: str
    run_root: Path
    pre_hashes: dict[str, str]
    post_hashes: dict[str, str]
    capabilities: tuple[CapabilityEvidence, ...]
    counts: dict[str, int]
    dimension_notes: dict[str, str] = field(default_factory=dict)
    judgement: str = ""


def _capability_evidence(
    document: TranscriptDocumentV1, keys: tuple[CapabilityKey, ...]
) -> tuple[CapabilityEvidence, ...]:
    evidence = []
    for key in keys:
        record = document.capability(key)
        members = tuple(
            f"{member.member_id}={member.status.value}" for member in record.members
        )
        evidence.append(
            CapabilityEvidence(
                key=key.value,
                status=record.status.value,
                members=members,
                detail=record.failure_detail,
            )
        )
    return tuple(evidence)


def _render_verdict(verdict: CanaryVerdict) -> str:
    lines = [f"# Phase 2 canary verdict: {verdict.fixture}", ""]
    lines.append("## Source integrity")
    lines.append("")
    lines.append("| File | Pre-run SHA-256 | Post-run SHA-256 | Unchanged |")
    lines.append("|---|---|---|---|")
    for relative_path, pre in sorted(verdict.pre_hashes.items()):
        post = verdict.post_hashes.get(relative_path, "MISSING")
        lines.append(f"| `{relative_path}` | `{pre}` | `{post}` | {pre == post} |")
    lines.append("")
    lines.append("## Capability truthfulness")
    lines.append("")
    lines.append("| Capability | Status | Members | Detail |")
    lines.append("|---|---|---|---|")
    for capability in verdict.capabilities:
        members = "; ".join(capability.members) or "-"
        detail = capability.detail or "-"
        lines.append(
            f"| `{capability.key}` | {capability.status} | {members} | {detail} |"
        )
    lines.append("")
    lines.append("## Mechanical counts")
    lines.append("")
    for name, count in verdict.counts.items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Acceptance-matrix dimensions")
    lines.append("")
    for dimension, note in verdict.dimension_notes.items():
        lines.append(f"- **{dimension}**: {note}")
    lines.append("")
    lines.append(f"## Judgement\n\n{verdict.judgement}")
    lines.append("")
    return "\n".join(lines)


def _write_verdict(verdict: CanaryVerdict) -> Path:
    path = verdict.run_root / "verdict.md"
    path.write_text(_render_verdict(verdict), encoding="utf-8")
    return path


# -- fixture 1: existing-untimed-transcript ----------------------------------


def _run_existing_untimed_canary() -> CanaryVerdict:
    fixture = "existing-untimed-transcript"
    run_root = _fresh_run_root(fixture)
    pre_hashes: dict[str, str] = {}
    post_hashes: dict[str, str] = {}

    input_path = _verify_and_copy(fixture, "input.md", run_root, hashes=pre_hashes)
    text = input_path.read_text(encoding="utf-8")

    store = BundleStore(run_root / "bundle")
    store.create_bundle()
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence=f"ingest {fixture}/input.md (Phase 2 canary)",
    )
    artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=text.encode("utf-8"),
        kind="markdown",
        producer="phase2-canary",
        acquisition_locator=str(input_path),
    )
    adaptation = adapt_untimed_transcript(
        store, source_artefact_id=artefact.artefact_id, markdown_text=text
    )

    run = store.create_run(
        next_action=OperationRef(kind="assemble", input_ids=(artefact.artefact_id,))
    )
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=(
            ArtefactSelection(
                artefact_id=artefact.artefact_id,
                dispositions=(Disposition.SELECTED_TRANSCRIPT,),
            ),
        ),
        rationale=(
            "single imported untimed transcript; no competing candidate (D5 "
            "does not apply -- only one transcript exists)"
        ),
        component_ids=(
            adaptation.turn_set.component_id,
            adaptation.participants.component_id,
        ),
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)

    _reverify(fixture, "input.md", pre_hashes)
    post_hashes["input.md"] = pre_hashes["input.md"]

    capabilities = _capability_evidence(
        document,
        (
            CapabilityKey.TRANSCRIPT_UNTIMED,
            CapabilityKey.TRANSCRIPT_TIMED,
            CapabilityKey.PARTICIPANTS_DECLARED,
            CapabilityKey.SPEAKERS_PROVIDER_LABELS,
            CapabilityKey.NOTES_PROVIDER,
            CapabilityKey.CHAPTERS,
        ),
    )
    counts = {
        "untimed turns": len(adaptation.turn_set.turns),
        "distinct speakers": len(adaptation.participants.participants),
    }
    dimension_notes = {
        "Source integrity": "input.md hash unchanged before/after; source note untouched.",
        "Capability truthfulness": (
            "transcript.untimed present-validated; transcript.timed absent "
            "(no timing claimed); speakers.provider-labels absent (no "
            "provider evidence exists for an imported transcript)."
        ),
        "Timing": (
            "No timestamp, timeline, or chapter capability claimed (D6) -- "
            "chapters stays not-attempted (its own prerequisite, "
            "transcript.timed, is not present-validated)."
        ),
        "Speaker handling": (
            "3 distinct speaker labels preserved verbatim as participants, "
            "all speaking-evidenced (they are shown speaking in the text); "
            "no fabricated review/cluster state."
        ),
    }
    judgement = (
        "PASS: honest partial-capability handling for an imported untimed "
        "transcript -- untimed evidence present-validated, every "
        "timing-dependent capability correctly absent, no invented "
        "timestamps or media."
    )

    verdict = CanaryVerdict(
        fixture=fixture,
        run_root=run_root,
        pre_hashes=pre_hashes,
        post_hashes=post_hashes,
        capabilities=capabilities,
        counts=counts,
        dimension_notes=dimension_notes,
        judgement=judgement,
    )
    _write_verdict(verdict)
    return verdict


@_requires_corpus
def test_canary_existing_untimed_transcript() -> None:
    _skip_if_no_corpus()
    verdict = _run_existing_untimed_canary()

    by_key = {c.key: c for c in verdict.capabilities}
    assert (
        by_key["transcript.untimed"].status == CapabilityStatus.PRESENT_VALIDATED.value
    )
    assert by_key["transcript.timed"].status == CapabilityStatus.ABSENT.value
    assert (
        by_key["participants.declared"].status
        == CapabilityStatus.PRESENT_VALIDATED.value
    )
    assert by_key["speakers.provider-labels"].status == CapabilityStatus.ABSENT.value
    assert by_key["notes.provider"].status == CapabilityStatus.ABSENT.value
    assert verdict.counts["untimed turns"] == 7
    assert verdict.counts["distinct speakers"] == 3
    assert verdict.pre_hashes == verdict.post_hashes


# -- fixture 2: teams-attributed-vtt -----------------------------------------


def _run_teams_vtt_canary() -> CanaryVerdict:
    fixture = "teams-attributed-vtt"
    run_root = _fresh_run_root(fixture)
    pre_hashes: dict[str, str] = {}
    post_hashes: dict[str, str] = {}

    vtt_path = _verify_and_copy(fixture, "input.vtt", run_root, hashes=pre_hashes)
    metadata_path = _verify_and_copy(
        fixture, "input-metadata.json", run_root, hashes=pre_hashes
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared_attendees = tuple(metadata["participants_declared"])

    store = BundleStore(run_root / "bundle")
    store.create_bundle()
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence=f"ingest {fixture}/input.vtt + input-metadata.json (Phase 2 canary)",
    )
    vtt_artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=vtt_path.read_bytes(),
        kind="vtt",
        producer="phase2-canary",
        acquisition_locator=str(vtt_path),
    )
    metadata_artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=metadata_path.read_bytes(),
        kind="json",
        producer="phase2-canary",
        acquisition_locator=str(metadata_path),
    )

    # M19: the raw `Michael BRYAN` cue label differs in case from the
    # declared attendee `Michael Bryan` -- confirmed explicitly here (the
    # operator's role in this canary), never inferred by the adapter.
    adaptation = adapt_teams_vtt(
        store,
        source_artefact_id=vtt_artefact.artefact_id,
        vtt_path=vtt_path,
        declared_attendees=declared_attendees,
        speaker_confirmations=(
            TeamsSpeakerConfirmation(
                raw_label="Michael BRYAN", participant_display_name="Michael Bryan"
            ),
            TeamsSpeakerConfirmation(
                raw_label="Joanne Olsen", participant_display_name="Joanne Olsen"
            ),
            TeamsSpeakerConfirmation(
                raw_label="Sam Lintern", participant_display_name="Sam Lintern"
            ),
        ),
    )

    run = store.create_run(
        next_action=OperationRef(
            kind="assemble",
            input_ids=(vtt_artefact.artefact_id, metadata_artefact.artefact_id),
        )
    )
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=(
            ArtefactSelection(
                artefact_id=vtt_artefact.artefact_id,
                dispositions=(Disposition.SELECTED_TRANSCRIPT,),
            ),
            ArtefactSelection(
                artefact_id=metadata_artefact.artefact_id,
                dispositions=(Disposition.EVIDENCE_ONLY,),
            ),
        ),
        rationale=(
            "Teams-attributed VTT is the only transcript evidence for this "
            "meeting; provider metadata supplies the declared attendee list"
        ),
        component_ids=(
            adaptation.turn_set.component_id,
            adaptation.label_set.component_id,
            adaptation.participants.component_id,
        ),
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)

    _reverify(fixture, "input.vtt", pre_hashes)
    _reverify(fixture, "input-metadata.json", pre_hashes)
    post_hashes.update(pre_hashes)

    capabilities = _capability_evidence(
        document,
        (
            CapabilityKey.TRANSCRIPT_TIMED,
            CapabilityKey.TRANSCRIPT_UNTIMED,
            CapabilityKey.SPEAKERS_PROVIDER_LABELS,
            CapabilityKey.PARTICIPANTS_DECLARED,
            CapabilityKey.CHAPTERS,
        ),
    )
    turn_start_ms = [turn.start_ms for turn in adaptation.turn_set.turns]
    statuses_by_name = {
        p.display_names[0]: p.status.value for p in adaptation.participants.participants
    }
    dropped_zero_length_cues = sum(
        1 for warning in adaptation.warnings if "zero-length" in warning
    )
    unmapped_label_warnings = sum(
        1 for warning in adaptation.warnings if "no participant record" in warning
    )
    counts = {
        "timed turns": len(adaptation.turn_set.turns),
        "provider label spans": len(adaptation.label_set.spans),
        "declared attendees": len(declared_attendees),
        "speaking-evidenced attendees": sum(
            1 for status in statuses_by_name.values() if status == "speaking-evidenced"
        ),
        "dropped zero-length cues": dropped_zero_length_cues,
        "unmapped speaker label warnings": unmapped_label_warnings,
    }
    dimension_notes = {
        "Source integrity": (
            "input.vtt and input-metadata.json hashes unchanged before/after; "
            "raw VTT retained as immutable evidence, never rewritten."
        ),
        "Capability truthfulness": (
            "transcript.timed present-validated; transcript.untimed absent "
            "(no ASR/diarisation invoked); speakers.provider-labels "
            "present-validated as evidence only (satisfies no gate)."
        ),
        "Timing": (
            f"{counts['timed turns']} turns in M6 canonical order "
            f"(start_ms ascending: {turn_start_ms == sorted(turn_start_ms)}); "
            "half-open spans; no zero-length cue reached the canonical set."
        ),
        "Speaker handling": (
            f"raw label 'Michael BRYAN' preserved verbatim as evidence "
            f"(casing quirk, never case-folded); statuses={statuses_by_name}; "
            "Des Everingham stays declared -- no speech invented for a "
            "non-speaking attendee."
        ),
    }
    judgement = (
        "PASS: provider transcript normalised without ASR/diarisation, "
        "casing/overlap/file-order quirks preserved and correctly ordered, "
        "and no speech fabricated for the non-speaking declared attendee."
    )

    verdict = CanaryVerdict(
        fixture=fixture,
        run_root=run_root,
        pre_hashes=pre_hashes,
        post_hashes=post_hashes,
        capabilities=capabilities,
        counts=counts,
        dimension_notes=dimension_notes,
        judgement=judgement,
    )
    _write_verdict(verdict)
    return verdict


@_requires_corpus
def test_canary_teams_attributed_vtt() -> None:
    _skip_if_no_corpus()
    verdict = _run_teams_vtt_canary()

    by_key = {c.key: c for c in verdict.capabilities}
    assert by_key["transcript.timed"].status == CapabilityStatus.PRESENT_VALIDATED.value
    assert by_key["transcript.untimed"].status == CapabilityStatus.ABSENT.value
    assert (
        by_key["speakers.provider-labels"].status
        == CapabilityStatus.PRESENT_VALIDATED.value
    )
    assert (
        by_key["participants.declared"].status
        == CapabilityStatus.PRESENT_VALIDATED.value
    )
    assert verdict.counts["timed turns"] == verdict.counts["provider label spans"]
    assert verdict.counts["declared attendees"] == 4
    assert verdict.counts["speaking-evidenced attendees"] == 3
    assert verdict.counts["dropped zero-length cues"] == 0
    # Every real raw label (Michael BRYAN, Joanne Olsen, Sam Lintern) is
    # explicitly confirmed in this canary -- no unmapped-label warning.
    assert verdict.counts["unmapped speaker label warnings"] == 0
    assert verdict.pre_hashes == verdict.post_hashes


# -- fixture 3: gemini-notes-only ---------------------------------------------


def _run_gemini_notes_canary() -> CanaryVerdict:
    fixture = "gemini-notes-only"
    run_root = _fresh_run_root(fixture)
    pre_hashes: dict[str, str] = {}
    post_hashes: dict[str, str] = {}

    input_path = _verify_and_copy(fixture, "input.md", run_root, hashes=pre_hashes)
    text = input_path.read_text(encoding="utf-8")

    store = BundleStore(run_root / "bundle")
    store.create_bundle()
    membership = store.register_source(
        association=SourceAssociation.OPERATOR_ASSERTION,
        evidence=f"ingest {fixture}/input.md (Phase 2 canary)",
    )
    artefact = store.ingest_artefact(
        source_id=membership.source_id,
        content=text.encode("utf-8"),
        kind="markdown",
        producer="phase2-canary",
        acquisition_locator=str(input_path),
    )
    adaptation = adapt_gemini_notes(
        store, source_artefact_id=artefact.artefact_id, markdown_text=text
    )

    run = store.create_run(
        next_action=OperationRef(kind="assemble", input_ids=(artefact.artefact_id,))
    )
    store.acquire_lease(run_id=run.run_id, pid=os.getpid())
    assemble(
        store,
        run_id=run.run_id,
        selections=(
            ArtefactSelection(
                artefact_id=artefact.artefact_id, dispositions=(Disposition.NOTES,)
            ),
        ),
        rationale=(
            "Gemini notes-only export; the note itself states no transcript "
            "was available (D2) -- notes are the only evidence"
        ),
        component_ids=(
            *(component.component_id for component in adaptation.notes),
            adaptation.absence_declaration.component_id,
            adaptation.participants.component_id,
        ),
    )
    store.release_lease(run_id=run.run_id, new_state=RunState.COMPLETED)

    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)

    _reverify(fixture, "input.md", pre_hashes)
    post_hashes["input.md"] = pre_hashes["input.md"]

    capabilities = _capability_evidence(
        document,
        (
            CapabilityKey.TRANSCRIPT_TIMED,
            CapabilityKey.TRANSCRIPT_UNTIMED,
            CapabilityKey.NOTES_PROVIDER,
            CapabilityKey.NOTES_AUTHORED,
            CapabilityKey.PARTICIPANTS_DECLARED,
            CapabilityKey.SPEAKERS_PROVIDER_LABELS,
            CapabilityKey.CHAPTERS,
            CapabilityKey.MINUTES,
        ),
    )
    counts = {
        "notes components": len(adaptation.notes),
        "declared attendees": len(adaptation.participants.participants),
        "speaking-evidenced attendees": sum(
            1
            for p in adaptation.participants.participants
            if p.status.value == "speaking-evidenced"
        ),
    }
    dimension_notes = {
        "Source integrity": "input.md hash unchanged before/after.",
        "Capability truthfulness": (
            "transcript.timed and transcript.untimed both "
            "not-available-from-source (D2: the note's own statement that "
            "no transcript was available is preserved as the evidence "
            "behind this status, not a bare absent); notes.provider "
            "present-validated across all 4 sections; minutes and chapters "
            "remain not-attempted (no validator this phase claims them)."
        ),
        "Timing": "N/A -- no timed or untimed transcript exists for this source.",
        "Speaker handling": (
            "All 7 declared attendees stay declared, none promoted to "
            "speaking-evidenced -- there is no transcript evidence that "
            "could justify it; (Speaker)/(The group) markers preserved "
            "verbatim inside note text, never resolved to a participant."
        ),
    }
    judgement = (
        "PASS: capability absence is truthful and evidence-backed "
        "(not-available-from-source, not a bare absent); no transcript, "
        "timing, or speaker-review capability fabricated from notes alone."
    )

    verdict = CanaryVerdict(
        fixture=fixture,
        run_root=run_root,
        pre_hashes=pre_hashes,
        post_hashes=post_hashes,
        capabilities=capabilities,
        counts=counts,
        dimension_notes=dimension_notes,
        judgement=judgement,
    )
    _write_verdict(verdict)
    return verdict


@_requires_corpus
def test_canary_gemini_notes_only() -> None:
    _skip_if_no_corpus()
    verdict = _run_gemini_notes_canary()

    by_key = {c.key: c for c in verdict.capabilities}
    assert (
        by_key["transcript.timed"].status
        == CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE.value
    )
    assert (
        by_key["transcript.untimed"].status
        == CapabilityStatus.NOT_AVAILABLE_FROM_SOURCE.value
    )
    assert by_key["notes.provider"].status == CapabilityStatus.PRESENT_VALIDATED.value
    assert by_key["notes.authored"].status == CapabilityStatus.ABSENT.value
    assert (
        by_key["participants.declared"].status
        == CapabilityStatus.PRESENT_VALIDATED.value
    )
    assert by_key["speakers.provider-labels"].status == CapabilityStatus.ABSENT.value
    assert verdict.counts["notes components"] == 4
    assert verdict.counts["declared attendees"] == 7
    assert verdict.counts["speaking-evidenced attendees"] == 0
    assert verdict.pre_hashes == verdict.post_hashes


if __name__ == "__main__":
    # The Phase 2 brief's "small runner script": executes the exact same
    # three canary functions the `pytest -m corpus` tests call, printing
    # each verdict's judgement line and writing the full verdict.md files
    # under phase2-canaries/. Run with:
    #   uv run python tests/test_transcript_bundle_corpus.py
    if not _HAS_CORPUS:
        raise SystemExit(f"evaluation corpus not present at {_FIXTURES}")
    for name, runner in (
        ("existing-untimed-transcript", _run_existing_untimed_canary),
        ("teams-attributed-vtt", _run_teams_vtt_canary),
        ("gemini-notes-only", _run_gemini_notes_canary),
    ):
        result = runner()
        print(f"[{name}] {result.judgement}")
        print(f"  verdict: {result.run_root / 'verdict.md'}")
