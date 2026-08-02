from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_transcript_bundle_render_apply import _reviewed_bundle

from jake_tools.transcripts.bundle.assignment import (
    SpeakerContext,
    canonical_turn_set,
)
from jake_tools.transcripts.bundle.components import (
    ChapterSetComponent,
    MinutesComponent,
)
from jake_tools.transcripts.bundle.corrections import (
    Correction,
    apply_correction_pack,
    export_correction_pack,
)
from jake_tools.transcripts.bundle.document import NoDocumentYet, project_head
from jake_tools.transcripts.bundle.utterances import (
    UtterancePlanError,
    apply_utterance_plan,
    export_utterance_plan,
)


def test_guarded_correction_pack_changes_one_turn_and_invalidates_products(
    tmp_path: Path,
) -> None:
    store = _reviewed_bundle(tmp_path)
    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)
    turns = canonical_turn_set(document.components)
    assert turns is not None
    original = turns.turns[0]
    path = export_correction_pack(store, destination=tmp_path / "corrections.json")
    pack = json.loads(path.read_text(encoding="utf-8"))
    pack["reviewer"] = "Fixture Reviewer"
    pack["corrections"] = [
        {
            "turn_id": original.turn_id,
            "old_text": original.text,
            "new_text": f"{original.text} [audio verified]",
            "evidence_ref": "audio:0-2s",
            "rationale": "verified eligibility figure against audio",
        }
    ]
    path.write_text(json.dumps(pack), encoding="utf-8")

    outcome = apply_correction_pack(store, pack_path=path)

    assert outcome.changed_turn_count == 1
    assert outcome.turn_set.turns[0].text.endswith("[audio verified]")
    head = project_head(store)
    assert not isinstance(head, NoDocumentYet)
    assert not head.components_of(ChapterSetComponent)
    assert not head.components_of(MinutesComponent)


def test_correction_pack_refuses_a_blank_replacement() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        Correction.model_validate(
            {
                "turn_id": "turn_019fb000-0000-7000-8000-000000000001",
                "old_text": "ten kilograms",
                "new_text": "   ",
                "evidence_ref": "audio:10-12s",
                "rationale": "verified against audio",
            }
        )


def test_utterance_plan_requires_exact_partition_and_refuses_cross_named_merge(
    tmp_path: Path,
) -> None:
    store = _reviewed_bundle(tmp_path)
    document = project_head(store)
    assert not isinstance(document, NoDocumentYet)
    turn_set = canonical_turn_set(document.components)
    assert turn_set is not None
    context = SpeakerContext.from_components(document.components)
    pair_index = next(
        index
        for index in range(len(turn_set.turns) - 1)
        if {
            context.assignment_for(turn).participant_id
            for turn in turn_set.turns[index : index + 2]
            if context.assignment_for(turn).participant_id is not None
        }
        and len(
            {
                context.assignment_for(turn).participant_id
                for turn in turn_set.turns[index : index + 2]
                if context.assignment_for(turn).participant_id is not None
            }
        )
        > 1
    )
    path = export_utterance_plan(store, destination=tmp_path / "utterances.json")
    plan = json.loads(path.read_text(encoding="utf-8"))
    first, second = plan["groups"][pair_index : pair_index + 2]
    plan["reviewer"] = "Fixture Reviewer"
    plan["groups"][pair_index : pair_index + 2] = [
        {
            "input_turn_ids": first["input_turn_ids"] + second["input_turn_ids"],
            "text": f"{first['text']} {second['text']}",
            "evidence_ref": "audio:overlap",
            "rationale": "candidate merge",
        }
    ]
    path.write_text(json.dumps(plan), encoding="utf-8")
    original_head = store.load_manifest().head_revision_id

    with pytest.raises(UtterancePlanError, match="cross-named-speaker"):
        apply_utterance_plan(store, plan_path=path)

    assert store.load_manifest().head_revision_id == original_head
