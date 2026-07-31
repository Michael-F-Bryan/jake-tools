"""M9: the text transforms -- ``correct`` and ``polish``.

Both rewrite canonical turn *text* and nothing else. Timings, source
segment lineage, and above all ``turn_id`` are carried through unchanged
(M7's identity remap), which is what makes routine polish after a speaker
review never invalidate that review.

The two passes are genuinely different and stay distinguishable in
provenance even though one command may orchestrate both (corpus §7):

- **correct** fixes mis-transcriptions -- homophones and names, using the
  participant list and surrounding context as evidence. No meaning
  change; removals limited to ASR artefacts.
- **polish** improves readability -- filler, stutter repeats, false
  starts, punctuation. Never connectives or content the evidence does not
  support, and never a cross-speaker merge.

Every difference between input and output is accounted for by an M9
ledger entry, including the unchanged turns (``identity``): a ledger that
does not account for the full diff fails validation, so "the model
quietly rewrote turn 40" is not a thing that can survive this transform.
Whole-turn removals additionally record ``drop-empty`` with a reason from
M9's closed enum, which is the lineage the registry later uses to tell a
legitimately dropped turn from a dangling binding.

Structural safety rails, all enforced here rather than requested in the
prompt:

- turn count may shrink only by drops the ledger records;
- ``turn_id``, ``start_ms``, ``end_ms``, ``speaker_label``,
  ``source_segment_id`` and ``source_artefact_id`` are re-attached from
  the *input* turn, so a model that tried to change them simply cannot;
- both existing M9 validators run over the before/after pair:
  ``verify.py``'s structural gates (ordering, span coverage, speakers
  still represented) and ``stages.py``'s content-retention gate. Neither
  alone is enough -- a pass that replaced every turn with one character
  satisfies every structural check and is not a transcript.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, Field

from ...claude import Reply
from ...prompting import StructuredPrompt
from ..models import TranscriptArtifact, TranscriptTurn
from ..stages import (
    StagePrimitiveError,
    StructuredAgent,
    ensure_polish_preserves_content,
    run_structured_with_retries,
)
from ..verify import verify_turns
from .assignment import (
    SpeakerContext,
    SpeakerError,
    canonical_turn_set,
    canonical_turns,
    speaker_display_name,
)
from .components import (
    RemovalReason,
    TextEditEntry,
    TextEditLedgerComponent,
    TextEditLedgerComponentBody,
    TextEditMode,
    TextEditOperation,
    TimedTurn,
    TimedTurnSetComponent,
    TimedTurnSetComponentBody,
)
from .document import NoDocumentYet, TranscriptDocumentV1, project_head
from .ids import RunId
from .records import OperationRef, RevisionRecord
from .store import BundleStore

EDITOR_VERSION = "v1"


class TextTransformError(SpeakerError):
    """Base class for every error this module raises."""


class NoCanonicalTurnsError(TextTransformError):
    """There is no single canonical timed turn set to edit."""


class UnfaithfulEditError(TextTransformError):
    """The edited text did not survive the M9 gates.

    Refused rather than published: an over-aggressive polish that removes
    a fifth of the words is not "a bit terse", it is a transcript that no
    longer says what was said.
    """


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_hash(*, mode: TextEditMode, model: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {"editor_version": EDITOR_VERSION, "mode": mode.value, "model": model},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


class EditedTurn(BaseModel):
    """One turn as the model returns it: an ID and replacement text.

    Deliberately carries *only* those two fields. A payload that also
    accepted timings or a speaker would be a payload a model could get
    wrong; the transform re-attaches those from the input turn regardless,
    so not asking for them removes a whole class of failure instead of
    validating it away afterwards.
    """

    turn_id: str
    text: str = ""
    removal_reasons: list[RemovalReason] = Field(default_factory=list)


class TextEditPayload(BaseModel):
    turns: list[EditedTurn] = Field(default_factory=list)


class CorrectPrompt(StructuredPrompt[TextEditPayload]):
    response_model: ClassVar[type[BaseModel]] = TextEditPayload
    template: ClassVar[str] = """
Correct mis-transcriptions in these transcript turns. This is speech-to-text
repair, not editing: fix what the recogniser heard wrongly, and change nothing
else.

Correct:
- misheard names of people, places, products, and organisations, using the
  participant list and the surrounding conversation as evidence;
- homophones and near-homophones the context clearly disambiguates;
- obvious recogniser artefacts (mangled word boundaries, wrong casing of a
  known name).

Never:
- reword, summarise, tidy, or shorten anything;
- remove filler, stutters, or false starts -- that is a separate pass;
- add words the speaker did not say, including connectives;
- change meaning, hedging, or uncertainty.

Rules:
- Return JSON only: one entry per turn below, same turn_id, corrected text.
- Return every turn, including ones you did not change.
- Never return a turn_id that is not listed below.
- `removal_reasons` stays empty for this pass.

Meeting context:
{{ context }}

Participants:
{{ participants | json }}

Transcript turns:
{{ turns | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    context: str = ""
    participants: list[str]
    turns: list[dict[str, object]]
    correction: str = ""


class PolishPrompt(StructuredPrompt[TextEditPayload]):
    response_model: ClassVar[type[BaseModel]] = TextEditPayload
    template: ClassVar[str] = """
Polish these transcript turns for readability while preserving exactly what was
said. A reader should be able to follow the conversation without noticing the
recogniser; they should not encounter a single claim the speaker did not make.

Do:
- remove filler ("um", "uh", "you know" used as filler), stutter repeats, and
  abandoned false starts;
- add sentence punctuation and capitalisation;
- keep every substantive word, including hedges, corrections, and disagreement.

Never:
- summarise, compress, or "improve" phrasing;
- add connectives, transitions, or explanation the speaker did not say;
- merge turns, reorder them, or move text between them;
- drop a turn that still says something.

If a turn is nothing but filler and would be empty once cleaned, return it with
empty text and one or more `removal_reasons` from: filler, stutter-repeat,
false-start, non-lexical, duplicate.

Rules:
- Return JSON only: one entry per turn below, same turn_id.
- Return every turn, including ones you did not change.
- Never return a turn_id that is not listed below.

Meeting context:
{{ context }}

Transcript turns:
{{ turns | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    context: str = ""
    turns: list[dict[str, object]]
    correction: str = ""


@dataclass(frozen=True)
class TextTransformOutcome:
    revision: RevisionRecord
    turn_set: TimedTurnSetComponent
    ledger: TextEditLedgerComponent
    reply: Reply
    changed_turn_count: int
    dropped_turn_count: int


def _as_legacy_artifact(
    turns: tuple[TimedTurn, ...], context: SpeakerContext
) -> TranscriptArtifact:
    """Project canonical turns into ``models.TranscriptArtifact``.

    The M9 validators are ``verify.py``'s existing retention/fidelity
    gates, which speak that shape -- reusing them (rather than
    reimplementing the same five checks against ``TimedTurn``) is what the
    contract asks for, and means a gate fixed in one place stays fixed for
    both the legacy recipes and this engine. Milliseconds become seconds
    because that is the unit those gates compare in.
    """
    return TranscriptArtifact(
        turns=[
            TranscriptTurn(
                start=turn.start_ms / 1000.0,
                end=turn.end_ms / 1000.0,
                # The raw machine label, not the resolved display name.
                # These gates ask structural questions ("are two adjacent
                # turns the same speaker saying the same thing?"), and
                # once a review resolves several voices to "Unclear
                # speaker" the display name stops distinguishing them --
                # two different people overlapping on "Yeah" would read as
                # one speaker duplicating themselves. The cluster label
                # keeps voices distinct, which is what the gate assumes.
                speaker=turn.speaker_label,
                text=turn.text,
            )
            for turn in turns
        ]
    )


def _turn_payload(
    turns: tuple[TimedTurn, ...], context: SpeakerContext
) -> list[dict[str, object]]:
    return [
        {
            "turn_id": turn.turn_id,
            "speaker": speaker_display_name(context.assignment_for(turn), context),
            "start_ms": turn.start_ms,
            "text": turn.text,
        }
        for turn in turns
    ]


def _build_edited_turns(
    inputs: tuple[TimedTurn, ...], payload: TextEditPayload, *, mode: TextEditMode
) -> tuple[tuple[TimedTurn, ...], tuple[TextEditEntry, ...]]:
    """Re-attach every structural field from the input turn and account for
    the full diff (M9).

    A returned turn_id that is not an input turn is a hard error: a stage
    that invents editorial nodes is not editing this transcript. A missing
    turn_id is treated as "unchanged" rather than "dropped" -- dropping is
    an explicit act that must carry a removal reason, and silence is not
    consent to delete evidence.
    """
    by_id = {edited.turn_id: edited for edited in payload.turns}
    unknown = sorted(set(by_id) - {turn.turn_id for turn in inputs})
    if unknown:
        raise UnfaithfulEditError(
            f"the edit stage returned turn ID(s) that are not in the input turn "
            f"set: {unknown}."
        )

    kept: list[TimedTurn] = []
    entries: list[TextEditEntry] = []
    for turn in inputs:
        edited = by_id.get(turn.turn_id)
        old_hash = _sha256_text(turn.text)
        if edited is None:
            kept.append(turn)
            entries.append(
                TextEditEntry(
                    operation=TextEditOperation.IDENTITY,
                    input_turn_ids=(turn.turn_id,),
                    output_turn_ids=(turn.turn_id,),
                    old_text_sha256=old_hash,
                    new_text_sha256=old_hash,
                )
            )
            continue
        new_text = edited.text.strip()
        if not new_text:
            if mode == TextEditMode.CORRECT:
                raise UnfaithfulEditError(
                    f"the correct pass emptied turn {turn.turn_id}; correction fixes "
                    "mis-transcriptions and never removes a turn (M9)."
                )
            reasons = tuple(edited.removal_reasons) or (RemovalReason.NON_LEXICAL,)
            entries.append(
                TextEditEntry(
                    operation=TextEditOperation.DROP_EMPTY,
                    input_turn_ids=(turn.turn_id,),
                    old_text_sha256=old_hash,
                    removal_reasons=reasons,
                )
            )
            continue
        kept.append(turn.model_copy(update={"text": new_text}))
        new_hash = _sha256_text(new_text)
        entries.append(
            TextEditEntry(
                operation=(
                    TextEditOperation.IDENTITY
                    if new_hash == old_hash
                    else TextEditOperation.TEXT_EDIT
                ),
                input_turn_ids=(turn.turn_id,),
                output_turn_ids=(turn.turn_id,),
                old_text_sha256=old_hash,
                new_text_sha256=new_hash,
            )
        )
    if not kept:
        raise UnfaithfulEditError(
            "the edit stage emptied every turn; a transcript with no turns is not "
            "a polished transcript."
        )
    return tuple(kept), tuple(entries)


def _gate_or_refuse(
    before: TranscriptArtifact, after: TranscriptArtifact, *, mode: TextEditMode
) -> None:
    """Run both M9 validators over the before/after pair.

    ``verify_turns`` checks structure (ordering, span coverage, speakers
    still represented, no adjacent duplicates); ``ensure_polish_preserves_content``
    checks that the words survived. Both are needed: a pass that replaced
    every turn with a single character satisfies every structural check
    and is still not a transcript.
    """
    report = verify_turns(before, after, affected_paths=[])
    if report.failed_gate_ids:
        raise UnfaithfulEditError(
            f"the {mode.value} pass failed M9's verification gates: "
            + ", ".join(report.failed_gate_ids)
        )
    try:
        ensure_polish_preserves_content(before.turns, after.turns)
    except StagePrimitiveError as exc:
        raise UnfaithfulEditError(
            f"the {mode.value} pass failed M9's content-retention gate: {exc}"
        ) from exc


async def transform_text(
    store: BundleStore,
    *,
    run_id: RunId,
    mode: TextEditMode,
    agent: StructuredAgent,
    model: str,
    context_note: str = "",
    max_attempts: int = 3,
) -> TextTransformOutcome:
    """M9: run one text pass over the head's canonical turns.

    Appends a revision carrying the rewritten turn set plus its ledger,
    superseding the turn set it edited (M21) so ``transcript.timed``
    stays a one-cardinality key. Turn IDs survive, so any applied speaker
    review, chapter set, or minutes evidence ref keeps binding -- with the
    single exception of a turn this pass dropped, whose bindings M7
    discharges and whose disappearance the ledger accounts for.
    """
    document = _require_document(store)
    turn_set = canonical_turn_set(document.components)
    if turn_set is None:
        raise NoCanonicalTurnsError(
            "this document has no single canonical timed turn set to edit (run "
            "`transform normalise` first)."
        )
    inputs = turn_set.turns
    speaker_context = SpeakerContext.from_components(document.components)

    prompt: StructuredPrompt[TextEditPayload]
    if mode == TextEditMode.CORRECT:
        prompt = CorrectPrompt(
            context=context_note,
            participants=[
                participant.display_names[0]
                for participant in speaker_context.participants.values()
            ],
            turns=_turn_payload(inputs, speaker_context),
        )
    else:
        prompt = PolishPrompt(
            context=context_note, turns=_turn_payload(inputs, speaker_context)
        )

    payload, reply = await run_structured_with_retries(
        agent, prompt, max_attempts=max_attempts
    )
    kept, entries = _build_edited_turns(inputs, payload, mode=mode)
    _gate_or_refuse(
        _as_legacy_artifact(inputs, speaker_context),
        _as_legacy_artifact(kept, speaker_context),
        mode=mode,
    )

    turn_set_body = TimedTurnSetComponentBody(
        source_artefact_ids=turn_set.source_artefact_ids,
        coordinate_domain=turn_set.coordinate_domain,
        turns=kept,
    )
    new_turn_set = store.add_component(turn_set_body)
    assert isinstance(new_turn_set, TimedTurnSetComponent)
    ledger_body = TextEditLedgerComponentBody(
        mode=mode,
        input_turn_set_component_id=turn_set.component_id,
        output_turn_set_component_id=new_turn_set.component_id,
        editor=f"claude-agent:{model}",
        config_hash=_config_hash(mode=mode, model=model),
        entries=entries,
    )
    ledger = store.add_component(ledger_body)
    assert isinstance(ledger, TextEditLedgerComponent)

    # Components are content-identified (M1), so a pass that changed no
    # text gets the *same* component back. Carrying and superseding one ID
    # in a single revision is an invalid supersession claim (M21) -- and
    # semantically wrong anyway: nothing was replaced. The ledger is still
    # appended, because "this pass ran and changed nothing" is exactly the
    # proof `text.corrected` should carry.
    unchanged = new_turn_set.component_id == turn_set.component_id
    revision = store.append_revision(
        operation=OperationRef(
            kind=f"text-{mode.value}",
            input_ids=(turn_set.component_id,),
            config_hash=ledger_body.config_hash,
            rationale=f"M9 {mode.value} pass over canonical turns",
        ),
        parent_revision_ids=(document.revision_id,),
        component_ids=(
            (ledger.component_id,)
            if unchanged
            else (new_turn_set.component_id, ledger.component_id)
        ),
        superseded_component_ids=() if unchanged else (turn_set.component_id,),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)
    return TextTransformOutcome(
        revision=revision,
        turn_set=new_turn_set,
        ledger=ledger,
        reply=reply,
        changed_turn_count=sum(
            1 for entry in entries if entry.operation == TextEditOperation.TEXT_EDIT
        ),
        dropped_turn_count=sum(
            1 for entry in entries if entry.operation == TextEditOperation.DROP_EMPTY
        ),
    )


def _require_document(store: BundleStore) -> TranscriptDocumentV1:
    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise TextTransformError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )
    return document


def turn_texts(document: TranscriptDocumentV1) -> tuple[str, ...]:
    """Diagnostic helper: the head's canonical turn texts, in order."""
    return tuple(turn.text for turn in canonical_turns(document.components))
