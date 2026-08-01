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
  the *input* turn, so a model that tried to change them simply cannot --
  the model is never even shown a ``turn_id``, and addresses turns by
  their position in the window instead;
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

#: How many turns one edit call sees. A pass asks the model to return
#: *every* turn it was given, so the request and the response both scale
#: with the transcript: a 51-minute meeting is 753 turns, and asking for
#: all of them in one call spends most of an hour before failing on the
#: output budget. Windowing is what makes the pass proportional to meeting
#: length instead of capped by it.
#:
#: Windows are cut on turn boundaries and each is edited independently,
#: which is sound because M9 already forbids the only operations that
#: would need cross-window context -- no merging turns, no moving text
#: between them, no reordering. A turn is only ever rewritten in place.
TEXT_WINDOW_TURNS = 60


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


def _config_hash(*, mode: TextEditMode, model: str, window_turns: int) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "editor_version": EDITOR_VERSION,
                "mode": mode.value,
                "model": model,
                # Window size changes what each call sees, so two passes
                # run at different sizes are not the same configuration
                # even when the model and mode match.
                "window_turns": window_turns,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


class EditedTurn(BaseModel):
    """One turn as the model returns it: a position and replacement text.

    Deliberately carries *only* those two fields. A payload that also
    accepted timings or a speaker would be a payload a model could get
    wrong; the transform re-attaches those from the input turn regardless,
    so not asking for them removes a whole class of failure instead of
    validating it away afterwards.

    ``index`` is the turn's 1-based position **within this window**, not
    its ``turn_id``. Asking a model to reproduce hundreds of uuid7s does
    not work -- chaptering learned this first (it collapsed to a single
    chapter over 413 turns), and a real 753-turn meeting then failed a
    text pass on a returned ``turn_..._placeholder``. A small integer is
    something a model can carry accurately, and the transform maps it back
    to the real turn, so identity never depends on the model at all.
    """

    index: int
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
- Return JSON only: one entry per turn below, same `index`, corrected text.
- Return every turn, including ones you did not change.
- Never return an `index` that is not listed below.
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
- Return JSON only: one entry per turn below, same `index`.
- Return every turn, including ones you did not change.
- Never return an `index` that is not listed below.

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
    #: One reply per window, in window order -- a pass is several agent
    #: calls, and collapsing them to one would misreport the usage.
    replies: tuple[Reply, ...]
    changed_turn_count: int
    dropped_turn_count: int
    window_count: int


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
    """The window as the model sees it: positions, not identities.

    ``turn_id`` is deliberately absent. The model has no use for it -- it
    cannot be asked to return one reliably (see :class:`EditedTurn`) -- and
    leaving it out costs it nothing while removing the temptation to echo
    a mangled one back.
    """
    return [
        {
            "index": position,
            "speaker": speaker_display_name(context.assignment_for(turn), context),
            "start_ms": turn.start_ms,
            "text": turn.text,
        }
        for position, turn in enumerate(turns, start=1)
    ]


def _build_edited_turns(
    inputs: tuple[TimedTurn, ...], payload: TextEditPayload, *, mode: TextEditMode
) -> tuple[tuple[TimedTurn, ...], tuple[TextEditEntry, ...]]:
    """Re-attach every structural field from the input turn and account for
    the full diff (M9).

    A returned index outside the window is a hard error: a stage that
    edits turns it was not shown is not editing this transcript. A missing
    index is treated as "unchanged" rather than "dropped" -- dropping is
    an explicit act that must carry a removal reason, and silence is not
    consent to delete evidence.
    """
    by_index = {edited.index: edited for edited in payload.turns}
    unknown = sorted(set(by_index) - set(range(1, len(inputs) + 1)))
    if unknown:
        raise UnfaithfulEditError(
            f"the edit stage returned index/indices outside the {len(inputs)} turns "
            f"it was given: {unknown}."
        )

    kept: list[TimedTurn] = []
    entries: list[TextEditEntry] = []
    for position, turn in enumerate(inputs, start=1):
        edited = by_index.get(position)
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


def _windows(count: int, size: int) -> tuple[tuple[int, int], ...]:
    """Split ``count`` turns into consecutive half-open index ranges."""
    if size < 1:
        raise TextTransformError(f"window size must be at least one turn, got {size}.")
    return tuple(
        (start, min(start + size, count)) for start in range(0, max(count, 1), size)
    )


#: Two gates ask transcript-level questions that are meaningless on an
#: arbitrary text window:
#:
#: - ``turns.coverage-preserved`` compares the first start and last end. A
#:   filler turn at a window edge is normally interior to the transcript.
#: - ``turns.speakers-preserved`` compares speaker sets. A window can contain
#:   one filler-only turn for a speaker -- especially ``Unclear speaker`` --
#:   while that speaker remains represented elsewhere in the transcript.
#:
#: The assembled gate runs over the whole before/after pair with no
#: exemptions, so neither invariant is weakened: truncating the transcript or
#: removing a speaker globally still fails. Windowing only stops an arbitrary
#: cut from creating a false failure.
_WINDOW_EXEMPT_GATE_IDS = frozenset(
    {"turns.coverage-preserved", "turns.speakers-preserved"}
)


def _gate_or_refuse(
    before: TranscriptArtifact,
    after: TranscriptArtifact,
    *,
    mode: TextEditMode,
    where: str = "",
    exempt_gate_ids: frozenset[str] = frozenset(),
) -> None:
    """Run both M9 validators over the before/after pair.

    ``verify_turns`` checks structure (ordering, span coverage, speakers
    still represented, no adjacent duplicates); ``ensure_polish_preserves_content``
    checks that the words survived. Both are needed: a pass that replaced
    every turn with a single character satisfies every structural check
    and is still not a transcript.
    """
    suffix = f" ({where})" if where else ""
    report = verify_turns(before, after, affected_paths=[])
    failed = [gate for gate in report.failed_gate_ids if gate not in exempt_gate_ids]
    if failed:
        raise UnfaithfulEditError(
            f"the {mode.value} pass failed M9's verification gates{suffix}: "
            + ", ".join(failed)
        )
    try:
        ensure_polish_preserves_content(before.turns, after.turns)
    except StagePrimitiveError as exc:
        raise UnfaithfulEditError(
            f"the {mode.value} pass failed M9's content-retention gate{suffix}: {exc}"
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
    window_turns: int = TEXT_WINDOW_TURNS,
) -> TextTransformOutcome:
    """M9: run one text pass over the head's canonical turns.

    Appends a revision carrying the rewritten turn set plus its ledger,
    superseding the turn set it edited (M21) so ``transcript.timed``
    stays a one-cardinality key. Turn IDs survive, so any applied speaker
    review, chapter set, or minutes evidence ref keeps binding -- with the
    single exception of a turn this pass dropped, whose bindings M7
    discharges and whose disappearance the ledger accounts for.

    The pass runs in windows of ``window_turns`` (see
    :data:`TEXT_WINDOW_TURNS`). Each window is gated on its own so a
    failure names the window that caused it, and the assembled result is
    gated again as a whole -- windowing is a way to fit the work through
    the model, not a reason to check less of it.
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
    participants = [
        participant.display_names[0]
        for participant in speaker_context.participants.values()
    ]

    ranges = _windows(len(inputs), window_turns)
    kept: tuple[TimedTurn, ...] = ()
    entries: tuple[TextEditEntry, ...] = ()
    replies: list[Reply] = []
    for number, (start, end) in enumerate(ranges, start=1):
        window = inputs[start:end]
        where = f"window {number} of {len(ranges)}, turns {start + 1}-{end}"
        prompt: StructuredPrompt[TextEditPayload]
        if mode == TextEditMode.CORRECT:
            prompt = CorrectPrompt(
                context=context_note,
                participants=participants,
                turns=_turn_payload(window, speaker_context),
            )
        else:
            prompt = PolishPrompt(
                context=context_note, turns=_turn_payload(window, speaker_context)
            )
        payload, reply = await run_structured_with_retries(
            agent, prompt, max_attempts=max_attempts
        )
        replies.append(reply)
        window_kept, window_entries = _build_edited_turns(window, payload, mode=mode)
        _gate_or_refuse(
            _as_legacy_artifact(window, speaker_context),
            _as_legacy_artifact(window_kept, speaker_context),
            mode=mode,
            where=where,
            exempt_gate_ids=_WINDOW_EXEMPT_GATE_IDS,
        )
        kept += window_kept
        entries += window_entries

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
        config_hash=_config_hash(mode=mode, model=model, window_turns=window_turns),
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
        replies=tuple(replies),
        window_count=len(ranges),
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
