"""M8 rung 7: the speaker *proposal* stage.

:func:`propose_speakers` runs a bounded typed stage through the
``ClaudeAgent`` seam (M15 -- the only route model judgement ever takes) to
suggest, per voice cluster, which declared participant the
*conversational* evidence points at. That is the corpus's own point:
"Yes, I am" is answerable from who asked the question, not from
voice-cluster order.

It is structurally incapable of assigning anyone. Its output is a
:class:`~.components.SpeakerHypothesisSetComponent`, which only rung 7 of
the ladder reads, and rung 7 never satisfies the meeting-note speaker
gate (M5) -- so a proposal can inform a reviewer and can never become a
published attribution on its own. Every proposal is validated against the
document's real cluster, participant, and turn IDs before it is stored; a
proposal naming anything else is rejected outright, never quietly dropped
while its siblings are kept.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, Field

from ...claude import Reply
from ...prompting import StructuredPrompt
from ..stages import StructuredAgent, run_structured_with_retries
from .assignment import SpeakerContext, SpeakerError, canonical_turns
from .components import (
    MachineAttributionSetComponent,
    SpeakerHypothesis,
    SpeakerHypothesisSetComponent,
    SpeakerHypothesisSetComponentBody,
)
from .document import TranscriptDocumentV1
from .ids import ClusterId, RunId
from .records import OperationRef, RevisionRecord
from .store import BundleStore

PROPOSER_VERSION = "v1"


class NoClustersToProposeError(SpeakerError):
    """There are no machine clusters to propose participants for."""


class InvalidProposalError(SpeakerError):
    """A proposal named a cluster, participant, or turn that does not
    exist in this document. Rejected rather than dropped: a stage that
    invents IDs is not producing evidence, and silently keeping its other
    proposals would hide that."""


class SpeakerProposal(BaseModel):
    """One cluster's proposed participant, as the model returns it."""

    cluster_id: str
    participant_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_turn_ids: list[str] = Field(default_factory=list)
    rationale: str


class SpeakerProposalPayload(BaseModel):
    proposals: list[SpeakerProposal] = Field(default_factory=list)


class SpeakerProposalPrompt(StructuredPrompt[SpeakerProposalPayload]):
    response_model: ClassVar[type[BaseModel]] = SpeakerProposalPayload
    template: ClassVar[str] = """
You are proposing -- not deciding -- which declared participant each machine
voice cluster belongs to. A human reviews every proposal before anything is
published, so an honest "I cannot tell" is a good answer and a confident wrong
answer is the worst possible one.

Reason from what is said, not from the order clusters appear: who asks the
questions, who answers them, who is addressed by name, who describes their own
circumstances. A cluster is a voice, not a person.

Rules:
- Return JSON only, one proposal per cluster listed below, and no others.
- `participant_id` must be one of the declared participant IDs, or null.
- Use null whenever the conversation does not actually identify the speaker.
- `confidence` is 0.0-1.0 and must reflect genuine evidential support.
- `evidence_turn_ids` must cite turn IDs from the transcript below that
  justify the proposal. A proposal with no cited turns must use null.
- `rationale` is one sentence naming the evidence, not restating the label.

Declared participants:
{{ participants | json }}

Voice clusters:
{{ clusters | json }}

Transcript turns (turn_id, cluster, time, text):
{{ turns | json }}
{% if correction %}

Your previous response was invalid: {{ correction }}
Return corrected JSON that fixes this issue.
{% endif %}
"""

    participants: list[dict[str, object]]
    clusters: list[dict[str, object]]
    turns: list[dict[str, object]]
    correction: str = ""


def _proposal_prompt_payload(
    document: TranscriptDocumentV1,
    attribution: MachineAttributionSetComponent,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """The three JSON blocks the proposal prompt renders, built once."""
    context = SpeakerContext.from_components(document.components)
    participants: list[dict[str, object]] = [
        {
            "participant_id": participant.participant_id,
            "display_name": participant.display_names[0],
            "status": participant.status.value,
        }
        for participant in sorted(
            context.participants.values(), key=lambda p: p.display_names[0]
        )
    ]
    turn_counts: dict[ClusterId, int] = {}
    for assignment in attribution.assignments:
        turn_counts[assignment.cluster_id] = (
            turn_counts.get(assignment.cluster_id, 0) + 1
        )
    clusters: list[dict[str, object]] = [
        {
            "cluster_id": cluster.cluster_id,
            "raw_label": cluster.raw_label,
            "turn_count": turn_counts.get(cluster.cluster_id, 0),
            "total_ms": cluster.total_ms,
        }
        for cluster in attribution.clusters
    ]
    turns: list[dict[str, object]] = [
        {
            "turn_id": turn.turn_id,
            "cluster_id": context.cluster_by_turn.get(turn.turn_id),
            "start_ms": turn.start_ms,
            "end_ms": turn.end_ms,
            "text": turn.text,
        }
        for turn in canonical_turns(document.components)
    ]
    return participants, clusters, turns


def _validate_proposals(
    payload: SpeakerProposalPayload,
    *,
    known_cluster_ids: frozenset[str],
    known_participant_ids: frozenset[str],
    known_turn_ids: frozenset[str],
) -> tuple[SpeakerHypothesis, ...]:
    """Reject anything the model invented, and drop nothing silently."""
    seen: set[str] = set()
    hypotheses: list[SpeakerHypothesis] = []
    for proposal in payload.proposals:
        if proposal.cluster_id not in known_cluster_ids:
            raise InvalidProposalError(
                f"proposal names cluster {proposal.cluster_id!r}, which is not a "
                f"cluster in this document."
            )
        if proposal.cluster_id in seen:
            raise InvalidProposalError(
                f"more than one proposal for cluster {proposal.cluster_id!r}."
            )
        seen.add(proposal.cluster_id)
        if (
            proposal.participant_id is not None
            and proposal.participant_id not in known_participant_ids
        ):
            raise InvalidProposalError(
                f"proposal names participant {proposal.participant_id!r}, which is "
                "not a declared participant of this document (F21: owners and "
                "speakers come only from participant records)."
            )
        unknown_turns = sorted(set(proposal.evidence_turn_ids) - known_turn_ids)
        if unknown_turns:
            raise InvalidProposalError(
                f"proposal for cluster {proposal.cluster_id!r} cites turn ID(s) that "
                f"are not in this document: {unknown_turns}."
            )
        participant_id = proposal.participant_id
        if participant_id is not None and not proposal.evidence_turn_ids:
            raise InvalidProposalError(
                f"proposal for cluster {proposal.cluster_id!r} names a participant "
                "but cites no evidence turns; an uncited proposal must be null."
            )
        hypotheses.append(
            SpeakerHypothesis(
                cluster_id=proposal.cluster_id,
                participant_id=participant_id,
                confidence=proposal.confidence,
                evidence_turn_ids=tuple(proposal.evidence_turn_ids),
                rationale=proposal.rationale.strip() or "no rationale supplied",
            )
        )
    missing = sorted(known_cluster_ids - seen)
    for cluster_id in missing:
        hypotheses.append(
            SpeakerHypothesis(
                cluster_id=cluster_id,
                participant_id=None,
                confidence=0.0,
                rationale=(
                    "no proposal returned for this cluster; recorded as no candidate "
                    "rather than omitted, so review still sees it"
                ),
            )
        )
    return tuple(sorted(hypotheses, key=lambda h: h.cluster_id))


@dataclass(frozen=True)
class ProposeOutcome:
    revision: RevisionRecord
    hypotheses: SpeakerHypothesisSetComponent
    reply: Reply | None
    named_cluster_count: int


def _config_hash(model: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {"proposer_version": PROPOSER_VERSION, "model": model}, sort_keys=True
        ).encode()
    ).hexdigest()


async def propose_speakers(
    store: BundleStore,
    *,
    run_id: RunId,
    agent: StructuredAgent,
    model: str,
    max_attempts: int = 3,
) -> ProposeOutcome:
    """M8: propose one participant candidate per voice cluster (rung 7).

    ``agent`` is the ``StructuredAgent`` protocol from ``stages.py`` -- the
    established ``ClaudeAgent`` seam, and the only route model judgement
    ever takes (M15). Every proposal is validated against the document's own cluster,
    participant, and turn IDs; a cluster the model skipped is recorded as
    "no candidate" rather than omitted, so review always sees every
    cluster it must decide.
    """
    document = _require_document(store)
    attributions = document.components_of(MachineAttributionSetComponent)
    if len(attributions) != 1:
        raise NoClustersToProposeError(
            f"{len(attributions)} machine-attribution components in this document; "
            "speaker proposal needs exactly one (run `transform normalise` first)."
        )
    attribution = attributions[0]

    participants, clusters, turns = _proposal_prompt_payload(document, attribution)
    payload, reply = await run_structured_with_retries(
        agent,
        SpeakerProposalPrompt(
            participants=participants, clusters=clusters, turns=turns
        ),
        max_attempts=max_attempts,
    )
    hypotheses = _validate_proposals(
        payload,
        known_cluster_ids=frozenset(
            cluster.cluster_id for cluster in attribution.clusters
        ),
        known_participant_ids=frozenset(
            SpeakerContext.from_components(document.components).participants
        ),
        known_turn_ids=frozenset(
            turn.turn_id for turn in canonical_turns(document.components)
        ),
    )

    body = SpeakerHypothesisSetComponentBody(
        proposer=f"claude-agent:{model}",
        config_hash=_config_hash(model),
        hypotheses=hypotheses,
    )
    component = store.add_component(body)
    assert isinstance(component, SpeakerHypothesisSetComponent)
    revision = store.append_revision(
        operation=OperationRef(
            kind="speakers-propose",
            input_ids=(attribution.component_id,),
            config_hash=body.config_hash,
            rationale=(
                "machine speaker hypotheses per voice cluster (M8 rung 7); "
                "proposals never assign -- review decides"
            ),
        ),
        parent_revision_ids=(document.revision_id,),
        component_ids=(component.component_id,),
        superseded_component_ids=tuple(
            existing.component_id
            for existing in document.components_of(SpeakerHypothesisSetComponent)
        ),
    )
    store.update_head(run_id=run_id, revision_id=revision.revision_id)
    return ProposeOutcome(
        revision=revision,
        hypotheses=component,
        reply=reply,
        named_cluster_count=sum(
            1 for hypothesis in hypotheses if hypothesis.participant_id is not None
        ),
    )


def _require_document(store: BundleStore) -> TranscriptDocumentV1:
    from .document import NoDocumentYet, project_head

    document = project_head(store)
    if isinstance(document, NoDocumentYet):
        raise SpeakerError(
            "this bundle has no assembled document yet (M18: run `transform "
            "assemble` first)."
        )
    return document
