"""Resolve diarisation clusters (``SPEAKER_NN``) to attendee names.

The evidence for naming a cluster is human-authored: the note's Attendees
frontmatter and free-text diarisation hints Michael writes in Meeting Prep
("Michael is the first voice you hear", "Nikki says 'Good morning
Michael'"). This module never guesses past that evidence:

- An explicit ``--assign`` is ground truth. Once a cluster is assigned it is
  never re-litigated by the LLM, in this run or a later one (assignments
  persist in the run cache).
- The LLM is only trusted at ``high`` confidence, or ``medium`` when a hint
  corroborates the name. Anything else — including a name outside the
  attendee list, which the LLM was told never to produce but might anyway —
  is treated as unresolved.
- An unresolved cluster becomes a :class:`~.models.SnippetRequest`: a couple
  of short audio snippets cut from the merged recording plus the
  already-resolved dialogue around them, for a human (via Discord, outside
  this module's scope) to identify. It is never silently force-assigned to
  a silent attendee or dropped. A text-sourced run (plan 005's Gemini/Teams
  adapter — no merged recording ever existed) gets the same request shape
  with ``clip_paths=[]`` and the representative quotes folded into
  ``context`` as text instead — never a cut against a file that isn't
  there.

:func:`resolve` is the pure per-run step: given a note, a raw transcript, an
agent, and the ground-truth assignments, it returns the transcript with
every resolvable cluster substituted and a list of requests for the rest.
:func:`run_speaker_resolution` is the orchestration the CLI command drives:
it loads/merges/stores the run cache state around one call to
:func:`resolve`, and — only once every cluster is resolved — stores
``resolved_transcript.json`` and appends **``--assign``-sourced** mappings
as Meeting Prep hint bullets (:func:`~.note.append_diarisation_hints`, the
one sanctioned write into that human-owned section). Per the interview
record's E15 decision, that write is scoped to identifications Michael
makes during the interactive snippet session — an LLM-only resolution,
however high its confidence, lands in ``resolved_transcript.json`` but
never in the note.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..claude import ClaudeAgent
from ..prompting import StructuredPrompt
from .audio import AudioTool
from .cache import RunCache
from .models import RawTranscript, SnippetRequest, SpeakerAssignment, Utterance
from .note import MEETING_PREP_HEADING, ParsedNote, append_diarisation_hints, parse_note

# The cache name `transcript asr`/`transcript adapt` store `RawTranscript`
# under — the same contract `cli/transcript.py` uses, duplicated here rather
# than imported: `transcription/` must not depend on `cli/`.
RAW_TRANSCRIPT_CACHE_NAME = "raw_transcript"
ASSIGNMENTS_CACHE_NAME = "assignments"
RESOLVED_TRANSCRIPT_CACHE_NAME = "resolved_transcript"

_UNKNOWN = "Unknown"
_SNIPPETS_PER_CLUSTER = 3
_MIN_QUOTE_SEPARATION_SECONDS = 30.0


class SpeakersError(RuntimeError):
    """Base for speaker-resolution domain errors."""


class MissingRawTranscriptError(SpeakersError):
    """Raised when no `raw_transcript.json` is cached for a run id."""

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"no cached raw transcript for run {run_id!r}; run `transcript "
            "merge-audio` + `transcript asr` (or `transcript adapt --run-id "
            f"{run_id}`) first."
        )
        self.run_id = run_id


class InvalidAssignmentError(SpeakersError):
    """Raised when an `--assign` value isn't `CLUSTER=NAME`."""

    def __init__(self, raw: str) -> None:
        super().__init__(
            f"invalid --assign value {raw!r}; expected CLUSTER=NAME, e.g. "
            '"SPEAKER_03=Nikki Staltari".'
        )
        self.raw = raw


# --- LLM-backed cluster -> name proposal -------------------------------------


class ClusterProposal(BaseModel):
    """One cluster's proposed identity, or an honest admission of doubt."""

    cluster: str
    name: str | None  # None = cannot resolve confidently
    confidence: Literal["high", "medium", "low"]
    reasoning: str


class SpeakerProposals(BaseModel):
    """The LLM's reply: one proposal per cluster it was asked about."""

    proposals: list[ClusterProposal]


class ClusterEvidence(BaseModel):
    """A compact digest of one unresolved cluster's evidence for the prompt."""

    cluster: str
    first_position_seconds: float
    utterance_count: int
    quotes: list[str]
    adjacent_speakers: list[str]


class SpeakerResolutionPrompt(StructuredPrompt[SpeakerProposals]):
    template = textwrap.dedent("""\
        You are identifying speakers in a diarised meeting transcript. Each
        diarisation "cluster" (e.g. SPEAKER_03) is one voice. For each
        cluster below, decide which attendee it is - or honestly admit you
        cannot tell.

        Attendees at this meeting: {{ attendees }}

        Meeting Prep section, verbatim. Michael's hints about who is who may
        appear anywhere in this text, in his own varied phrasing - read the
        whole thing, not just a bullet list:
        ---
        {{ meeting_prep }}
        ---

        The same hints, where they were found as bullets under a
        "Diarisation hints:" list, extracted for convenience only - the
        section text above is still the authoritative source, not this list:
        {{ diarisation_hints }}

        Evidence for each unresolved cluster (its first position in the
        meeting, a few representative quotes, and who talks immediately
        before/after it):
        {{ evidence | json }}

        Binding rules:
        - Your `name` for each cluster must be exactly one of the attendee
          names listed above, or the literal string "Unknown", or `null`.
          Never invent a name that isn't in the attendee list.
        - A hint in the Meeting Prep text outranks anything you infer from
          the conversation's content.
        - If the evidence is thin, ambiguous, or ties between two attendees,
          return `name: null` and `confidence: "low"`. Never guess: an
          unresolved cluster can always be asked about later, but a wrong
          name silently poisons the meeting record.
        - Respond with exactly one proposal per cluster listed in the
          evidence, in the same order.
        """)
    response_model = SpeakerProposals

    attendees: list[str]
    meeting_prep: str
    diarisation_hints: list[str]
    evidence: list[ClusterEvidence]


async def resolve(
    note: ParsedNote,
    transcript: RawTranscript,
    *,
    agent: ClaudeAgent,
    assignments: Sequence[SpeakerAssignment],
    audio_tool: AudioTool,
    merged_audio: Path,
    snippet_dir: Path,
) -> tuple[RawTranscript, list[SnippetRequest]]:
    """Resolve as many clusters in `transcript` to names as the evidence allows.

    1. `assignments` are applied first and are never re-litigated - even a
       contradictory LLM proposal for an assigned cluster is ignored.
    2. Every other cluster present in `transcript` is asked about in one LLM
       call; `high`-confidence proposals are accepted, as are `medium`
       ones corroborated by a diarisation hint mentioning the name.
    3. Every cluster that stays unresolved gets a `SnippetRequest`: when
       `merged_audio` is a real recording (see `_has_mergeable_audio`), 2-3
       representative utterances are cut from it into `snippet_dir`; for a
       text-sourced run (plan 005's adapter - there is no merged recording)
       the same representative quotes are folded into `context` as text
       instead, with `clip_paths=[]`. Either way the surrounding
       already-resolved dialogue is included as context. Never errors, never
       fabricates a clip path, and never skips a cluster.

    Only present clusters are ever considered - a silent attendee (listed
    in Attendees but never a speaker) is never force-assigned to anything.
    """
    has_audio = _has_mergeable_audio(transcript, merged_audio)
    assigned = {item.cluster: item.name for item in assignments}
    present_clusters = list(dict.fromkeys(u.speaker for u in transcript.utterances))
    unassigned_clusters = [c for c in present_clusters if c not in assigned]

    accepted: dict[str, str] = {}
    if unassigned_clusters:
        valid_names = set(note.attendees) | {_UNKNOWN}
        evidence = [
            _cluster_evidence(cluster, transcript.utterances)
            for cluster in unassigned_clusters
        ]
        proposals, _reply = await agent.run_structured(
            SpeakerResolutionPrompt(
                attendees=note.attendees,
                meeting_prep=_meeting_prep_body(note),
                diarisation_hints=note.diarisation_hints,
                evidence=evidence,
            )
        )
        by_cluster = {item.cluster: item for item in proposals.proposals}
        for cluster in unassigned_clusters:
            proposal = by_cluster.get(cluster)
            if proposal is None or proposal.name is None:
                continue
            if proposal.name not in valid_names:
                continue  # the LLM broke the contract; treat as unresolved
            # "low" confidence, or an uncorroborated "medium", is never accepted.
            accept = proposal.confidence == "high" or (
                proposal.confidence == "medium"
                and _corroborated_by_hint(proposal.name, note.diarisation_hints)
            )
            if accept:
                accepted[cluster] = proposal.name

    # Assignments always win, even over an accepted proposal - defence in
    # depth for "never re-litigated" beyond simply not asking about them.
    resolution: dict[str, str] = {**accepted, **assigned}

    resolved_utterances = [
        u.model_copy(update={"speaker": resolution[u.speaker]})
        if u.speaker in resolution
        else u
        for u in transcript.utterances
    ]
    resolved_transcript = transcript.model_copy(
        update={"utterances": resolved_utterances}
    )

    unresolved_clusters = [c for c in unassigned_clusters if c not in accepted]
    requests = [
        _snippet_request(
            cluster,
            transcript.utterances,
            resolved_utterances,
            audio_tool=audio_tool,
            merged_audio=merged_audio,
            snippet_dir=snippet_dir,
            has_audio=has_audio,
        )
        for cluster in unresolved_clusters
    ]
    return resolved_transcript, requests


def _has_mergeable_audio(transcript: RawTranscript, merged_audio: Path) -> bool:
    """Is there an actual merged recording to cut snippets from?

    A text-sourced run (`transcript adapt`, plan 005's Gemini/Teams ramp)
    never has one: `RawTranscript.audio_sha256` is `None`, and no
    `merged.m4a` was ever written to the run directory. Neither signal
    alone is trustworthy on its own - `audio_sha256 is None` doesn't prove
    the file is absent (paranoia, not a real path), and `merged_audio.
    exists()` alone could be fooled by a stale/relocated file - so this
    requires both: a `RawTranscript` that claims audio provenance *and* a
    file that is actually there right now. Anything else means "don't try
    to cut, and don't fabricate a path" (see `resolve`/`_snippet_request`).
    """
    return transcript.audio_sha256 is not None and merged_audio.exists()


def _corroborated_by_hint(name: str, hints: Sequence[str]) -> bool:
    """Does some hint mention `name` (a first name is enough, e.g. "Grace")?

    Hints are Michael's own shorthand ("Grace did most of the talking"), not
    full-name citations, so corroboration checks per-word overlap rather
    than requiring the whole name as a substring. Matching is word-bounded
    (`\\bword\\b`), not a bare substring check — "ada" must not corroborate
    from "Canada", nor "ed" from "needed".
    """
    name_words = {word.lower() for word in name.split() if word}
    return any(
        any(re.search(rf"\b{re.escape(word)}\b", hint.lower()) for word in name_words)
        for hint in hints
    )


def _meeting_prep_body(note: ParsedNote) -> str:
    for section in note.sections:
        if (
            section.heading is not None
            and section.heading.strip().lower() == MEETING_PREP_HEADING
        ):
            return section.body
    return ""


def _cluster_evidence(cluster: str, utterances: Sequence[Utterance]) -> ClusterEvidence:
    cluster_utterances = [u for u in utterances if u.speaker == cluster]
    picks = _select_representative_indices(cluster_utterances, _SNIPPETS_PER_CLUSTER)
    return ClusterEvidence(
        cluster=cluster,
        first_position_seconds=cluster_utterances[0].start,
        utterance_count=len(cluster_utterances),
        quotes=[cluster_utterances[i].text for i in picks],
        adjacent_speakers=_adjacent_speakers(cluster, utterances),
    )


def _adjacent_speakers(cluster: str, utterances: Sequence[Utterance]) -> list[str]:
    neighbours: list[str] = []
    for index, utterance in enumerate(utterances):
        if utterance.speaker != cluster:
            continue
        if index > 0 and utterances[index - 1].speaker != cluster:
            neighbours.append(utterances[index - 1].speaker)
        if index + 1 < len(utterances) and utterances[index + 1].speaker != cluster:
            neighbours.append(utterances[index + 1].speaker)
    return list(dict.fromkeys(neighbours))


def _select_representative_indices(
    utterances: Sequence[Utterance], count: int
) -> list[int]:
    """Indices of up to `count` representative picks: longest text, spread in time.

    A known heuristic (longest is not always clearest) - tune only with
    real-use feedback, per plan 006's maintenance notes.
    """
    if len(utterances) <= count:
        return list(range(len(utterances)))

    order = sorted(
        range(len(utterances)), key=lambda i: len(utterances[i].text), reverse=True
    )
    selected: list[int] = []
    for i in order:
        if len(selected) >= count:
            break
        if all(
            abs(utterances[i].start - utterances[j].start)
            >= _MIN_QUOTE_SEPARATION_SECONDS
            for j in selected
        ):
            selected.append(i)
    for i in order:
        if len(selected) >= count:
            break
        if i not in selected:
            selected.append(i)
    selected.sort(key=lambda i: utterances[i].start)
    return selected


def _snippet_request(
    cluster: str,
    original_utterances: Sequence[Utterance],
    resolved_utterances: Sequence[Utterance],
    *,
    audio_tool: AudioTool,
    merged_audio: Path,
    snippet_dir: Path,
    has_audio: bool,
) -> SnippetRequest:
    cluster_indices = [
        i for i, u in enumerate(original_utterances) if u.speaker == cluster
    ]
    cluster_utterances = [original_utterances[i] for i in cluster_indices]
    local_picks = _select_representative_indices(
        cluster_utterances, _SNIPPETS_PER_CLUSTER
    )
    picked_positions = [cluster_indices[i] for i in local_picks]

    clip_paths: list[str] = []
    quote_lines: list[str] = []
    if has_audio:
        snippet_dir.mkdir(parents=True, exist_ok=True)
        for n, position in enumerate(picked_positions, start=1):
            utterance = original_utterances[position]
            out = snippet_dir / f"{cluster}-{n}.m4a"
            audio_tool.cut(merged_audio, utterance.start, utterance.end, out)
            clip_paths.append(str(out))
    else:
        # No merged recording to cut from (a text-sourced run): fall back to
        # the same representative picks as text, so there's still something
        # for a human to identify the cluster from over Discord.
        quote_lines = [
            f'{cluster}: "{original_utterances[position].text}"'
            for position in picked_positions
        ]

    context_lines = quote_lines + _context_lines(
        cluster, picked_positions, resolved_utterances
    )
    return SnippetRequest(
        cluster=cluster, clip_paths=clip_paths, context="\n".join(context_lines)
    )


def _context_lines(
    cluster: str, positions: Sequence[int], resolved_utterances: Sequence[Utterance]
) -> list[str]:
    lines: list[str] = []
    seen: set[int] = set()
    for position in positions:
        for neighbour in (position - 1, position + 1):
            if (
                neighbour < 0
                or neighbour >= len(resolved_utterances)
                or neighbour in seen
            ):
                continue
            candidate = resolved_utterances[neighbour]
            if candidate.speaker == cluster:
                continue
            seen.add(neighbour)
            lines.append(f"{candidate.speaker}: {candidate.text}")
    return lines


# --- run orchestration: cache state around one `resolve` call ---------------


class AssignmentSet(BaseModel):
    """The accumulated `--assign` overrides for one run, as cached JSON."""

    assignments: list[SpeakerAssignment] = Field(default_factory=list)


class SpeakersResponse(BaseModel):
    """The JSON document `jake-tools transcript speakers` prints."""

    status: Literal["needs_input", "resolved"]
    run_id: str
    requests: list[SnippetRequest] = Field(default_factory=list)


def parse_assign(raw: str) -> SpeakerAssignment:
    """Parse one `--assign` value (`CLUSTER=NAME`) into a `SpeakerAssignment`."""
    cluster, separator, name = raw.partition("=")
    cluster = cluster.strip()
    name = name.strip()
    if not separator or not cluster or not name:
        raise InvalidAssignmentError(raw)
    return SpeakerAssignment(cluster=cluster, name=name)


def _merge_assignments(
    previous: Sequence[SpeakerAssignment], new: Sequence[SpeakerAssignment]
) -> list[SpeakerAssignment]:
    merged: dict[str, str] = {item.cluster: item.name for item in previous}
    for item in new:  # a later --assign for the same cluster wins
        merged[item.cluster] = item.name
    return [
        SpeakerAssignment(cluster=cluster, name=name)
        for cluster, name in merged.items()
    ]


def _finalise_unknown(
    transcript: RawTranscript, requests: Sequence[SnippetRequest]
) -> RawTranscript:
    unresolved = {request.cluster for request in requests}
    utterances = [
        u.model_copy(update={"speaker": _UNKNOWN}) if u.speaker in unresolved else u
        for u in transcript.utterances
    ]
    return transcript.model_copy(update={"utterances": utterances})


def _recording_phrase(note: ParsedNote) -> str:
    raw = note.frontmatter.get("Date")
    if raw is None:
        return "in this recording"
    text = str(raw).strip()
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2]
    return f"in the {text} recording" if text else "in this recording"


def _hint_lines_for_assignments(
    note: ParsedNote, assignments: Sequence[SpeakerAssignment]
) -> list[str]:
    """Hint bullets for HUMAN-confirmed mappings only.

    Per the interview record's E15 decision, the sanctioned Meeting Prep
    write is scoped to identifications Michael makes during the interactive
    snippet session (`--assign`) — never an LLM-only resolution, however
    high its confidence. `append_diarisation_hints` is idempotent, so it's
    safe to pass every non-"Unknown" assignment on record each call, not
    just ones new to this invocation.
    """
    phrase = _recording_phrase(note)
    return [
        f"{assignment.name} was {assignment.cluster} {phrase}"
        for assignment in sorted(assignments, key=lambda item: item.cluster)
        if assignment.name != _UNKNOWN
    ]


async def run_speaker_resolution(
    note_path: Path,
    run_id: str,
    *,
    assign: Sequence[str],
    finalise: bool,
    agent: ClaudeAgent,
    audio_tool: AudioTool,
    cache: RunCache,
) -> SpeakersResponse:
    """Drive one `jake-tools transcript speakers` invocation.

    Loads the cached raw transcript (`MissingRawTranscriptError` if there
    isn't one), merges `assign` into the run's persisted assignments
    (ground truth, durable across invocations), and calls `resolve`.

    If unresolved clusters remain and `finalise` is not set, returns a
    `needs_input` response with their `SnippetRequest`s and stores nothing
    further (the assignments merge above already persisted). If `finalise`
    is set, remaining clusters are mapped to "Unknown". Either way, once
    nothing is left unresolved: `resolved_transcript.json` is cached, and
    every non-"Unknown" `--assign`-sourced mapping is appended as a Meeting
    Prep hint bullet (LLM-only resolutions are cached but never written to
    the note — see the module docstring), before a `resolved` response is
    returned.
    """
    note = parse_note(note_path)
    transcript = cache.load(run_id, RAW_TRANSCRIPT_CACHE_NAME, RawTranscript)
    if transcript is None:
        raise MissingRawTranscriptError(run_id)

    new_assignments = [parse_assign(raw) for raw in assign]
    previous = cache.load(run_id, ASSIGNMENTS_CACHE_NAME, AssignmentSet)
    merged_assignments = _merge_assignments(
        previous.assignments if previous is not None else [], new_assignments
    )
    cache.store(
        run_id, ASSIGNMENTS_CACHE_NAME, AssignmentSet(assignments=merged_assignments)
    )

    run_dir = cache.run_dir(run_id)
    resolved, requests = await resolve(
        note,
        transcript,
        agent=agent,
        assignments=merged_assignments,
        audio_tool=audio_tool,
        merged_audio=run_dir / "merged.m4a",
        snippet_dir=run_dir / "snippets",
    )

    if requests and not finalise:
        return SpeakersResponse(status="needs_input", run_id=run_id, requests=requests)

    if requests:  # --finalise: whatever is left becomes "Unknown"
        resolved = _finalise_unknown(resolved, requests)

    cache.store(run_id, RESOLVED_TRANSCRIPT_CACHE_NAME, resolved)

    hint_lines = _hint_lines_for_assignments(note, merged_assignments)
    if hint_lines:
        append_diarisation_hints(Path(note.path), hint_lines)

    return SpeakersResponse(status="resolved", run_id=run_id, requests=[])
