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
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..ai_usage import AITelemetry
from ..cache_models import CacheEnvelope
from ..claude import ClaudeAgent
from ..prompting import StructuredPrompt
from .audio import AudioTool
from .cache import RunCache
from .models import (
    RawTranscript,
    SnippetRequest,
    SpeakerAssignment,
    SpeakerCorrection,
    StageTiming,
    TimestampedSegment,
    Utterance,
)
from .note import MEETING_PREP_HEADING, ParsedNote, append_diarisation_hints, parse_note

# The cache name `transcript asr`/`transcript adapt` store `RawTranscript`
# under — the same contract `cli/transcript.py` uses, duplicated here rather
# than imported: `transcription/` must not depend on `cli/`.
RAW_TRANSCRIPT_CACHE_NAME = "raw_transcript"
ASSIGNMENTS_CACHE_NAME = "assignments"
CORRECTIONS_CACHE_NAME = "corrections"
RESOLVED_TRANSCRIPT_CACHE_NAME = "resolved_transcript"

_UNKNOWN = "Unknown"
_SNIPPETS_PER_CLUSTER = 3
_MIN_QUOTE_SEPARATION_SECONDS = 30.0
_HINT_QUOTE_MAX_CHARS = 80


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


class InvalidCorrectionError(SpeakersError):
    """Raised when a range correction is malformed or unsafe."""


class SpeakerCorrectionConflictError(SpeakersError):
    """Raised when durable human corrections contradict one another."""


class CorrectionSet(CacheEnvelope):
    """Typed durable range corrections plus their source identity."""

    corrections: list[SpeakerCorrection]
    audio_sha256: str | None = None
    recording_identity: str | None = None


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
    corrections: Sequence[SpeakerCorrection] = (),
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
    if isinstance(agent, ClaudeAgent):
        agent = agent.for_stage("speakers")
    has_audio = _has_mergeable_audio(transcript, merged_audio)
    _validate_corrections(transcript, note, corrections)
    assigned = {item.cluster: item.name for item in assignments}
    present_clusters = list(dict.fromkeys(u.speaker for u in transcript.utterances))
    correction_clusters = {item.cluster for item in corrections}
    unassigned_clusters = [
        c
        for c in present_clusters
        if c not in assigned and c not in correction_clusters
    ]

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

    # Assignments always win over proposals. Range corrections are applied
    # afterwards, so they can override a legacy whole-cluster default locally.
    resolution: dict[str, str] = {**accepted, **assigned}
    resolved_utterances = _apply_corrections(transcript, assignments, corrections)
    resolved_utterances = [
        u.model_copy(update={"speaker": resolution[u.speaker]})
        if u.speaker in resolution
        else u
        for u in resolved_utterances
    ]
    resolved_transcript = transcript.model_copy(
        update={"utterances": resolved_utterances}
    )

    unresolved_clusters: list[str] = []
    request_positions: dict[str, list[int]] = {}
    for cluster in present_clusters:
        uncovered = [
            index
            for index, utterance in enumerate(transcript.utterances)
            if utterance.speaker == cluster
            and not any(
                utterance.start >= correction.start_seconds
                and utterance.end <= correction.end_seconds
                for correction in corrections
                if correction.cluster == cluster
            )
        ]
        if cluster in assigned or cluster in accepted:
            continue
        if uncovered:
            unresolved_clusters.append(cluster)
            request_positions[cluster] = uncovered

    requests = [
        _snippet_request(
            cluster,
            transcript.utterances,
            resolved_utterances,
            audio_tool=audio_tool,
            merged_audio=merged_audio,
            snippet_dir=snippet_dir,
            has_audio=has_audio,
            candidate_positions=request_positions[cluster],
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
    candidate_positions: Sequence[int] | None = None,
) -> SnippetRequest:
    cluster_indices = [
        i
        for i, u in enumerate(original_utterances)
        if u.speaker == cluster
        and (candidate_positions is None or i in candidate_positions)
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
    segments = [
        TimestampedSegment(
            start_seconds=original_utterances[position].start,
            end_seconds=original_utterances[position].end,
            speaker=cluster,
            text=original_utterances[position].text,
        )
        for position in picked_positions
    ]
    return SnippetRequest(
        cluster=cluster,
        clip_paths=clip_paths,
        context="\n".join(context_lines),
        segments=segments,
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


class AssignmentSet(CacheEnvelope):
    """The accumulated `--assign` overrides for one run, as cached JSON."""

    assignments: list[SpeakerAssignment]


class SpeakersResponse(BaseModel):
    """The JSON document `jake-tools transcript speakers` prints."""

    status: Literal["needs_input", "resolved"]
    run_id: str
    requests: list[SnippetRequest] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    telemetry: AITelemetry | None = None


def parse_assign(raw: str) -> SpeakerAssignment:
    """Parse one `--assign` value (`CLUSTER=NAME`) into a `SpeakerAssignment`."""
    cluster, separator, name = raw.partition("=")
    cluster = cluster.strip()
    name = name.strip()
    if not separator or not cluster or not name:
        raise InvalidAssignmentError(raw)
    return SpeakerAssignment(cluster=cluster, name=name)


def parse_correct(raw: str) -> SpeakerCorrection:
    """Parse `CLUSTER=START-END=NAME` with finite, positive half-open bounds."""
    cluster, first_separator, remainder = raw.partition("=")
    range_text, second_separator, name = remainder.partition("=")
    start_text, range_separator, end_text = range_text.partition("-")
    try:
        start = float(start_text)
        end = float(end_text)
    except ValueError as exc:
        raise InvalidCorrectionError(
            f"invalid --correct value {raw!r}; expected CLUSTER=START-END=NAME"
        ) from exc
    if (
        not first_separator
        or not second_separator
        or not cluster.strip()
        or not name.strip()
        or not range_separator
        or start < 0
        or not isfinite(start)
        or not isfinite(end)
        or end <= start
    ):
        raise InvalidCorrectionError(
            f"invalid --correct value {raw!r}; expected finite START >= 0 and END > START"
        )
    return SpeakerCorrection(
        cluster=cluster.strip(),
        start_seconds=start,
        end_seconds=end,
        name=name.strip(),
    )


parse_correction = parse_correct


def _ranges_overlap(left: SpeakerCorrection, right: SpeakerCorrection) -> bool:
    return (
        left.cluster == right.cluster
        and left.start_seconds < right.end_seconds
        and right.start_seconds < left.end_seconds
    )


def _merge_corrections(
    previous: Sequence[SpeakerCorrection], new: Sequence[SpeakerCorrection]
) -> list[SpeakerCorrection]:
    merged = list(previous)
    for correction in new:
        if correction in merged:
            continue
        for existing in merged:
            if (
                _ranges_overlap(existing, correction)
                and existing.name != correction.name
            ):
                raise SpeakerCorrectionConflictError(
                    "conflicting range corrections for "
                    f"{correction.cluster!r}: "
                    f"{existing.start_seconds}-{existing.end_seconds}={existing.name!r} "
                    "overlaps "
                    f"{correction.start_seconds}-{correction.end_seconds}={correction.name!r}"
                )
        merged.append(correction)
    return sorted(
        merged,
        key=lambda item: (
            item.cluster,
            item.start_seconds,
            item.end_seconds,
            item.name,
        ),
    )


def _validate_corrections(
    transcript: RawTranscript,
    note: ParsedNote,
    corrections: Sequence[SpeakerCorrection],
) -> None:
    present = {utterance.speaker for utterance in transcript.utterances}
    valid_names = set(note.attendees) | {_UNKNOWN}
    _merge_corrections((), corrections)
    for correction in corrections:
        if correction.cluster not in present:
            raise InvalidCorrectionError(
                f"--correct targets absent cluster {correction.cluster!r}"
            )
        if correction.name not in valid_names:
            raise InvalidCorrectionError(
                f"--correct name {correction.name!r} is not an attendee or Unknown"
            )
        if (
            correction.start_seconds < 0
            or not isfinite(correction.start_seconds)
            or not isfinite(correction.end_seconds)
            or correction.end_seconds <= correction.start_seconds
        ):
            raise InvalidCorrectionError("--correct requires finite end > start")
        matched = False
        for utterance in transcript.utterances:
            if utterance.speaker != correction.cluster:
                continue
            overlaps = (
                utterance.start < correction.end_seconds
                and correction.start_seconds < utterance.end
            )
            contained = (
                utterance.start >= correction.start_seconds
                and utterance.end <= correction.end_seconds
            )
            if overlaps and not contained:
                raise InvalidCorrectionError(
                    f"--correct range {correction.start_seconds}-{correction.end_seconds} "
                    f"has partial overlap with utterance {utterance.start}-{utterance.end}"
                )
            if contained:
                matched = True
        if not matched:
            raise InvalidCorrectionError(
                f"--correct range {correction.start_seconds}-{correction.end_seconds} "
                "contains no utterance for the target cluster"
            )


def _apply_corrections(
    transcript: RawTranscript,
    assignments: Sequence[SpeakerAssignment],
    corrections: Sequence[SpeakerCorrection],
) -> list[Utterance]:
    defaults = {assignment.cluster: assignment.name for assignment in assignments}
    corrected: list[Utterance] = []
    for utterance in transcript.utterances:
        speaker = defaults.get(utterance.speaker, utterance.speaker)
        matching = [
            correction
            for correction in corrections
            if correction.cluster == utterance.speaker
            and utterance.start >= correction.start_seconds
            and utterance.end <= correction.end_seconds
        ]
        if matching:
            speaker = matching[-1].name
        corrected.append(utterance.model_copy(update={"speaker": speaker}))
    return corrected


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


def _distinctive_quote(cluster: str, utterances: Sequence[Utterance]) -> str | None:
    """The single most representative utterance text for `cluster`, truncated.

    Reuses `_select_representative_indices`'s "longest text" heuristic
    (already used to pick evidence quotes for the resolution prompt and
    snippet requests) rather than inventing a second one - just asking for
    one pick instead of `_SNIPPETS_PER_CLUSTER`. Returns `None` only if
    `cluster` has no utterances at all, which should never happen for a
    cluster that was actually assigned.
    """
    cluster_utterances = [u for u in utterances if u.speaker == cluster]
    if not cluster_utterances:
        return None
    pick = _select_representative_indices(cluster_utterances, 1)[0]
    text = cluster_utterances[pick].text.strip()
    if len(text) > _HINT_QUOTE_MAX_CHARS:
        text = text[: _HINT_QUOTE_MAX_CHARS - 1].rstrip() + "…"
    return text


def _hint_lines_for_assignments(
    note: ParsedNote,
    assignments: Sequence[SpeakerAssignment],
    utterances: Sequence[Utterance],
) -> list[str]:
    """Hint bullets for HUMAN-confirmed mappings only.

    Per the interview record's E15 decision, the sanctioned Meeting Prep
    write is scoped to identifications Michael makes during the interactive
    snippet session (`--assign`) — never an LLM-only resolution, however
    high its confidence. `append_diarisation_hints` is idempotent, so it's
    safe to pass every non-"Unknown" assignment on record each call, not
    just ones new to this invocation.

    The bullet never names the run-local diarisation cluster id
    (`SPEAKER_NN`) - a pyannote cluster number from *this* recording has no
    meaning to a later run over different audio, where the prompt treats a
    Meeting Prep hint as outranking its own inference. Presenting a stale
    `SPEAKER_NN` as if it were still authoritative risks steering that
    later, unrelated run's proposals. Instead the hint carries the name
    plus a short, distinctive quote actually attributed to that cluster
    (`_distinctive_quote`) - identifying evidence a human can recognise
    regardless of how clusters are numbered next time.
    """
    phrase = _recording_phrase(note)
    lines: list[str] = []
    for assignment in sorted(assignments, key=lambda item: item.cluster):
        if assignment.name == _UNKNOWN:
            continue
        quote = _distinctive_quote(assignment.cluster, utterances)
        if quote is None:
            continue
        lines.append(f'{assignment.name} said "{quote}" {phrase}')
    return lines


def _hint_lines_for_corrections(
    corrections: Sequence[SpeakerCorrection],
    utterances: Sequence[Utterance],
    *,
    audio_sha256: str | None,
    recording_identity: str,
) -> list[str]:
    """Render exact range evidence without leaking run-local cluster ids."""
    lines: list[str] = []
    for correction in corrections:
        if correction.name == _UNKNOWN:
            continue
        covered = [
            utterance
            for utterance in utterances
            if utterance.speaker == correction.cluster
            and utterance.start >= correction.start_seconds
            and utterance.end <= correction.end_seconds
        ]
        quote = " ".join(utterance.text for utterance in covered)
        if not quote:
            continue
        identity = (
            f"{recording_identity} (audio {audio_sha256})"
            if audio_sha256
            else recording_identity
        )
        lines.append(
            f'{correction.name} said "{quote}" from '
            f"{correction.start_seconds}-{correction.end_seconds} seconds in {identity}"
        )
    return lines


async def run_speaker_resolution(
    note_path: Path,
    run_id: str,
    *,
    assign: Sequence[str],
    correct: Sequence[str] = (),
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
    transcript = cache.load_resumable(run_id, RAW_TRANSCRIPT_CACHE_NAME, RawTranscript)
    if transcript is None:
        raise MissingRawTranscriptError(run_id)

    new_assignments = [parse_assign(raw) for raw in assign]
    previous = cache.load(run_id, ASSIGNMENTS_CACHE_NAME, AssignmentSet)
    merged_assignments = _merge_assignments(
        previous.assignments if previous is not None else [], new_assignments
    )
    previous_assignments = previous.assignments if previous is not None else []
    new_corrections = [parse_correct(raw) for raw in correct]
    previous_correction_set = cache.load(run_id, "corrections", CorrectionSet)
    previous_corrections = (
        previous_correction_set.corrections
        if previous_correction_set is not None
        else []
    )
    merged_corrections = _merge_corrections(previous_corrections, new_corrections)
    _validate_corrections(transcript, note, merged_corrections)
    cache.store(
        run_id, ASSIGNMENTS_CACHE_NAME, AssignmentSet(assignments=merged_assignments)
    )
    cache.store(
        run_id,
        "corrections",
        CorrectionSet(
            corrections=merged_corrections,
            audio_sha256=transcript.audio_sha256,
            recording_identity=_recording_phrase(note),
        ),
    )
    if (
        merged_assignments != previous_assignments
        or merged_corrections != previous_corrections
    ):
        cache.invalidate_downstream(
            run_id, reason="speaker assignment or correction changed"
        )

    run_dir = cache.run_dir(run_id)
    stage_agent = agent.for_stage("speakers").with_telemetry(
        cache.telemetry_sink(run_id)
    )
    resolved, requests = await resolve(
        note,
        transcript,
        agent=stage_agent,
        assignments=merged_assignments,
        corrections=merged_corrections,
        audio_tool=audio_tool,
        merged_audio=run_dir / "merged.m4a",
        snippet_dir=run_dir / "snippets",
    )

    if requests and not finalise:
        return SpeakersResponse(
            status="needs_input",
            run_id=run_id,
            requests=requests,
            timings=cache.load_timings(run_id),
            telemetry=cache.load(run_id, "ai_telemetry", AITelemetry),
        )

    if requests:  # --finalise: whatever is left becomes "Unknown"
        resolved = _finalise_unknown(resolved, requests)

    cache.store(run_id, RESOLVED_TRANSCRIPT_CACHE_NAME, resolved)

    hint_lines = _hint_lines_for_assignments(
        note, merged_assignments, transcript.utterances
    ) + _hint_lines_for_corrections(
        merged_corrections,
        transcript.utterances,
        audio_sha256=transcript.audio_sha256,
        recording_identity=_recording_phrase(note),
    )
    if hint_lines:
        append_diarisation_hints(Path(note.path), hint_lines)

    return SpeakersResponse(
        status="resolved",
        run_id=run_id,
        requests=[],
        timings=cache.load_timings(run_id),
        telemetry=cache.load(run_id, "ai_telemetry", AITelemetry),
    )
