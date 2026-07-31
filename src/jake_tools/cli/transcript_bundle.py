"""Bundle control-plane commands: ``bundle create``, ``source ingest``,
``inspect``, ``validate``, ``status``, ``resume``, ``transform timeline``,
``transform transcribe``.

Every callback here is thin: parse/validate options, call one
``jake_tools.transcripts.bundle.control`` function, render the typed
result. All orchestration lives in ``control.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import click

from ..claude import ClaudeAgent
from ..transcripts.bundle.adapters import adapt_local_media, adapt_obsidian_note
from ..transcripts.bundle.apply import apply_render
from ..transcripts.bundle.components import TextEditMode
from ..transcripts.bundle.control import (
    DEFAULT_BUNDLES_ROOT,
    BundleOverview,
    BundleStatus,
    bundle_executors_with_agent,
    capability_failures,
    ingest_source,
    inspect_bundle,
    project_document,
    render_document,
    resume_run,
    run_assemble_transform,
    run_chapter_transform,
    run_minutes_transform,
    run_normalise_transform,
    run_obsidian_recording_recipe,
    run_review_apply,
    run_speakers_propose,
    run_text_transform,
    run_timeline_transform,
    run_transcribe_transform,
    status_bundle,
)
from ..transcripts.bundle.control import create_bundle as _create_bundle
from ..transcripts.bundle.document import TranscriptDocumentV1
from ..transcripts.bundle.records import NoDocumentYet, RunRecord, SourceAssociation
from ..transcripts.bundle.registry import CapabilityStatus
from ..transcripts.bundle.review import export_review_pack
from ..transcripts.bundle.store import BundleStore
from ..transcripts.errors import TranscriptError
from .context import app_context
from .options import agent, coro

#: `--kind` values that trigger a typed adapter in addition to plain
#: ingestion (source_ingest below) -- every other `--kind` value stays
#: free-text metadata only, matching the original Phase 2 behaviour.
_KIND_OBSIDIAN_NOTE = "obsidian-note"
_KIND_LOCAL_MEDIA = "local-media"

_bundle_option = click.option(
    "--bundle",
    "bundle_path",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Bundle directory (see `transcript bundle create`).",
)
_json_option = click.option(
    "--json", "as_json", is_flag=True, help="Emit machine-readable JSON."
)


def _echo_error_and_exit(exc: TranscriptError, *, as_json: bool) -> NoReturn:
    """The one error boundary every command's ``except TranscriptError``
    goes through: a clean one-line message (Click's own shape) for
    humans, or a stable JSON error object on stdout when ``--json`` was
    requested, with the same nonzero exit code. ``--json`` must never
    silently fall back to Click's plain-text error format once the
    operator asked for machine-readable output.
    """
    if as_json:
        click.echo(
            json.dumps(
                {"error": {"type": type(exc).__name__, "message": str(exc)}},
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc
    raise click.ClickException(str(exc)) from exc


# -- bundle create ------------------------------------------------------------


@click.group("bundle", help="Manage transcript bundles.")
def bundle_group() -> None:
    pass


@bundle_group.command("create")
@click.option(
    "--root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=f"Parent directory for the new bundle. Defaults to {DEFAULT_BUNDLES_ROOT}/.",
)
@_json_option
def bundle_create(root: Path | None, as_json: bool) -> None:
    """Create a new, empty bundle and print its IDs.

    The bundle's own directory is named after its bundle_id (minted by the
    store; never chosen ahead of time) and lives under --root. Ingest
    sources into it with `transcript source ingest --bundle <path> ...`.
    """
    try:
        created = _create_bundle(root)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "bundle_id": created.manifest.bundle_id,
                    "document_id": created.manifest.document_id,
                    "root": str(created.store.root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    click.echo(f"bundle_id: {created.manifest.bundle_id}")
    click.echo(f"document_id: {created.manifest.document_id}")
    click.echo(f"root: {created.store.root}")


# -- source ingest --------------------------------------------------------


@click.group("source", help="Register and ingest bundle sources.")
def source_group() -> None:
    pass


def _rendered_invocation(ctx: click.Context) -> str:
    """The genuine command line that produced this invocation (M3:
    "naming inputs on the command line IS the operator assertion ...
    recorded as such"), reconstructed from Click's own parsed parameters
    for *this* command -- every option/argument Click accepted, not a
    hand-picked subset that silently goes stale the next time a flag is
    added (the earlier version dropped --bundle/--json and hardcoded the
    command name; this cannot drift the same way).
    """
    parts = [ctx.command_path]
    for parameter in ctx.command.params:
        value = ctx.params.get(parameter.name) if parameter.name else None
        if value is None or value is False:
            continue
        if isinstance(parameter, click.Argument):
            parts.append(str(value))
            continue
        flag = parameter.opts[0]
        parts.append(flag if value is True else f"{flag} {value}")
    return " ".join(parts)


@source_group.command("ingest")
@_bundle_option
@click.option(
    "--kind",
    required=True,
    help="What this artefact is. Free text for most sources (e.g. "
    "teams-transcript), but 'obsidian-note' and 'local-media' additionally "
    "trigger a typed adapter (destination/participants/recording-references, "
    "or ffprobe-verified media metadata) alongside plain ingestion.",
)
@click.option(
    "--producer",
    required=True,
    help="What produced these bytes (e.g. an adapter name, or 'operator').",
)
@click.option(
    "--source",
    "source_id",
    default=None,
    help="Reuse an existing source's membership instead of registering a new one.",
)
@click.option(
    "--provider-id",
    default=None,
    help="External provider ID (Teams/Google event ID, YouTube video ID) as the "
    "association evidence. Registers a new source.",
)
@click.option(
    "--note-embed",
    default=None,
    help="Obsidian embed link (plus note snapshot) as the association evidence. "
    "Registers a new source.",
)
@click.option(
    "--participant",
    "participants",
    multiple=True,
    help="Declare a meeting participant by display name (repeatable). Use this "
    "when the note names its people in the body rather than in a frontmatter "
    "Attendees list -- naming them here is the operator assertion (M19), and "
    "body wikilinks are never guessed at. Only meaningful with "
    "--kind obsidian-note.",
)
@_json_option
@click.argument(
    "input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.pass_context
def source_ingest(
    ctx: click.Context,
    bundle_path: Path,
    kind: str,
    producer: str,
    source_id: str | None,
    provider_id: str | None,
    note_embed: str | None,
    participants: tuple[str, ...],
    as_json: bool,
    input_file: Path,
) -> None:
    """Register a source (if needed) and ingest INPUT_FILE as an artefact.

    Without --source/--provider-id/--note-embed, this registers a brand
    new source with association `operator-assertion`: naming INPUT_FILE
    on this command line *is* the assertion that it belongs to --bundle
    (M3), and the evidence recorded is the genuine invocation -- every
    flag actually supplied, not a hand-picked reconstruction. --source
    reuses an existing source's membership instead of registering a new
    one. Ingestion is idempotent: re-running the exact same command
    against the same source is a no-op that reports the existing
    artefact rather than duplicating it.

    `--kind obsidian-note` additionally parses INPUT_FILE as an Obsidian
    source note (destination component, participants from frontmatter
    Attendees and/or --participant, recording references from embeds --
    M3/M13/M19; a
    reference alone never satisfies `media.recording`, ingest the
    recording separately with `--kind local-media`). `--kind local-media`
    additionally runs `ffprobe` over INPUT_FILE and records its
    duration/codec/sample-rate/channels as the `media.recording` proof
    (M4/M11). Both adapter calls are themselves idempotent (content-
    addressed components), so re-running this command is always safe.
    """
    given = [
        flag
        for flag, value in (
            ("--source", source_id),
            ("--provider-id", provider_id),
            ("--note-embed", note_embed),
        )
        if value is not None
    ]
    if len(given) > 1:
        raise click.UsageError(f"Use at most one of {', '.join(given)}.")

    acquisition_locator = str(input_file)
    if provider_id is not None:
        association, evidence = SourceAssociation.PROVIDER_ID, provider_id
    elif note_embed is not None:
        association, evidence = SourceAssociation.NOTE_EMBED, note_embed
    else:
        association = SourceAssociation.OPERATOR_ASSERTION
        evidence = _rendered_invocation(ctx)

    store = BundleStore(bundle_path)
    adapter_summary: dict[str, object] = {}
    try:
        outcome = ingest_source(
            store,
            source_id=source_id,
            association=association,
            evidence=evidence,
            content=input_file.read_bytes(),
            kind=kind,
            producer=producer,
            acquisition_locator=acquisition_locator,
        )
        if kind == _KIND_OBSIDIAN_NOTE:
            note_adaptation = adapt_obsidian_note(
                store,
                note_artefact_id=outcome.artefact.artefact_id,
                note_path=input_file,
                operator_participants=participants,
            )
            adapter_summary = {
                "destination_component_id": note_adaptation.destination.component_id,
                "participants_component_id": note_adaptation.participants.component_id,
                "reference_set_component_id": (
                    note_adaptation.reference_set.component_id
                    if note_adaptation.reference_set is not None
                    else None
                ),
            }
        elif kind == _KIND_LOCAL_MEDIA:
            media = adapt_local_media(
                store,
                source_artefact_id=outcome.artefact.artefact_id,
                media_path=input_file,
            )
            adapter_summary = {
                "media_recording_component_id": media.component_id,
                "duration_ms": media.duration_ms,
            }
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "source_id": outcome.membership.source_id,
                    "association": outcome.membership.association.value,
                    "evidence": outcome.membership.evidence,
                    "artefact_id": outcome.artefact.artefact_id,
                    "sha256": outcome.artefact.sha256,
                    "kind": outcome.artefact.kind,
                    "producer": outcome.artefact.producer,
                    "already_ingested": outcome.already_ingested,
                    **adapter_summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    click.echo(f"source_id: {outcome.membership.source_id}")
    if outcome.already_ingested:
        click.echo(f"already ingested, artefact_id {outcome.artefact.artefact_id}")
    else:
        click.echo(f"artefact_id: {outcome.artefact.artefact_id}")
    for key, value in adapter_summary.items():
        click.echo(f"{key}: {value}")


# -- shared rendering helpers ----------------------------------------------


def _document_state_payload(
    document: TranscriptDocumentV1 | NoDocumentYet,
) -> dict[str, object]:
    if isinstance(document, NoDocumentYet):
        return {
            "state": "no-document-yet",
            "candidate_artefact_ids": list(document.candidate_artefact_ids),
        }
    return {
        "state": "revision",
        "revision_id": document.revision_id,
        "parent_revision_ids": list(document.parent_revision_ids),
    }


def _capabilities_payload(
    document: TranscriptDocumentV1 | NoDocumentYet,
) -> dict[str, object]:
    if isinstance(document, NoDocumentYet):
        return {}
    return {
        key.value: record.model_dump(mode="json")
        for key, record in document.capabilities.items()
    }


def _heads(document: TranscriptDocumentV1 | NoDocumentYet) -> tuple[str, ...]:
    """v1's single-head bundles (D3 defers richer branching) mean this is
    always zero or one revision ID; kept plural to match the "revisions
    (count + heads)" fact `inspect` reports as one unit, and to stay
    correct if a later phase ever allows more than one.
    """
    if isinstance(document, NoDocumentYet):
        return ()
    return (document.revision_id,)


def _echo_head(document: TranscriptDocumentV1 | NoDocumentYet) -> None:
    if isinstance(document, NoDocumentYet):
        count = len(document.candidate_artefact_ids)
        if count:
            click.echo(f"head: no document yet ({count} candidate artefact(s))")
        else:
            click.echo("head: no document yet")
        return
    click.echo(f"head: {document.revision_id}")


def _echo_capabilities(document: TranscriptDocumentV1 | NoDocumentYet) -> None:
    click.echo("capabilities:")
    if isinstance(document, NoDocumentYet):
        click.echo("  (none -- no document yet)")
        return
    for key, record in sorted(
        document.capabilities.items(), key=lambda kv: kv[0].value
    ):
        click.echo(f"  {key.value}: {record.status.value}")
        for member in record.members:
            marker = " FAILED" if member.status == CapabilityStatus.FAILED else ""
            click.echo(f"    {member.member_id}: {member.status.value}{marker}")


# -- inspect ------------------------------------------------------------


def _overview_payload(overview: BundleOverview) -> dict[str, object]:
    return {
        "bundle_id": overview.bundle_id,
        "document_id": overview.document_id,
        "root": str(overview.root),
        "document": _document_state_payload(overview.document),
        "sources": [
            membership.model_dump(mode="json") for membership in overview.sources
        ],
        "artefact_ids": list(overview.artefact_ids),
        "revision_count": overview.revision_count,
        "heads": list(_heads(overview.document)),
        "runs": [
            {
                "run_id": run.run_id,
                "state": run.state.value,
                "next_action": run.next_action.model_dump(mode="json")
                if run.next_action
                else None,
            }
            for run in overview.runs
        ],
        "components_by_kind": dict(overview.components_by_kind),
        "capabilities": _capabilities_payload(overview.document),
    }


def _echo_overview(overview: BundleOverview) -> None:
    click.echo(f"bundle_id: {overview.bundle_id}")
    click.echo(f"document_id: {overview.document_id}")
    click.echo(f"root: {overview.root}")
    click.echo("")
    _echo_head(overview.document)
    click.echo("")
    click.echo(f"sources ({len(overview.sources)}):")
    for membership in overview.sources:
        click.echo(
            f"  {membership.source_id}  {membership.association.value}  "
            f"{membership.evidence!r}"
        )
    click.echo("")
    click.echo(f"artefacts ({len(overview.artefact_ids)}):")
    for artefact_id in overview.artefact_ids:
        click.echo(f"  {artefact_id}")
    click.echo("")
    heads = _heads(overview.document)
    heads_suffix = f" (heads: {', '.join(heads)})" if heads else ""
    click.echo(f"revisions: {overview.revision_count}{heads_suffix}")
    click.echo("")
    click.echo(f"runs ({len(overview.runs)}):")
    for run in overview.runs:
        next_action = f"  next_action={run.next_action.kind}" if run.next_action else ""
        click.echo(f"  {run.run_id}  {run.state.value}{next_action}")
    click.echo("")
    click.echo("components by kind:")
    if overview.components_by_kind:
        for kind, count in sorted(overview.components_by_kind.items()):
            click.echo(f"  {kind}: {count}")
    else:
        click.echo("  (none)")
    click.echo("")
    _echo_capabilities(overview.document)


@click.command("inspect")
@_bundle_option
@_json_option
def inspect_command(bundle_path: Path, as_json: bool) -> None:
    """Show a bundle's overview: IDs, head, sources, artefacts, revisions,
    runs, components, and capability statuses.

    Reads the head projection (M16) -- a bundle with no assembly revision
    yet reports "no document yet" plus the candidate artefacts waiting to
    be assembled, never an error.
    """
    store = BundleStore(bundle_path)
    try:
        overview = inspect_bundle(store)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    if as_json:
        click.echo(json.dumps(_overview_payload(overview), indent=2, sort_keys=True))
        return
    _echo_overview(overview)


# -- validate ------------------------------------------------------------


@click.command("validate")
@_bundle_option
@click.option(
    "--revision",
    "revision_id",
    default=None,
    help="Validate this revision instead of the bundle's current head.",
)
@_json_option
@click.pass_context
def validate_command(
    ctx: click.Context, bundle_path: Path, revision_id: str | None, as_json: bool
) -> None:
    """Project the head (or --revision) and report per-capability status.

    A bundle with no assembly revision yet is a clean report ("no
    document yet"), never an error. Exits 1 if any capability has
    genuinely failed validation (M4) -- a many-cardinality key's failed
    member, or a one-cardinality key's own failed top-level status (which
    has no members, e.g. more than one participants.declared candidate).
    A merely absent or not-available-from-source status never fails the
    command.
    """
    store = BundleStore(bundle_path)
    try:
        document = project_document(store, revision_id=revision_id)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    failures = (
        capability_failures(document)
        if isinstance(document, TranscriptDocumentV1)
        else ()
    )

    if as_json:
        click.echo(
            json.dumps(
                {
                    "bundle_id": document.bundle_id,
                    "document": _document_state_payload(document),
                    "capabilities": _capabilities_payload(document),
                    "capability_failures": [
                        {
                            "capability": failure.capability.value,
                            "member_id": failure.member_id,
                            "detail": failure.detail,
                        }
                        for failure in failures
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        click.echo(f"bundle_id: {document.bundle_id}")
        _echo_head(document)
        click.echo("")
        _echo_capabilities(document)
        if failures:
            click.echo("")
            click.echo(f"failed capabilities ({len(failures)}):")
            for failure in failures:
                label = (
                    f"{failure.capability.value} / {failure.member_id}"
                    if failure.member_id is not None
                    else failure.capability.value
                )
                click.echo(f"  {label}: {failure.detail}")

    if failures:
        ctx.exit(1)


# -- status ------------------------------------------------------------


def _status_payload(status: BundleStatus) -> dict[str, object]:
    lease_payload: dict[str, object] | None = None
    if status.lease is not None:
        lease_payload = {
            "run_id": status.lease.run_id,
            "pid": status.lease.pid,
            "started_at": status.lease.started_at.isoformat(),
            "alive": status.lease_holder_alive,
        }
    return {
        "bundle_id": status.bundle_id,
        "lease": lease_payload,
        "runs": [
            {
                "run_id": run.run_id,
                "state": run.state.value,
                "next_action": run.next_action.model_dump(mode="json")
                if run.next_action
                else None,
                "resumes_run_id": run.resumes_run_id,
                "takeover_of_run_id": run.takeover_of_run_id,
                "created_at": run.created_at.isoformat(),
                "started_at": run.started_at.isoformat() if run.started_at else None,
            }
            for run in status.runs
        ],
    }


def _echo_status(status: BundleStatus) -> None:
    click.echo(f"bundle_id: {status.bundle_id}")
    if status.lease is None:
        click.echo("lease: none held")
    else:
        liveness = "alive" if status.lease_holder_alive else "dead"
        click.echo(
            f"lease: run {status.lease.run_id}, pid {status.lease.pid} ({liveness}), "
            f"since {status.lease.started_at.isoformat()}"
        )
    click.echo("")
    click.echo(f"runs ({len(status.runs)}):")
    for run in status.runs:
        next_action = f"  next_action={run.next_action.kind}" if run.next_action else ""
        started = f", started {run.started_at.isoformat()}" if run.started_at else ""
        click.echo(
            f"  {run.run_id}  {run.state.value}{next_action}  "
            f"created {run.created_at.isoformat()}{started}"
        )


@click.command("status")
@_bundle_option
@_json_option
def status_command(bundle_path: Path, as_json: bool) -> None:
    """Show every run's state and the bundle's lease, if one is held.

    A run in review_required/refused/failed is durable and resumable
    (`transcript resume`). The lease, when held, names the run currently
    allowed to move the head, plus whether its holding process is still
    alive -- a dead holder is a crash, not an active operation (see
    `transcript resume --take-over`).
    """
    store = BundleStore(bundle_path)
    try:
        status = status_bundle(store)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    if as_json:
        click.echo(json.dumps(_status_payload(status), indent=2, sort_keys=True))
        return
    _echo_status(status)


# -- resume ------------------------------------------------------------


@click.command("resume")
@_bundle_option
@click.option(
    "--run",
    "run_id",
    default=None,
    help="Run to resume. Defaults to the single resumable run, if unambiguous. "
    "With --take-over, defaults to the run currently holding the lease.",
)
@click.option(
    "--take-over",
    "take_over",
    is_flag=True,
    help="Recover a crashed run: a lease held by a dead process on a run stuck "
    "'running' (its process was killed mid-dispatch and never released the "
    "lease itself). Without this flag, that state is a diagnosable refusal "
    "naming this flag -- never silently retried.",
)
@_json_option
@agent
@click.pass_context
def resume_command(
    ctx: click.Context,
    claude_agent: ClaudeAgent,
    bundle_path: Path,
    run_id: str | None,
    take_over: bool,
    as_json: bool,
) -> None:
    """Re-acquire a durable-state run's lease and dispatch its recorded
    next_action (M2).

    Never re-plans: the exact operation, input IDs, and config hash
    recorded when the run entered review_required/refused/failed are
    replayed as-is. `timeline` and `transcribe` (this bundle's own
    transforms, see `transform timeline`/`transform transcribe`) have
    real registered executors; resuming a run whose next_action names any
    other, still-unimplemented operation kind is a safe, explicit no-op:
    the error names the missing kind, and the run is left exactly where
    it was, lease released, ready to resume again once that transform
    exists. If the executor itself raises, the run moves to `failed`
    (durable, resumable) with the exception recorded, rather than being
    left stuck `running` with the lease held forever.
    """
    # An injected mapping (tests, diagnostics) wins for the kinds it
    # defines; the agent-bound kinds fill the gaps, so a run waiting at
    # the review checkpoint with next_action=products can actually be
    # resumed here without every caller having to assemble the mapping.
    executors = dict(
        bundle_executors_with_agent(claude_agent, model=claude_agent.defaults.model)
    )
    executors.update(app_context(ctx).bundle_executors)
    store = BundleStore(bundle_path)
    try:
        run = resume_run(store, run_id=run_id, executors=executors, take_over=take_over)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    head_revision_id = store.load_manifest().head_revision_id
    if as_json:
        click.echo(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "state": run.state.value,
                    "next_action": run.next_action.model_dump(mode="json")
                    if run.next_action
                    else None,
                    "head_revision_id": head_revision_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    click.echo(f"run_id: {run.run_id}")
    click.echo(f"state: {run.state.value}")
    if head_revision_id is not None:
        click.echo(f"head: {head_revision_id}")


# -- transform: timeline / transcribe (M6 / M11) --------------------------


@click.group(
    "transform",
    help="Run bundle transforms (timeline, transcribe, normalise, "
    "speakers-propose, text, chapter, minutes).",
)
def transform_group() -> None:
    pass


def _echo_run_result(run: RunRecord, store: BundleStore, *, as_json: bool) -> None:
    """Shared success rendering for `transform timeline`/`transform
    transcribe` -- both are a fresh run dispatched immediately
    (`run_bundle_operation`, ``control.py``), so both report the same
    shape: run id, terminal state, and the bundle's head afterwards.
    """
    head_revision_id = store.load_manifest().head_revision_id
    if as_json:
        click.echo(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "state": run.state.value,
                    "head_revision_id": head_revision_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    click.echo(f"run_id: {run.run_id}")
    click.echo(f"state: {run.state.value}")
    if head_revision_id is not None:
        click.echo(f"head: {head_revision_id}")


@transform_group.command("assemble")
@_bundle_option
@click.option(
    "--rationale",
    default="",
    help="Why these sources belong in this document (recorded on the revision).",
)
@_json_option
def transform_assemble_command(
    bundle_path: Path, rationale: str, as_json: bool
) -> None:
    """Bring every ingested candidate into the document (M18).

    Ingesting a source creates a *candidate*; nothing downstream can see
    it until an assembly revision selects it. This selects every
    unassembled artefact, taking each one's role from the `--kind` you
    declared at ingest -- obsidian-note becomes the destination and the
    notes carrier, local-media becomes media, and anything unrecognised
    becomes evidence-only rather than being guessed at.

    Safe to re-run after ingesting more sources: already-assembled
    artefacts and components are left alone.
    """
    store = BundleStore(bundle_path)
    try:
        run = run_assemble_transform(store)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)
    _echo_run_result(run, store, as_json=as_json)


@transform_group.command("timeline")
@_bundle_option
@click.option(
    "--order",
    default=None,
    help="Comma-separated artefact IDs giving explicit member order (M6: "
    '"explicit operator order"). Defaults to note-embed order, read from the '
    "bundle's own recording-reference-set component.",
)
@_json_option
def transform_timeline_command(
    bundle_path: Path, order: str | None, as_json: bool
) -> None:
    """Build the M6 combined-timeline component over the bundle's media
    members and move the head to a revision carrying it.

    Requires an already-assembled document (`transform assemble` first).
    Piecewise-maps each media member's own `[0, duration)` into a shared
    combined timeline, sequential and non-overlapping (v1 policy) --
    default member order is the note's own embed order; --order overrides
    it explicitly. Refuses if no media member resolves (no explicit order
    given and no note-embed reference resolved to ingested media).
    """
    store = BundleStore(bundle_path)
    resolved_order = tuple(order.split(",")) if order else None
    try:
        run = run_timeline_transform(store, order=resolved_order)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)
    _echo_run_result(run, store, as_json=as_json)


@transform_group.command("transcribe")
@_bundle_option
@click.option(
    "--force",
    is_flag=True,
    help="Re-invoke the worker even for media whose ASR/diarisation already "
    "match this request's own fingerprint (M11). Recorded on the run's "
    "rationale.",
)
@_json_option
def transform_transcribe_command(bundle_path: Path, force: bool, as_json: bool) -> None:
    """Invoke the pinned local inference worker (Parakeet ASR + Pyannote
    Community-1 diarisation) over every media member and promote
    inference.asr/inference.diarisation (M11).

    Resume-by-hash: a media member whose ASR and diarisation already
    match this request's own fingerprint (audio hash + declared model +
    audio-preparation config, computed before the worker ever runs) is
    never re-invoked -- pass --force to override anyway. Completed stages
    promote even when a sibling stage fails (M11's partial-failure rule):
    diarisation failing for one recording never blocks that recording's
    own ASR capability, nor any other recording's. Requires local `uv`
    and the `inference-worker/` project's own environment (its first
    invocation may need to resolve model weights; see its own docs).
    """
    store = BundleStore(bundle_path)
    try:
        run = run_transcribe_transform(store, force=force)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)
    _echo_run_result(run, store, as_json=as_json)


@transform_group.command("normalise")
@_bundle_option
@_json_option
def transform_normalise_command(bundle_path: Path, as_json: bool) -> None:
    """Turn raw ASR and diarisation evidence into canonical timed turns
    (M7), with machine voice clusters and full drop lineage.

    Needs a combined timeline (`transform timeline`) and at least one
    completed ASR result (`transform transcribe`). Raw artefacts are never
    rewritten -- zero-length and duplicate tokens are legal evidence there
    and are removed from the *canonical* set with a recorded reason. Voice
    clusters are recorded as clusters, never as people; assigning them is
    the review step's job and nothing here does it.

    Re-running supersedes the previous canonical set. That mints fresh
    turn IDs, so an applied review bound to the old inventory is correctly
    invalidated rather than carried onto different turns.
    """
    store = BundleStore(bundle_path)
    try:
        run = run_normalise_transform(store)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)
    _echo_run_result(run, store, as_json=as_json)


@transform_group.command("speakers-propose")
@_bundle_option
@click.option(
    "--no-stop-for-review",
    "stop_for_review",
    flag_value=False,
    default=True,
    help="Do not enter review_required after proposing. Proposals still never "
    "satisfy the meeting-note speaker gate, so this skips only the durable "
    "checkpoint -- it never publishes an unreviewed attribution.",
)
@_json_option
@agent
@coro
async def transform_speakers_propose_command(
    claude_agent: ClaudeAgent,
    bundle_path: Path,
    stop_for_review: bool,
    as_json: bool,
) -> None:
    """Propose a participant for each machine voice cluster (M8 rung 7),
    then stop at the durable review checkpoint.

    A proposal is evidence for a human, never an assignment: a bare
    machine hypothesis can never render into a meeting note (M5). The run
    ends in `review_required` with its lease released, so you can export a
    pack (`review export`), decide each cluster -- "unclear speaker" is a
    perfectly good decision -- apply it (`review apply`), and `resume`,
    without repeating ingest, transcription, or normalisation.
    """
    store = BundleStore(bundle_path)
    try:
        run = await run_speakers_propose(
            store,
            agent=claude_agent,
            model=claude_agent.defaults.model,
            stop_for_review=stop_for_review,
        )
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)
    _echo_run_result(run, store, as_json=as_json)


@transform_group.command("text")
@_bundle_option
@click.option(
    "--mode",
    type=click.Choice([mode.value for mode in TextEditMode]),
    required=True,
    help="correct: fix mis-transcriptions only. polish: readability only "
    "(filler, stutters, punctuation). They are separate passes with separate "
    "provenance, and correct should run first.",
)
@click.option(
    "--context",
    "context_note",
    default="",
    help="One line of meeting context for the stage (subject, organisation).",
)
@_json_option
@agent
@coro
async def transform_text_command(
    claude_agent: ClaudeAgent,
    bundle_path: Path,
    mode: str,
    context_note: str,
    as_json: bool,
) -> None:
    """Rewrite canonical turn text (M9) -- and nothing else.

    Timings, source lineage, and turn IDs are re-attached from the input
    turns, so a text pass can never move a turn, change a speaker, or
    invent an editorial node. Every difference is accounted for in a
    ledger entry, and the pass is refused outright if it fails the
    retention and fidelity gates.

    Because turn IDs survive, running this *after* a speaker review does
    not invalidate the review.
    """
    store = BundleStore(bundle_path)
    try:
        outcome = await run_text_transform(
            store,
            mode=TextEditMode(mode),
            agent=claude_agent,
            model=claude_agent.defaults.model,
            context_note=context_note,
        )
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    payload = {
        "revision_id": outcome.revision.revision_id,
        "mode": mode,
        "changed_turns": outcome.changed_turn_count,
        "dropped_turns": outcome.dropped_turn_count,
        "ledger_component_id": outcome.ledger.component_id,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        click.echo(f"{key}: {value}")


@transform_group.command("chapter")
@_bundle_option
@click.option("--context", "context_note", default="", help="Meeting context line.")
@_json_option
@agent
@coro
async def transform_chapter_command(
    claude_agent: ClaudeAgent, bundle_path: Path, context_note: str, as_json: bool
) -> None:
    """Divide the canonical turn sequence into chapters (M10).

    Coverage is exact by construction: the returned boundaries are
    projected back onto the real ordered sequence, so every turn lands in
    exactly one chapter and the capability refuses to validate if it
    does not.
    """
    store = BundleStore(bundle_path)
    try:
        outcome = await run_chapter_transform(
            store,
            agent=claude_agent,
            model=claude_agent.defaults.model,
            context_note=context_note,
        )
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    payload = {
        "revision_id": outcome.revision.revision_id,
        "chapters": [
            {"title": chapter.title, "start_ms": chapter.start_ms}
            for chapter in outcome.chapters.chapters
        ],
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(f"revision_id: {outcome.revision.revision_id}")
    for chapter in outcome.chapters.chapters:
        click.echo(f"  {chapter.start_ms:>9}ms  {chapter.title}")


@transform_group.command("minutes")
@_bundle_option
@click.option("--context", "context_note", default="", help="Meeting context line.")
@_json_option
@agent
@coro
async def transform_minutes_command(
    claude_agent: ClaudeAgent, bundle_path: Path, context_note: str, as_json: bool
) -> None:
    """Derive evidence-linked minutes (M10).

    Every finding cites the turns or notes sections it came from; an
    uncited finding is dropped before storage and refused again at render
    time. Owners come only from participant records -- a name mentioned in
    passing never becomes an assignee.
    """
    store = BundleStore(bundle_path)
    try:
        outcome = await run_minutes_transform(
            store,
            agent=claude_agent,
            model=claude_agent.defaults.model,
            context_note=context_note,
        )
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    payload = {
        "revision_id": outcome.revision.revision_id,
        "summary": outcome.minutes.summary.text,
        "findings": [
            {"kind": finding.kind.value, "text": finding.text}
            for finding in outcome.minutes.findings
        ],
        "dropped_unsourced_findings": list(outcome.dropped_unsourced_findings),
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    click.echo(f"revision_id: {outcome.revision.revision_id}")
    click.echo(f"summary: {outcome.minutes.summary.text}")
    for finding in outcome.minutes.findings:
        click.echo(f"  [{finding.kind.value}] {finding.text}")
    for dropped in outcome.dropped_unsourced_findings:
        click.echo(f"  DROPPED (unsourced): {dropped}")


# -- review: export / apply (M8) --------------------------------------------


@click.group("review", help="Export and apply the speaker review pack (M8).")
def review_group() -> None:
    pass


@review_group.command("export")
@_bundle_option
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the review pack JSON.",
)
@_json_option
def review_export_command(bundle_path: Path, out_path: Path, as_json: bool) -> None:
    """Write a review pack bound to this bundle's current head.

    The pack lists every voice cluster (plus any turn no cluster covers),
    the machine's proposal for each, and the full transcript. Fill in
    `reviewer` and add one entry to `decisions` per item, then
    `review apply` it.

    Leaving an item out is a partial review, recorded honestly -- but an
    undecided turn does not satisfy the meeting-note speaker gate. Set
    `remaining` to "unclear-speaker" to record every item you did not
    decide as an explicit unclear-speaker decision. There is deliberately
    no way to bulk-assign a person: that would be a guess wearing your
    name.

    Exporting reads only; it takes no lease and changes nothing.
    """
    store = BundleStore(bundle_path)
    try:
        written = export_review_pack(store, destination=out_path)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    if as_json:
        click.echo(json.dumps({"pack": str(written)}, indent=2, sort_keys=True))
        return
    click.echo(f"pack: {written}")


@review_group.command("apply")
@_bundle_option
@_json_option
@click.argument(
    "pack_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def review_apply_command(bundle_path: Path, as_json: bool, pack_path: Path) -> None:
    """Apply a filled review pack (M8).

    Refuses a pack whose bound revision or turn/cluster inventory no
    longer matches the head -- re-export rather than applying decisions to
    turns the reviewer never saw. Applying the same pack twice is a no-op
    that returns the first application's revision. A second, differing
    review against the same input revision is rejected in v1.
    """
    store = BundleStore(bundle_path)
    try:
        outcome = run_review_apply(store, pack_path=pack_path)
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    payload = {
        "review_id": outcome.review.review_id,
        "result_revision_id": outcome.review.result_revision_id,
        "already_applied": outcome.already_applied,
        "addressed_items": outcome.addressed_item_count,
        "total_items": outcome.total_item_count,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        click.echo(f"{key}: {value}")


# -- render / apply (M17 / M13) ---------------------------------------------


@click.command("render")
@_bundle_option
@click.option(
    "--revision",
    "revision_id",
    default=None,
    help="Render this revision instead of the current head.",
)
@click.option(
    "--no-transcript",
    "include_transcript",
    flag_value=False,
    default=True,
    help="Render meeting notes and chapters only, omitting the transcript body.",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also write the rendered markdown here (it is always stored in the "
    "bundle regardless).",
)
@_json_option
def render_command(
    bundle_path: Path,
    revision_id: str | None,
    include_transcript: bool,
    out_path: Path | None,
    as_json: bool,
) -> None:
    """Render the DUM-C meeting note for one revision (M17).

    Pure and byte-deterministic: the same revision rendered twice produces
    the same bytes. Refuses if any turn would publish an unreviewed
    speaker attribution (M5), or if the profile's required chapters or
    minutes are missing or did not validate.

    This writes nothing into the note. `apply` does that, separately.
    """
    store = BundleStore(bundle_path)
    try:
        record = render_document(
            store, revision_id=revision_id, include_transcript=include_transcript
        )
        body = store.load_render_output(record.render_id).decode("utf-8")
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body, encoding="utf-8")

    if as_json:
        click.echo(
            json.dumps(
                {
                    "render_id": record.render_id,
                    "revision_id": record.revision_id,
                    "output_sha256": record.output_sha256,
                    "out": str(out_path) if out_path else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    click.echo(f"render_id: {record.render_id}")
    click.echo(f"revision_id: {record.revision_id}")
    click.echo(f"output_sha256: {record.output_sha256}")
    if out_path is not None:
        click.echo(f"out: {out_path}")


@click.command("apply")
@_bundle_option
@click.option("--render", "render_id", required=True, help="Render to apply.")
@click.option(
    "--target",
    "target_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Note to write. Use a copy while evaluating -- never the vault original.",
)
@click.option(
    "--allow-stale-render",
    is_flag=True,
    help="Apply a render of a revision that is no longer the head. Recorded on "
    "the apply record.",
)
@click.option(
    "--adopt-edited-region",
    is_flag=True,
    help="Authorise adopting legacy generated sections that have authored content "
    "between them. Without this the migration fails closed. Recorded.",
)
@_json_option
def apply_command(
    bundle_path: Path,
    render_id: str,
    target_path: Path,
    allow_stale_render: bool,
    adopt_edited_region: bool,
    as_json: bool,
) -> None:
    """Write a render into its note's owned region (M13).

    Everything outside the `<!-- jake-tools:transcript:begin/end -->`
    markers is preserved byte for byte, including authored content *below*
    the generated sections -- the specific thing the legacy merge path
    truncated. A note that predates the markers has its legacy
    `## Meeting Notes` / `## Chapters` / `## Transcript` sections adopted,
    failing closed if authored content sits between them.

    Refuses before writing if the note changed since the render was
    computed, or if the render is of a superseded revision. The write is
    atomic and read back before it is reported as verified.
    """
    store = BundleStore(bundle_path)
    try:
        outcome = apply_render(
            store,
            render_id=render_id,
            target_path=target_path,
            allow_stale_render=allow_stale_render,
            adopt_edited_region=adopt_edited_region,
        )
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    payload = {
        "apply_id": outcome.record.apply_id,
        "state": outcome.record.state.value,
        "target": outcome.record.target_path,
        "written": outcome.written,
        "unchanged": outcome.unchanged,
        "migrated_legacy_headings": outcome.record.migrated_legacy_headings,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        click.echo(f"{key}: {value}")


# -- recipe (the composed path) ---------------------------------------------


@click.group("recipe", help="Composed transcript paths.")
def recipe_group() -> None:
    pass


@recipe_group.command("obsidian-recording")
@_bundle_option
@click.option("--context", "context_note", default="", help="Meeting context line.")
@click.option(
    "--no-stop-for-review",
    "stop_for_review",
    flag_value=False,
    default=True,
    help="Run straight through to the products without stopping for review. The "
    "speaker gate still applies, so this only succeeds once a review is applied.",
)
@_json_option
@agent
@coro
async def recipe_obsidian_recording_command(
    claude_agent: ClaudeAgent,
    bundle_path: Path,
    context_note: str,
    stop_for_review: bool,
    as_json: bool,
) -> None:
    """Run an already-ingested Obsidian note's recordings all the way to a
    polished, chaptered transcript and meeting notes.

    Timeline, transcription, normalisation, and speaker proposal, then a
    stop at the durable review checkpoint. Each step is skipped when its
    output already exists, so re-running after a speaker correction never
    re-runs acquisition or ASR.

    After `review apply`, either re-run this command or `resume` the
    waiting run to build the corrected and polished transcript, chapters,
    minutes, and the render. Writing the note is a separate, explicit
    `apply`.
    """
    store = BundleStore(bundle_path)
    try:
        outcome = await run_obsidian_recording_recipe(
            store,
            agent=claude_agent,
            model=claude_agent.defaults.model,
            context_note=context_note,
            stop_for_review=stop_for_review,
        )
    except TranscriptError as exc:
        _echo_error_and_exit(exc, as_json=as_json)

    payload: dict[str, object] = {
        "run_id": outcome.run.run_id,
        "state": outcome.run.state.value,
        "stopped_for_review": outcome.stopped_for_review,
        "head_revision_id": store.load_manifest().head_revision_id,
    }
    if outcome.products is not None:
        payload["render_id"] = outcome.products.render.render_id
        payload["output_sha256"] = outcome.products.render.output_sha256
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        click.echo(f"{key}: {value}")
    if outcome.stopped_for_review:
        click.echo("")
        click.echo(
            "Stopped for speaker review. Next: `transcript review export --bundle "
            f"{bundle_path} --out review.json`, fill it in, then `transcript review "
            f"apply --bundle {bundle_path} review.json` and re-run this command."
        )
