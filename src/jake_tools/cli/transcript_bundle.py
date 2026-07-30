"""Bundle control-plane commands: ``bundle create``, ``source ingest``,
``inspect``, ``validate``, ``status``, ``resume``.

Every callback here is thin: parse/validate options, call one
``jake_tools.transcripts.bundle.control`` function, render the typed
result. All orchestration lives in ``control.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import click

from ..transcripts.bundle.control import (
    DEFAULT_BUNDLES_ROOT,
    BundleOverview,
    BundleStatus,
    capability_failures,
    ingest_source,
    inspect_bundle,
    project_document,
    resume_run,
    status_bundle,
)
from ..transcripts.bundle.control import create_bundle as _create_bundle
from ..transcripts.bundle.document import TranscriptDocumentV1
from ..transcripts.bundle.records import NoDocumentYet, SourceAssociation
from ..transcripts.bundle.registry import CapabilityStatus
from ..transcripts.bundle.store import BundleStore
from ..transcripts.errors import TranscriptError
from .context import app_context

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
    help="Free-text label for what this artefact is (e.g. obsidian-note, "
    "recording, teams-transcript).",
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
@click.pass_context
def resume_command(
    ctx: click.Context,
    bundle_path: Path,
    run_id: str | None,
    take_over: bool,
    as_json: bool,
) -> None:
    """Re-acquire a durable-state run's lease and dispatch its recorded
    next_action (M2).

    Never re-plans: the exact operation, input IDs, and config hash
    recorded when the run entered review_required/refused/failed are
    replayed as-is. v1 ships no real transform executors -- resuming a
    run whose next_action names an unimplemented operation kind is a
    safe, explicit no-op: the error names the missing kind, and the run
    is left exactly where it was, lease released, ready to resume again
    once that transform exists. If the executor itself raises, the run
    moves to `failed` (durable, resumable) with the exception recorded,
    rather than being left stuck `running` with the lease held forever.
    """
    executors = app_context(ctx).bundle_executors
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
