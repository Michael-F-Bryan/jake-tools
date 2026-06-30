from __future__ import annotations

import json
from pathlib import Path

import click
from pydantic import BaseModel, ValidationError

from ..hermes import Hermes
from ..transcripts.models import (
    ChapterPlan,
    SourceArtifact,
    TranscriptArtifact,
    VerificationReport,
    example_for_public_transcript_artifact,
    list_public_transcript_artifact_models,
    resolve_public_transcript_artifact_model,
)
from ..transcripts.parse_primitives import (
    ParsePrimitiveError,
    parse_gemini_transcript,
    parse_scribe_transcript,
)
from ..transcripts.source_primitives import (
    SourcePrimitiveError,
    source_from_gemini_pdf,
    source_from_gemini_text,
    source_from_obsidian_note,
)
from ..transcripts.stage_primitives import (
    StagePrimitiveError,
    load_transcript_or_manifest,
    run_map_speakers_stage,
    run_minutes_stage,
    run_polish_stage,
    run_title_chapters_stage,
)
from ..transcripts.transform_primitives import (
    TransformPrimitiveError,
    draft_chapter_boundaries,
    merge_adjacent_turns,
    normalise_transcript_artifact,
    split_transcript_manifest,
    strip_source_boilerplate,
)
from ..transcripts.verify_primitives import (
    BOILERPLATE_CHECK_IDS,
    CHAPTERS_CHECK_IDS,
    NOTE_CHECK_IDS,
    TURNS_CHECK_IDS,
    read_source_text_for_verification,
    verify_boilerplate_text,
    verify_chapters,
    verify_note,
    verify_turns,
)
from .options import hermes


def _public_artifact_model_names() -> list[str]:
    return [model.__name__ for model in list_public_transcript_artifact_models()]


MODEL_NAME_ARGUMENT = click.Choice(_public_artifact_model_names(), case_sensitive=True)


def _emit_written_model(
    model: BaseModel,
    *,
    out_path: Path,
    as_json: bool,
) -> None:
    out_path.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if as_json:
        click.echo(json.dumps(model.model_dump(mode="json"), sort_keys=True))
        return
    click.echo(f"wrote {out_path}")


def _emit_verification_report(
    report: VerificationReport,
    *,
    out_path: Path | None,
    as_json: bool,
) -> None:
    if out_path is not None:
        out_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if as_json:
        click.echo(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    elif out_path is not None:
        click.echo(f"wrote {out_path}")
    else:
        click.echo(report.status)

    if report.failed_gate_ids:
        raise click.exceptions.Exit(1)


@click.group(
    help="Transcript primitive toolbox for schema discovery and future stages."
)
def transcript() -> None:
    pass


@transcript.group(help="Discover transcript artefact schemas.")
def schema() -> None:
    pass


@schema.command("list", help="List all public transcript artefact models.")
def schema_list() -> None:
    for model in list_public_transcript_artifact_models():
        click.echo(model.__name__)


@schema.command("show", help="Show a model schema.")
@click.argument("model", type=MODEL_NAME_ARGUMENT)
@click.option(
    "--format",
    "schema_format",
    type=click.Choice(["json-schema"], case_sensitive=True),
    default="json-schema",
    show_default=True,
    help="Output schema format.",
)
def schema_show(model: str, schema_format: str) -> None:
    model_type = resolve_public_transcript_artifact_model(model)
    if model_type is None:
        raise click.ClickException(f"unknown model: {model}")
    if schema_format != "json-schema":
        raise click.ClickException(f"unsupported format: {schema_format}")

    click.echo(json.dumps(model_type.model_json_schema(), sort_keys=True))


@schema.command("example", help="Emit a minimal valid model example.")
@click.argument("model", type=MODEL_NAME_ARGUMENT)
def schema_example(model: str) -> None:
    example = example_for_public_transcript_artifact(model)
    if example is None:
        raise click.ClickException(f"unknown model: {model}")
    click.echo(json.dumps(example.model_dump(mode="json"), sort_keys=True))


@transcript.group(help="Source adapters that emit SourceArtifact JSON.")
def source() -> None:
    pass


@source.command("obsidian-note")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write SourceArtifact JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit SourceArtifact JSON to stdout.",
)
@click.argument(
    "obsidian_note",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def source_obsidian_note(out_path: Path, as_json: bool, obsidian_note: Path) -> None:
    """
    Build a SourceArtifact from an Obsidian note.

    Input: Obsidian note markdown path with recording embeds.
    Output: SourceArtifact JSON at --out, optionally echoed via --json.
    Side effects: Reads note and referenced recordings; writes --out.
    """
    artifact = source_from_obsidian_note(obsidian_note)
    _emit_written_model(artifact, out_path=out_path, as_json=as_json)


@source.command("gemini-pdf")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write SourceArtifact JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit SourceArtifact JSON to stdout.",
)
@click.argument(
    "pdf_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def source_gemini_pdf(out_path: Path, as_json: bool, pdf_path: Path) -> None:
    """
    Build a SourceArtifact from Gemini-style PDF notes.

    Input: PDF note export path.
    Output: SourceArtifact JSON at --out and extracted text beside it.
    Side effects: Runs pdftotext, writes extracted text, writes --out.
    """
    try:
        artifact = source_from_gemini_pdf(pdf_path, source_output_path=out_path)
    except SourcePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(artifact, out_path=out_path, as_json=as_json)


@source.command("gemini-text")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write SourceArtifact JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit SourceArtifact JSON to stdout.",
)
@click.argument(
    "text_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def source_gemini_text(out_path: Path, as_json: bool, text_path: Path) -> None:
    """
    Build a SourceArtifact from extracted Gemini text.

    Input: Plain-text file containing Gemini transcript notes.
    Output: SourceArtifact JSON at --out, optionally echoed via --json.
    Side effects: Reads text input and writes --out.
    """
    artifact = source_from_gemini_text(text_path)
    _emit_written_model(artifact, out_path=out_path, as_json=as_json)


@transcript.group(help="Parse source formats into TranscriptArtifact JSON.")
def parse() -> None:
    pass


@parse.command("gemini")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write TranscriptArtifact JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit TranscriptArtifact JSON to stdout.",
)
@click.argument(
    "source_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def parse_gemini(out_path: Path, as_json: bool, source_artifact_path: Path) -> None:
    """
    Parse Gemini source text into canonical transcript turns.

    Input: SourceArtifact JSON that references Gemini text via raw_text_path.
    Output: TranscriptArtifact JSON at --out, optionally echoed via --json.
    Side effects: Reads source and text files; writes --out.
    """
    source_payload = SourceArtifact.model_validate_json(
        source_artifact_path.read_text(encoding="utf-8")
    )
    try:
        artifact = parse_gemini_transcript(source_payload)
    except ParsePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(artifact, out_path=out_path, as_json=as_json)


@parse.command("scribe")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write TranscriptArtifact JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit TranscriptArtifact JSON to stdout.",
)
@click.argument(
    "scribe_transcript_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def parse_scribe(out_path: Path, as_json: bool, scribe_transcript_path: Path) -> None:
    """
    Parse Scribe transcript JSON into canonical transcript turns.

    Input: Scribe JSON with a top-level segments list.
    Output: TranscriptArtifact JSON at --out, optionally echoed via --json.
    Side effects: Reads Scribe JSON and writes --out.
    """
    try:
        artifact = parse_scribe_transcript(scribe_transcript_path)
    except ParsePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(artifact, out_path=out_path, as_json=as_json)


@transcript.group(help="Deterministic transform primitives over transcript artefacts.")
def transform() -> None:
    pass


@transform.command("strip-boilerplate")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write cleaned SourceArtifact JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit SourceArtifact JSON to stdout.",
)
@click.argument(
    "source_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def transform_strip_boilerplate(
    out_path: Path, as_json: bool, source_artifact_path: Path
) -> None:
    """
    Remove deterministic Gemini boilerplate from source text.

    Input: SourceArtifact JSON with raw_text_path or source_path.
    Output: SourceArtifact JSON at --out, with raw_text_path set to cleaned text.
    Side effects: Reads source text and writes --out plus a sibling .clean.txt file.
    """
    source = SourceArtifact.model_validate_json(
        source_artifact_path.read_text(encoding="utf-8")
    )
    try:
        cleaned = strip_source_boilerplate(source, source_output_path=out_path)
    except TransformPrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(cleaned, out_path=out_path, as_json=as_json)


@transform.command("normalise")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write normalised TranscriptArtifact JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit TranscriptArtifact JSON to stdout.",
)
@click.argument(
    "transcript_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def transform_normalise(
    out_path: Path, as_json: bool, transcript_artifact_path: Path
) -> None:
    """
    Normalise transcript turn text and remove empty turns.

    Input: TranscriptArtifact JSON.
    Output: TranscriptArtifact JSON at --out, optionally echoed via --json.
    Side effects: Reads transcript input and writes --out.
    """
    artifact = TranscriptArtifact.model_validate_json(
        transcript_artifact_path.read_text(encoding="utf-8")
    )
    normalised = normalise_transcript_artifact(artifact)
    _emit_written_model(normalised, out_path=out_path, as_json=as_json)


@transform.command("merge-adjacent")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write merged TranscriptArtifact JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit TranscriptArtifact JSON to stdout.",
)
@click.option(
    "--max-gap",
    "max_gap",
    type=float,
    default=3.0,
    show_default=True,
    help="Maximum gap in seconds to merge adjacent turns by the same speaker.",
)
@click.argument(
    "transcript_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def transform_merge_adjacent(
    out_path: Path, as_json: bool, max_gap: float, transcript_artifact_path: Path
) -> None:
    """
    Merge adjacent turns when speaker matches and gap is within threshold.

    Input: TranscriptArtifact JSON plus --max-gap threshold in seconds.
    Output: TranscriptArtifact JSON at --out, optionally echoed via --json.
    Side effects: Reads transcript input and writes --out.
    """
    artifact = TranscriptArtifact.model_validate_json(
        transcript_artifact_path.read_text(encoding="utf-8")
    )
    try:
        merged = merge_adjacent_turns(artifact, max_gap_seconds=max_gap)
    except TransformPrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(merged, out_path=out_path, as_json=as_json)


@transform.command("split")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write chunk RunManifest JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit RunManifest JSON to stdout.",
)
@click.option(
    "--target-minutes",
    type=float,
    default=12.0,
    show_default=True,
    help="Target chunk duration in minutes.",
)
@click.argument(
    "transcript_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def transform_split(
    out_path: Path,
    as_json: bool,
    target_minutes: float,
    transcript_artifact_path: Path,
) -> None:
    """
    Split transcript turns into deterministic chunk manifest stages.

    Input: TranscriptArtifact JSON and --target-minutes chunk budget.
    Output: RunManifest JSON at --out, optionally echoed via --json.
    Side effects: Reads transcript input and writes --out.
    """
    artifact = TranscriptArtifact.model_validate_json(
        transcript_artifact_path.read_text(encoding="utf-8")
    )
    try:
        manifest = split_transcript_manifest(artifact, target_minutes=target_minutes)
    except TransformPrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(manifest, out_path=out_path, as_json=as_json)


@transform.command("chapter-boundaries")
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write draft ChapterPlan JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit ChapterPlan JSON to stdout.",
)
@click.option(
    "--window-minutes",
    type=float,
    default=10.0,
    show_default=True,
    help="Deterministic time window used for draft chapter boundaries.",
)
@click.argument(
    "transcript_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def transform_chapter_boundaries(
    out_path: Path,
    as_json: bool,
    window_minutes: float,
    transcript_artifact_path: Path,
) -> None:
    """
    Draft deterministic chapter boundaries from transcript turns.

    Input: TranscriptArtifact JSON and --window-minutes chapter window.
    Output: ChapterPlan JSON at --out, optionally echoed via --json.
    Side effects: Reads transcript input and writes --out.
    """
    artifact = TranscriptArtifact.model_validate_json(
        transcript_artifact_path.read_text(encoding="utf-8")
    )
    try:
        plan = draft_chapter_boundaries(artifact, window_minutes=window_minutes)
    except TransformPrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(plan, out_path=out_path, as_json=as_json)


@transcript.group(help="Structured LLM stage primitives over transcript artefacts.")
def stage() -> None:
    pass


@stage.command("polish")
@hermes
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write polished TranscriptArtifact JSON.",
)
@click.option(
    "--ledger-out",
    "ledger_out_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Optional path to write polish ledger JSON (defaults beside --out).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit polished TranscriptArtifact JSON to stdout.",
)
@click.option(
    "--max-attempts",
    default=2,
    show_default=True,
    type=int,
    help="Maximum structured-output attempts before failing.",
)
@click.argument(
    "input_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def stage_polish(
    hermes: Hermes,
    out_path: Path,
    ledger_out_path: Path | None,
    as_json: bool,
    max_attempts: int,
    input_path: Path,
) -> None:
    """
    Polish transcript turns with schema-validated LLM output.

    Input: TranscriptArtifact JSON or chunk RunManifest JSON.
    Output: Polished TranscriptArtifact at --out and polish ledger JSON.
    Side effects: Reads input artefacts and writes --out plus --ledger-out.
    """
    try:
        transcript = load_transcript_or_manifest(input_path)
        polished, ledger, _reply = run_polish_stage(
            hermes,
            transcript,
            max_attempts=max_attempts,
        )
    except (StagePrimitiveError, ValidationError) as exc:
        raise click.ClickException(str(exc)) from exc

    _emit_written_model(polished, out_path=out_path, as_json=as_json)
    resolved_ledger_path = ledger_out_path or out_path.with_suffix(".ledger.json")
    resolved_ledger_path.write_text(
        json.dumps(ledger.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not as_json:
        click.echo(f"wrote {resolved_ledger_path}")


@stage.command("map-speakers")
@hermes
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write SpeakerMapping JSON.",
)
@click.option(
    "--attendee",
    "attendees",
    multiple=True,
    help="Known attendee name; can be supplied multiple times.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit SpeakerMapping JSON to stdout.",
)
@click.option(
    "--max-attempts",
    default=2,
    show_default=True,
    type=int,
    help="Maximum structured-output attempts before failing.",
)
@click.argument(
    "transcript_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def stage_map_speakers(
    hermes: Hermes,
    out_path: Path,
    attendees: tuple[str, ...],
    as_json: bool,
    max_attempts: int,
    transcript_artifact_path: Path,
) -> None:
    """
    Infer structured speaker identities from transcript turns.

    Input: TranscriptArtifact JSON and optional --attendee hints.
    Output: SpeakerMapping JSON at --out, optionally echoed via --json.
    Side effects: Reads transcript input and writes --out.
    """
    transcript = TranscriptArtifact.model_validate_json(
        transcript_artifact_path.read_text(encoding="utf-8")
    )
    try:
        speaker_mapping, _reply = run_map_speakers_stage(
            hermes,
            transcript,
            attendees=list(attendees),
            max_attempts=max_attempts,
        )
    except StagePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(speaker_mapping, out_path=out_path, as_json=as_json)


@stage.command("title-chapters")
@hermes
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write ChapterPlan JSON.",
)
@click.option(
    "--chapters",
    "draft_chapters_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="Optional draft ChapterPlan JSON with boundaries to preserve.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit ChapterPlan JSON to stdout.",
)
@click.option(
    "--max-attempts",
    default=2,
    show_default=True,
    type=int,
    help="Maximum structured-output attempts before failing.",
)
@click.argument(
    "transcript_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def stage_title_chapters(
    hermes: Hermes,
    out_path: Path,
    draft_chapters_path: Path | None,
    as_json: bool,
    max_attempts: int,
    transcript_artifact_path: Path,
) -> None:
    """
    Generate structured chapter titles and summaries.

    Input: TranscriptArtifact JSON and optional draft ChapterPlan JSON.
    Output: ChapterPlan JSON at --out, optionally echoed via --json.
    Side effects: Reads transcript/draft chapter artefacts and writes --out.
    """
    transcript = TranscriptArtifact.model_validate_json(
        transcript_artifact_path.read_text(encoding="utf-8")
    )
    draft_plan = None
    if draft_chapters_path is not None:
        draft_plan = ChapterPlan.model_validate_json(
            draft_chapters_path.read_text(encoding="utf-8")
        )
    try:
        chapter_plan, _reply = run_title_chapters_stage(
            hermes,
            transcript,
            draft_plan=draft_plan,
            max_attempts=max_attempts,
        )
    except StagePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(chapter_plan, out_path=out_path, as_json=as_json)


@stage.command("minutes")
@hermes
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write MeetingMinutes JSON.",
)
@click.option(
    "--chapters",
    "chapters_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="Optional ChapterPlan JSON to guide minutes structure.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit MeetingMinutes JSON to stdout.",
)
@click.option(
    "--max-attempts",
    default=2,
    show_default=True,
    type=int,
    help="Maximum structured-output attempts before failing.",
)
@click.argument(
    "transcript_artifact_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def stage_minutes(
    hermes: Hermes,
    out_path: Path,
    chapters_path: Path | None,
    as_json: bool,
    max_attempts: int,
    transcript_artifact_path: Path,
) -> None:
    """
    Produce structured meeting minutes from transcript turns.

    Input: TranscriptArtifact JSON and optional ChapterPlan JSON.
    Output: MeetingMinutes JSON at --out, optionally echoed via --json.
    Side effects: Reads transcript/chapter artefacts and writes --out.
    """
    transcript = TranscriptArtifact.model_validate_json(
        transcript_artifact_path.read_text(encoding="utf-8")
    )
    chapters = None
    if chapters_path is not None:
        chapters = ChapterPlan.model_validate_json(
            chapters_path.read_text(encoding="utf-8")
        )
    try:
        minutes, _reply = run_minutes_stage(
            hermes,
            transcript,
            chapters=chapters,
            max_attempts=max_attempts,
        )
    except StagePrimitiveError as exc:
        raise click.ClickException(str(exc)) from exc
    _emit_written_model(minutes, out_path=out_path, as_json=as_json)


@transcript.group(
    help="Placeholder for rendering primitives (Phase 5+). Use `transcript schema` today.",
)
def render() -> None:
    pass


@transcript.group(
    help="Placeholder for note primitives (Phase 5+). Use `transcript schema` today.",
)
def note() -> None:
    pass


@transcript.group(help="Verification primitives with stable check IDs.")
def verify() -> None:
    pass


@verify.command(
    "boilerplate", help=f"Stable checks: {', '.join(BOILERPLATE_CHECK_IDS)}"
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Optional path to write VerificationReport JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit VerificationReport JSON to stdout.",
)
@click.argument(
    "source_or_text_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def verify_boilerplate_command(
    out_path: Path | None, as_json: bool, source_or_text_path: Path
) -> None:
    """
    Verify source text does not contain transcript boilerplate artefacts.

    Input: SourceArtifact JSON or plain text path.
    Output: VerificationReport via --json and optionally written with --out.
    Side effects: Reads input path and writes --out when provided.
    Stable check IDs: boilerplate.no-operational-chatter, boilerplate.no-markdown-fences.
    """
    raw_input = source_or_text_path.read_text(encoding="utf-8")
    try:
        source = SourceArtifact.model_validate_json(raw_input)
    except ValidationError:
        report = verify_boilerplate_text(raw_input, affected_path=source_or_text_path)
    else:
        source_text_path, source_text = read_source_text_for_verification(source)
        report = verify_boilerplate_text(source_text, affected_path=source_text_path)
    _emit_verification_report(report, out_path=out_path, as_json=as_json)


@verify.command("turns", help=f"Stable checks: {', '.join(TURNS_CHECK_IDS)}")
@click.option(
    "--out",
    "out_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Optional path to write VerificationReport JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit VerificationReport JSON to stdout.",
)
@click.argument(
    "before_transcript_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
@click.argument(
    "after_transcript_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def verify_turns_command(
    out_path: Path | None,
    as_json: bool,
    before_transcript_path: Path,
    after_transcript_path: Path,
) -> None:
    """
    Compare transcript artefacts before and after a deterministic transform.

    Input: BEFORE and AFTER TranscriptArtifact JSON paths.
    Output: VerificationReport via --json and optionally written with --out.
    Side effects: Reads both transcript artefacts and writes --out when provided.
    Stable check IDs: turns.non-empty, turns.monotonic-order, turns.coverage-preserved, turns.speakers-preserved.
    """
    before = TranscriptArtifact.model_validate_json(
        before_transcript_path.read_text(encoding="utf-8")
    )
    after = TranscriptArtifact.model_validate_json(
        after_transcript_path.read_text(encoding="utf-8")
    )
    report = verify_turns(
        before,
        after,
        affected_paths=[before_transcript_path, after_transcript_path],
    )
    _emit_verification_report(report, out_path=out_path, as_json=as_json)


@verify.command("chapters", help=f"Stable checks: {', '.join(CHAPTERS_CHECK_IDS)}")
@click.option(
    "--out",
    "out_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Optional path to write VerificationReport JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit VerificationReport JSON to stdout.",
)
@click.option(
    "--turns",
    "turns_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="Optional TranscriptArtifact JSON to validate chapter span coverage.",
)
@click.argument(
    "chapter_plan_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def verify_chapters_command(
    out_path: Path | None,
    as_json: bool,
    turns_path: Path | None,
    chapter_plan_path: Path,
) -> None:
    """
    Verify chapter boundary structure and optional transcript span coverage.

    Input: ChapterPlan JSON and optional --turns TranscriptArtifact JSON.
    Output: VerificationReport via --json and optionally written with --out.
    Side effects: Reads chapter/turn artefacts and writes --out when provided.
    Stable check IDs: chapters.non-empty, chapters.monotonic-order, chapters.non-overlapping, chapters.covers-transcript-span.
    """
    chapter_plan = ChapterPlan.model_validate_json(
        chapter_plan_path.read_text(encoding="utf-8")
    )
    transcript = None
    affected_paths = [chapter_plan_path]
    if turns_path is not None:
        transcript = TranscriptArtifact.model_validate_json(
            turns_path.read_text(encoding="utf-8")
        )
        affected_paths.append(turns_path)
    report = verify_chapters(
        chapter_plan,
        transcript=transcript,
        affected_paths=affected_paths,
    )
    _emit_verification_report(report, out_path=out_path, as_json=as_json)


@verify.command("note", help=f"Stable checks: {', '.join(NOTE_CHECK_IDS)}")
@click.option(
    "--out",
    "out_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Optional path to write VerificationReport JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit VerificationReport JSON to stdout.",
)
@click.option(
    "--source",
    "source_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="Optional SourceArtifact JSON; used to include source path in affected artefacts.",
)
@click.option(
    "--chapters",
    "chapters_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="Optional ChapterPlan JSON to enforce chapter heading count.",
)
@click.argument(
    "note_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
)
def verify_note_command(
    out_path: Path | None,
    as_json: bool,
    source_path: Path | None,
    chapters_path: Path | None,
    note_path: Path,
) -> None:
    """
    Verify rendered note structure and note-level transcript hygiene checks.

    Input: Note markdown path with optional --source and --chapters artefacts.
    Output: VerificationReport via --json and optionally written with --out.
    Side effects: Reads note and optional artefacts; writes --out when provided.
    Stable check IDs: note.has-meeting-notes, note.has-chapters, note.has-transcript, note.chapter-heading-count, note.no-operational-chatter.
    """
    expected_chapter_count = None
    affected_paths = [note_path]
    if source_path is not None:
        affected_paths.append(source_path)
    if chapters_path is not None:
        chapter_plan = ChapterPlan.model_validate_json(
            chapters_path.read_text(encoding="utf-8")
        )
        expected_chapter_count = len(chapter_plan.chapters)
        affected_paths.append(chapters_path)

    report = verify_note(
        note_path.read_text(encoding="utf-8"),
        expected_chapter_count=expected_chapter_count,
        affected_path=note_path,
    )
    report = report.model_copy(update={"affected_artifact_paths": affected_paths})
    _emit_verification_report(report, out_path=out_path, as_json=as_json)


@transcript.group(
    help="Placeholder for recipe primitives (Phase 6+). Use `transcript schema` today.",
)
def recipe() -> None:
    pass
