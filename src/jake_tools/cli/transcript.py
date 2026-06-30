from __future__ import annotations

import json
from pathlib import Path

import click

from ..transcripts.models import (
    SourceArtifact,
    TranscriptArtifact,
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


def _public_artifact_model_names() -> list[str]:
    return [model.__name__ for model in list_public_transcript_artifact_models()]


MODEL_NAME_ARGUMENT = click.Choice(_public_artifact_model_names(), case_sensitive=True)


def _emit_written_artifact(
    artifact: SourceArtifact | TranscriptArtifact,
    *,
    out_path: Path,
    as_json: bool,
) -> None:
    out_path.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if as_json:
        click.echo(json.dumps(artifact.model_dump(mode="json"), sort_keys=True))
        return
    click.echo(f"wrote {out_path}")


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
    _emit_written_artifact(artifact, out_path=out_path, as_json=as_json)


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
    _emit_written_artifact(artifact, out_path=out_path, as_json=as_json)


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
    _emit_written_artifact(artifact, out_path=out_path, as_json=as_json)


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
    _emit_written_artifact(artifact, out_path=out_path, as_json=as_json)


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
    _emit_written_artifact(artifact, out_path=out_path, as_json=as_json)


@transcript.group(
    help=(
        "Placeholder for deterministic transform primitives (Phase 3+). "
        "Use `transcript schema` today."
    ),
)
def transform() -> None:
    pass


@transcript.group(
    help="Placeholder for LLM stage primitives (Phase 4+). Use `transcript schema` today.",
)
def stage() -> None:
    pass


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


@transcript.group(
    help="Placeholder for verification primitives (Phase 3+). Use `transcript schema` today.",
)
def verify() -> None:
    pass


@transcript.group(
    help="Placeholder for recipe primitives (Phase 6+). Use `transcript schema` today.",
)
def recipe() -> None:
    pass
