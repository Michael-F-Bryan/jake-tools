from __future__ import annotations

import json

import click

from ..transcripts.models import (
    example_for_public_transcript_artifact,
    list_public_transcript_artifact_models,
    resolve_public_transcript_artifact_model,
)


def _public_artifact_model_names() -> list[str]:
    return [model.__name__ for model in list_public_transcript_artifact_models()]


MODEL_NAME_ARGUMENT = click.Choice(_public_artifact_model_names(), case_sensitive=True)


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


@transcript.group(
    help="Placeholder for source primitives (Phase 2+). Use `transcript schema` today.",
)
def source() -> None:
    pass


@transcript.group(
    help="Placeholder for parse primitives (Phase 2+). Use `transcript schema` today.",
)
def parse() -> None:
    pass


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
