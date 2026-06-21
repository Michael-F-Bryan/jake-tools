from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import click

from jake_tools.daily_report.coordinator import (
    DailyReportCommandOptions,
    DailyReportCommandResult,
    run_daily_report_command,
)
from jake_tools.daily_report.stages import HermesDailyReportStages
from jake_tools.hermes import Hermes


def _parse_date(_ctx: click.Context, _param: click.Parameter, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise click.BadParameter("must be YYYY-MM-DD") from error


@click.command("daily-report")
@click.option(
    "--date",
    "target_date",
    required=True,
    callback=_parse_date,
    metavar="YYYY-MM-DD",
    help="Target local date for the report.",
)
@click.option(
    "--provider",
    default="openrouter",
    show_default=True,
    help="LLM provider for lane workers.",
)
@click.option(
    "--judgement-model",
    default="openrouter/auto",
    show_default=True,
    help="Model for judgement-heavy lanes.",
)
@click.option(
    "--evidence-model",
    default="openrouter/auto",
    show_default=True,
    help="Model for evidence-fed lanes.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit a machine-readable JSON summary only.",
)
def daily_report(
    target_date: date,
    provider: str,
    judgement_model: str,
    evidence_model: str,
    as_json: bool,
) -> None:
    """Run the deterministic daily-report coordinator."""

    hermes = Hermes()
    stages = HermesDailyReportStages(hermes)
    result = run_daily_report_command(
        command_options=DailyReportCommandOptions(
            target_date=target_date,
            provider=provider,
            judgement_model=judgement_model,
            evidence_model=evidence_model,
            base_dir=Path.cwd() / "_working",
        ),
        stages=stages,
    )
    _emit_result(result, as_json=as_json)
    if result.status == "fail":
        raise click.exceptions.Exit(1)


def _emit_result(result: DailyReportCommandResult, *, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(result.to_json(), sort_keys=True))
        return

    click.echo(f"status: {result.status}")
    click.echo(f"report: {result.report_path}")
    click.echo(f"summary: {result.summary_path}")
    if result.failed_lanes:
        click.echo("failed lanes:")
        for lane in result.failed_lanes:
            click.echo(f"- {lane}")
