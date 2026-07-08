from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import click

from ..ai_usage import AITotals
from ..ai_watch.collect import run_collect
from ..ai_watch.config import resolve_discord_target
from ..ai_watch.curate import run_curate
from ..ai_watch.delivery import build_discord_payload, run_delivery
from ..ai_watch.digest import run_digest
from ..ai_watch.fetch import run_fetch
from ..ai_watch.manifest import write_manifest
from ..ai_watch.models import AiWatchCommandOptions, RunStatus
from ..ai_watch.obsidian import run_obsidian_sync
from ..ai_watch.paths import AiWatchPaths
from ..ai_watch.runner import run_ai_watch_command
from ..ai_watch.scout import run_scout
from ..ai_watch.stages import HermesAiWatchStages
from ..ai_watch.state import SeenIndex
from ..ai_watch.tuning import apply_surface_policy
from ..ai_watch.web_tools import HermesWebTools
from ..hermes import Hermes


def _parse_date(_ctx: click.Context, _param: click.Parameter, value: str) -> date:
    if value == "today":
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise click.BadParameter("must be YYYY-MM-DD or today") from error


def _common_options(func):
    options = [
        click.option(
            "--date",
            "target_date",
            default="today",
            callback=_parse_date,
            show_default=True,
            help="Run date (YYYY-MM-DD or today).",
        ),
        click.option(
            "--base-dir",
            type=click.Path(path_type=Path, file_okay=False),
            default=Path.cwd() / "_working",
            show_default=True,
            help="Working directory root.",
        ),
        click.option(
            "--dry-run", is_flag=True, help="Avoid vault writes and live delivery."
        ),
        click.option("--max-candidates", default=80, show_default=True, type=int),
        click.option("--calibration-only", is_flag=True),
        click.option("--json", "as_json", is_flag=True),
    ]
    for option in reversed(options):
        func = option(func)
    return func


@click.group("ai-watch")
def ai_watch() -> None:
    """Low-noise AI developments radar."""


@ai_watch.command()
@_common_options
def collect(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
) -> None:
    paths = AiWatchPaths.for_date(base_dir, target_date)
    state = SeenIndex(paths.state_root)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        max_candidates=max_candidates,
        calibration_only=calibration_only,
    )
    _run_stage(
        lambda: run_collect(
            options=options,
            paths=paths,
            state=state,
            web_tools=HermesWebTools(),
        ),
        as_json=as_json,
    )


@ai_watch.command()
@_common_options
def fetch(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
) -> None:
    paths = AiWatchPaths.for_date(base_dir, target_date)
    state = SeenIndex(paths.state_root)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        max_candidates=max_candidates,
        calibration_only=calibration_only,
    )
    _run_stage(
        lambda: run_fetch(
            options=options,
            paths=paths,
            state=state,
            web_tools=HermesWebTools(),
        ),
        as_json=as_json,
    )


@ai_watch.command()
@_common_options
@click.option("--scout-model", default="gpt-5.4-mini", show_default=True)
@click.option("--scout-provider", default="openai-codex", show_default=True)
def scout(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
    scout_model: str,
    scout_provider: str,
) -> None:
    paths = AiWatchPaths.for_date(base_dir, target_date)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        max_candidates=max_candidates,
        calibration_only=calibration_only,
        scout_model=scout_model,
        scout_provider=scout_provider,
    )
    _run_stage(
        lambda: run_scout(
            options=options,
            paths=paths,
            stages=HermesAiWatchStages(Hermes()),
        ),
        as_json=as_json,
    )


@ai_watch.command()
@_common_options
@click.option("--curator-model", default="gpt-5.5", show_default=True)
@click.option("--curator-provider", default="openai-codex", show_default=True)
@click.option("--force-candidate", default=None)
def curate(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
    curator_model: str,
    curator_provider: str,
    force_candidate: str | None,
) -> None:
    paths = AiWatchPaths.for_date(base_dir, target_date)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        max_candidates=max_candidates,
        calibration_only=calibration_only,
        curator_model=curator_model,
        curator_provider=curator_provider,
        force_candidate=force_candidate,
    )
    _run_stage(
        lambda: run_curate(
            options=options,
            paths=paths,
            stages=HermesAiWatchStages(Hermes()),
        ),
        as_json=as_json,
    )


@ai_watch.command("obsidian-sync")
@_common_options
@click.option(
    "--vault-path",
    type=click.Path(path_type=Path, exists=False, file_okay=False),
    default=Path("/Users/work/Documents/Vault"),
    show_default=True,
)
def obsidian_sync(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
    vault_path: Path,
) -> None:
    paths = AiWatchPaths.for_date(base_dir, target_date)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        max_candidates=max_candidates,
        calibration_only=calibration_only,
        vault_path=vault_path,
    )
    result = run_obsidian_sync(options=options, paths=paths)
    _emit_stage(result.__dict__, as_json=as_json)


@ai_watch.command()
@_common_options
def digest(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
) -> None:
    paths = AiWatchPaths.for_date(base_dir, target_date)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        max_candidates=max_candidates,
        calibration_only=calibration_only,
    )
    surfaced, speculative = run_digest(options=options, paths=paths)
    _emit_stage({"surfaced": surfaced, "speculative": speculative}, as_json=as_json)


@ai_watch.command()
@_common_options
@click.option("--target", "delivery_target", default="discord", show_default=True)
@click.option("--discord-target", default="")
def deliver(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
    delivery_target: str,
    discord_target: str,
) -> None:
    paths = AiWatchPaths.for_date(base_dir, target_date)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        max_candidates=max_candidates,
        calibration_only=calibration_only,
        discord_target=resolve_discord_target(discord_target),
    )
    result = run_delivery(options=options, paths=paths)
    _emit_stage({**result.__dict__, "target": delivery_target}, as_json=as_json)


@ai_watch.command("tune")
@_common_options
@click.option("--surface-limit", default=2, show_default=True, type=int)
@click.option("--max-article-age-days", default=90, show_default=True, type=int)
@click.option(
    "--vault-path",
    type=click.Path(path_type=Path),
    default=Path("/Users/work/Documents/Vault"),
)
@click.option(
    "--remove-unsurfaced-notes",
    is_flag=True,
    help="Delete generated Obsidian notes for demoted items when run markers match.",
)
def tune(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
    surface_limit: int,
    max_article_age_days: int,
    vault_path: Path,
    remove_unsurfaced_notes: bool,
) -> None:
    del max_candidates, calibration_only
    paths = AiWatchPaths.for_date(base_dir, target_date)
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        surface_limit=surface_limit,
        max_article_age_days=max_article_age_days,
        vault_path=vault_path,
    )
    result = apply_surface_policy(
        options=options,
        paths=paths,
        remove_notes=remove_unsurfaced_notes and not dry_run,
    )
    surfaced, speculative = run_digest(options=options, paths=paths)
    digest_text = paths.digest.read_text(encoding="utf-8")
    (paths.root / "delivery-payload.txt").write_text(
        build_discord_payload(digest_text) if surfaced else digest_text,
        encoding="utf-8",
    )
    summary = (
        AITotals.model_validate_json(paths.summary.read_text(encoding="utf-8"))
        if paths.summary.exists()
        else AITotals()
    )
    write_manifest(
        paths=paths,
        options=options,
        status=RunStatus.OK,
        surfaced_count=surfaced,
        speculative_count=speculative,
        failed_stages=[],
        summary=summary,
    )
    _emit_stage(
        {
            **result.__dict__,
            "digest_surfaced": surfaced,
            "digest_speculative": speculative,
        },
        as_json=as_json,
    )


@ai_watch.command()
@_common_options
@click.option("--scout-model", default="gpt-5.4-mini", show_default=True)
@click.option("--scout-provider", default="openai-codex", show_default=True)
@click.option("--curator-model", default="gpt-5.5", show_default=True)
@click.option("--curator-provider", default="openai-codex", show_default=True)
@click.option("--discord-target", default="")
@click.option(
    "--vault-path",
    type=click.Path(path_type=Path),
    default=Path("/Users/work/Documents/Vault"),
)
@click.option(
    "--surface-limit",
    default=2,
    show_default=True,
    type=int,
    help="Maximum main-digest items per run. Extra surfaced items become speculative.",
)
@click.option(
    "--max-article-age-days",
    default=90,
    show_default=True,
    type=int,
    help="Demote surfaced articles older than this many days.",
)
@click.option("--cost-cap-usd", type=float, default=None)
def run(
    target_date: date,
    base_dir: Path,
    dry_run: bool,
    max_candidates: int,
    calibration_only: bool,
    as_json: bool,
    scout_model: str,
    scout_provider: str,
    curator_model: str,
    curator_provider: str,
    discord_target: str,
    vault_path: Path,
    surface_limit: int,
    max_article_age_days: int,
    cost_cap_usd: float | None,
) -> None:
    options = AiWatchCommandOptions(
        target_date=target_date,
        base_dir=base_dir,
        dry_run=dry_run,
        max_candidates=max_candidates,
        calibration_only=calibration_only,
        scout_model=scout_model,
        scout_provider=scout_provider,
        curator_model=curator_model,
        curator_provider=curator_provider,
        discord_target=resolve_discord_target(discord_target),
        vault_path=vault_path,
        surface_limit=surface_limit,
        max_article_age_days=max_article_age_days,
        cost_cap_usd=cost_cap_usd,
    )
    result = run_ai_watch_command(options=options)
    if as_json:
        click.echo(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    else:
        click.echo(f"status: {result.status.value}")
        click.echo(f"root: {result.root}")
        click.echo(f"digest: {result.digest_path}")
        if result.failed_stages:
            click.echo("failed stages:")
            for stage in result.failed_stages:
                click.echo(f"- {stage}")
    if result.status == RunStatus.FAIL:
        raise click.exceptions.Exit(1)


@ai_watch.command()
@click.option("--since", default="7d", show_default=True)
@click.option(
    "--base-dir", type=click.Path(path_type=Path), default=Path.cwd() / "_working"
)
def audit(since: str, base_dir: Path) -> None:
    days = int(since.removesuffix("d"))
    cutoff = date.today() - timedelta(days=days)
    watch_root = base_dir / "ai-watch"
    if not watch_root.exists():
        click.echo("no runs found")
        return
    for run_dir in sorted(watch_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "state":
            continue
        try:
            run_date = date.fromisoformat(run_dir.name)
        except ValueError:
            continue
        if run_date < cutoff:
            continue
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            click.echo(manifest.read_text(encoding="utf-8"))


def _emit_stage(payload: dict, *, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, default=str))
        return
    for key, value in payload.items():
        click.echo(f"{key}: {value}")


def _run_stage(stage_fn, *, as_json: bool):
    try:
        result = stage_fn()
    except RuntimeError as error:
        raise click.exceptions.Exit(1) from error
    _emit_stage(result.__dict__, as_json=as_json)
    return result
