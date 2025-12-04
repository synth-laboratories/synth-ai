"""CLI commands for experiment queue management."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import click
from rich.console import Console

# Clear config cache if env vars are set (must happen before other imports)
if os.getenv("EXPERIMENT_QUEUE_DB_PATH") or os.getenv("EXPERIMENT_QUEUE_TRAIN_CMD"):
    from synth_ai.cli.local.experiment_queue import config as queue_config

    queue_config.reset_config_cache()

from synth_ai.cli.local.experiment_queue.models import ExperimentStatus
from synth_ai.cli.local.experiment_queue.schemas import (
    ExperimentJobSummary,
    ExperimentSubmitRequest,
    ExperimentSummary,
    TrialSummary,
)
from synth_ai.cli.local.experiment_queue.service import (
    cancel_experiment,
    collect_dashboard_data,
    create_experiment,
    fetch_experiment,
    list_experiments,
)
from synth_ai.cli.local.experiment_queue.status import (
    experiment_jobs_table,  # type: ignore[attr-defined]
    experiment_trials_table,  # type: ignore[attr-defined]
    render_dashboard,  # type: ignore[attr-defined]
)

STATUS_CHOICES = [status.value for status in ExperimentStatus]


def _load_request_payload(source: str, *, inline: bool = False) -> ExperimentSubmitRequest:
    if inline:
        data = json.loads(source)
    elif source == "-":
        data = json.load(sys.stdin)
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise click.ClickException(f"Request file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    return ExperimentSubmitRequest.model_validate(data)


def _parse_statuses(values: tuple[str, ...]) -> list[ExperimentStatus] | None:
    if not values:
        return None
    return [ExperimentStatus(value) for value in values]


@click.command("experiments")
@click.option(
    "--status",
    "status_filter",
    multiple=True,
    type=click.Choice(STATUS_CHOICES),
    help="Filter experiments by status.",
)
@click.option("--recent", default=5, show_default=True, help="Number of recent experiments to show.")
@click.option("--watch", is_flag=True, help="Continuously refresh the dashboard.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of tables.")
@click.option("--interval", default=2.0, show_default=True, help="Refresh interval for --watch.")
def experiments_cmd(
    status_filter: tuple[str, ...],
    recent: int,
    watch: bool,
    as_json: bool,
    interval: float,
) -> None:
    """Show live and recent experiment activity."""

    console = Console()

    def _snapshot() -> None:
        live, recent_data = collect_dashboard_data(
            status_filter=_parse_statuses(status_filter),
            recent_limit=recent,
        )
        if as_json:
            payload = {
                "live": [ExperimentSummary.from_experiment(exp).model_dump(mode="json") for exp in live],
                "recent": [ExperimentSummary.from_experiment(exp).model_dump(mode="json") for exp in recent_data],
            }
            click.echo(json.dumps(payload, indent=2, default=str))
        else:
            console.clear()
            render_dashboard(console, live, recent_data)

    if watch and not as_json:
        while True:
            _snapshot()
            time.sleep(max(interval, 0.5))
    else:
        _snapshot()


@click.group("experiment")
def experiment_group() -> None:
    """Manage experiment queue submissions."""


@experiment_group.command("submit")
@click.argument("request", type=str)
@click.option("--inline", is_flag=True, help="Treat REQUEST argument as inline JSON payload.")
def experiment_submit(request: str, inline: bool) -> None:
    """Submit a new experiment from JSON request file."""

    payload = _load_request_payload(request, inline=inline)
    experiment = create_experiment(payload)
    summary = ExperimentSummary.from_experiment(experiment)
    click.echo(f"Enqueued experiment {summary.experiment_id} ({summary.name}) with {summary.job_count} jobs.")


@experiment_group.command("list")
@click.option(
    "--status",
    "status_filter",
    multiple=True,
    type=click.Choice(STATUS_CHOICES),
    help="Status filter.",
)
@click.option("--limit", default=20, show_default=True, help="Max experiments to list.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
def experiment_list(
    status_filter: tuple[str, ...],
    limit: int,
    as_json: bool,
) -> None:
    """List experiments."""
    experiments = list_experiments(
        status=_parse_statuses(status_filter),
        limit=limit,
        include_live=True,
    )
    if as_json:
        payload = [ExperimentSummary.from_experiment(exp).model_dump(mode="json") for exp in experiments]
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    console = Console()
    render_dashboard(console, experiments, [])


def _experiment_detail_json(experiment_id: str) -> str:
    experiment = fetch_experiment(experiment_id)
    if not experiment:
        raise click.ClickException(f"Experiment {experiment_id} not found.")
    payload = ExperimentSummary.from_experiment(experiment).model_dump(mode="json")
    payload["jobs"] = [ExperimentJobSummary.from_job(job).model_dump(mode="json") for job in experiment.jobs]
    payload["trials"] = [TrialSummary.from_trial(trial).model_dump(mode="json") for trial in experiment.trials]
    return json.dumps(payload, indent=2, default=str)


def _experiment_detail_console(experiment_id: str, *, console: Console | None = None, clear: bool = False) -> None:
    experiment = fetch_experiment(experiment_id)
    if not experiment:
        raise click.ClickException(f"Experiment {experiment_id} not found.")
    console = console or Console()
    if clear:
        console.clear()
    summary = ExperimentSummary.from_experiment(experiment)
    console.rule(f"[bold]Experiment {summary.experiment_id} — {summary.name}")
    console.print(f"Status: {summary.status.value}")
    console.print(f"Description: {summary.description or '-'}")
    metadata_blob = experiment.metadata_json if isinstance(experiment.metadata_json, dict) else {}
    aggregate = metadata_blob.get("aggregate", {})
    if aggregate:
        console.print(f"Best Score: {aggregate.get('best_score')}")
        console.print(f"Baseline: {aggregate.get('baseline_score')}")
        console.print(f"Rollouts: {aggregate.get('total_rollouts')}")
        console.print(f"Total Time: {aggregate.get('total_time')}")
    console.print(experiment_jobs_table(experiment.jobs))
    if experiment.trials:
        console.print(experiment_trials_table(experiment.trials))
    else:
        console.print("No trials recorded yet.")


@experiment_group.command("status")
@click.argument("experiment_id")
@click.option("--watch", is_flag=True, help="Continuously refresh status view.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON payload.")
@click.option("--interval", default=2.0, show_default=True, help="Refresh interval for --watch.")
def experiment_status(experiment_id: str, watch: bool, as_json: bool, interval: float) -> None:
    """Show detailed experiment status."""
    if as_json:
        click.echo(_experiment_detail_json(experiment_id))
        return

    if watch:
        console = Console()
        while True:
            _experiment_detail_console(experiment_id, console=console, clear=True)
            time.sleep(max(interval, 0.5))
    else:
        _experiment_detail_console(experiment_id)


@experiment_group.command("results")
@click.argument("experiment_id")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON payload.")
def experiment_results(experiment_id: str, as_json: bool) -> None:
    """Show aggregate results for an experiment."""
    if as_json:
        click.echo(_experiment_detail_json(experiment_id))
    else:
        _experiment_detail_console(experiment_id)


@experiment_group.command("cancel")
@click.argument("experiment_id")
def experiment_cancel(experiment_id: str) -> None:
    """Cancel an experiment and revoke in-flight jobs."""
    experiment = cancel_experiment(experiment_id)
    if not experiment:
        raise click.ClickException(f"Experiment {experiment_id} not found.")
    click.echo(f"Experiment {experiment_id} canceled.")


def register(cli: click.Group) -> None:
    """Register commands on the main CLI."""
    cli.add_command(experiments_cmd)
    cli.add_command(experiment_group)
