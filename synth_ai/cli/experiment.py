"""Experiment command group."""

import json
import time

import click
from rich.console import Console

STATUS_CHOICES = ["queued", "running", "completed", "failed", "canceled"]


@click.group()
def experiment() -> None:
    """Manage experiment queue submissions."""


@experiment.command()
@click.argument("request", type=str)
@click.option("--inline", is_flag=True, help="Treat REQUEST argument as inline JSON payload.")
def submit(request: str, inline: bool) -> None:
    """Submit a new experiment from JSON request file."""
    from synth_ai.cli.utils import experiment_queue as queue_utils
    from synth_ai.core.experiment_queue.schemas import ExperimentSummary

    payload = queue_utils.load_request_payload(request, inline=inline)
    created = queue_utils.create_experiment(payload)
    summary = ExperimentSummary.from_experiment(created)
    click.echo(
        f"Enqueued experiment {summary.experiment_id} ({summary.name}) with {summary.job_count} jobs."
    )


@experiment.command()
@click.option(
    "--status",
    "status_filter",
    multiple=True,
    type=click.Choice(STATUS_CHOICES),
    help="Status filter.",
)
@click.option("--limit", default=20, show_default=True, help="Max experiments to list.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON output.")
def list(status_filter: tuple[str, ...], limit: int, as_json: bool) -> None:
    """List experiments."""
    from synth_ai.cli.utils import experiment_queue as queue_utils
    from synth_ai.core.experiment_queue.schemas import ExperimentSummary

    experiments = queue_utils.list_experiments(
        status=queue_utils.parse_statuses(status_filter),
        limit=limit,
        include_live=True,
    )
    if as_json:
        payload = [
            ExperimentSummary.from_experiment(exp).model_dump(mode="json") for exp in experiments
        ]
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    console = Console()
    queue_utils.render_dashboard(console, experiments, [])


@experiment.command()
@click.argument("experiment_id")
@click.option("--watch", is_flag=True, help="Continuously refresh status view.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON payload.")
@click.option("--interval", default=2.0, show_default=True, help="Refresh interval for --watch.")
def status(experiment_id: str, watch: bool, as_json: bool, interval: float) -> None:
    """Show detailed experiment status."""
    from synth_ai.cli.utils import experiment_queue as queue_utils

    if as_json:
        click.echo(queue_utils.experiment_detail_json(experiment_id))
        return

    if watch:
        console = Console()
        while True:
            queue_utils.experiment_detail_console(experiment_id, console=console, clear=True)
            time.sleep(max(interval, 0.5))
    else:
        queue_utils.experiment_detail_console(experiment_id)


@experiment.command()
@click.argument("experiment_id")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON payload.")
def results(experiment_id: str, as_json: bool) -> None:
    """Show aggregate results for an experiment."""
    from synth_ai.cli.utils import experiment_queue as queue_utils

    if as_json:
        click.echo(queue_utils.experiment_detail_json(experiment_id))
    else:
        queue_utils.experiment_detail_console(experiment_id)


@experiment.command()
@click.argument("experiment_id")
def cancel(experiment_id: str) -> None:
    """Cancel an experiment and revoke in-flight jobs."""
    from synth_ai.cli.utils import experiment_queue as queue_utils

    experiment = queue_utils.cancel_experiment(experiment_id)
    if not experiment:
        raise click.ClickException(f"Experiment {experiment_id} not found.")
    click.echo(f"Experiment {experiment_id} canceled.")
