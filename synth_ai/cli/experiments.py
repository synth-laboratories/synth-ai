"""Experiments command."""

import json
import time

import click
from rich.console import Console

STATUS_CHOICES = ["queued", "running", "completed", "failed", "canceled"]


@click.command()
@click.option(
    "--status",
    "status_filter",
    multiple=True,
    type=click.Choice(STATUS_CHOICES),
    help="Filter experiments by status.",
)
@click.option(
    "--recent", default=5, show_default=True, help="Number of recent experiments to show."
)
@click.option("--watch", is_flag=True, help="Continuously refresh the dashboard.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of tables.")
@click.option("--interval", default=2.0, show_default=True, help="Refresh interval for --watch.")
def experiments(
    status_filter: tuple[str, ...],
    recent: int,
    watch: bool,
    as_json: bool,
    interval: float,
) -> None:
    """Show live and recent experiment activity."""
    console = Console()

    def _snapshot() -> None:
        from synth_ai.cli.utils import experiment_queue as queue_utils
        from synth_ai.core.experiment_queue.schemas import ExperimentSummary

        live, recent_data = queue_utils.collect_dashboard_data(
            status_filter=queue_utils.parse_statuses(status_filter),
            recent_limit=recent,
        )
        if as_json:
            payload = {
                "live": [
                    ExperimentSummary.from_experiment(exp).model_dump(mode="json") for exp in live
                ],
                "recent": [
                    ExperimentSummary.from_experiment(exp).model_dump(mode="json")
                    for exp in recent_data
                ],
            }
            click.echo(json.dumps(payload, indent=2, default=str))
        else:
            console.clear()
            queue_utils.render_dashboard(console, live, recent_data)

    if watch and not as_json:
        while True:
            _snapshot()
            time.sleep(max(interval, 0.5))
    else:
        _snapshot()
