"""CLI helpers for experiment queue commands."""

import json
import os
import sys
from pathlib import Path

import click
from rich.console import Console

STATUS_CHOICES = ["queued", "running", "completed", "failed", "canceled"]


def _reset_queue_config_cache_if_needed() -> None:
    if os.getenv("EXPERIMENT_QUEUE_DB_PATH") or os.getenv("EXPERIMENT_QUEUE_TRAIN_CMD"):
        from synth_ai.core.experiment_queue import config as queue_config

        queue_config.reset_config_cache()


def load_request_payload(source: str, *, inline: bool = False):
    from synth_ai.core.experiment_queue.schemas import ExperimentSubmitRequest

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


def parse_statuses(values: tuple[str, ...]) -> list[str] | None:
    if not values:
        return None
    return list(values)


def experiment_detail_json(experiment_id: str) -> str:
    _reset_queue_config_cache_if_needed()
    from synth_ai.core.experiment_queue.schemas import (
        ExperimentJobSummary,
        ExperimentSummary,
        TrialSummary,
    )
    from synth_ai.core.experiment_queue.service import fetch_experiment

    experiment = fetch_experiment(experiment_id)
    if not experiment:
        raise click.ClickException(f"Experiment {experiment_id} not found.")
    payload = ExperimentSummary.from_experiment(experiment).model_dump(mode="json")
    payload["jobs"] = [
        ExperimentJobSummary.from_job(job).model_dump(mode="json") for job in experiment.jobs
    ]
    payload["trials"] = [
        TrialSummary.from_trial(trial).model_dump(mode="json") for trial in experiment.trials
    ]
    return json.dumps(payload, indent=2, default=str)


def experiment_detail_console(
    experiment_id: str,
    *,
    console: Console | None = None,
    clear: bool = False,
) -> None:
    _reset_queue_config_cache_if_needed()
    from synth_ai.core.experiment_queue.schemas import ExperimentSummary
    from synth_ai.core.experiment_queue.service import fetch_experiment
    from synth_ai.core.experiment_queue.status import (
        experiment_jobs_table,  # type: ignore[attr-defined]
        experiment_trials_table,  # type: ignore[attr-defined]
    )

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


def _load_service():
    _reset_queue_config_cache_if_needed()
    from synth_ai.core.experiment_queue import service

    return service


def _load_status():
    _reset_queue_config_cache_if_needed()
    from synth_ai.core.experiment_queue import status

    return status


def create_experiment(payload):
    return _load_service().create_experiment(payload)


def fetch_experiment(experiment_id: str):
    return _load_service().fetch_experiment(experiment_id)


def list_experiments(*, status=None, limit: int = 20, include_live: bool = True):
    return _load_service().list_experiments(status=status, limit=limit, include_live=include_live)


def cancel_experiment(experiment_id: str):
    return _load_service().cancel_experiment(experiment_id)


def collect_dashboard_data(*, status_filter=None, recent_limit: int = 5):
    return _load_service().collect_dashboard_data(
        status_filter=status_filter, recent_limit=recent_limit
    )


def render_dashboard(console: Console, live, recent):
    return _load_status().render_dashboard(console, live, recent)


__all__ = [
    "STATUS_CHOICES",
    "cancel_experiment",
    "collect_dashboard_data",
    "create_experiment",
    "experiment_detail_console",
    "experiment_detail_json",
    "fetch_experiment",
    "list_experiments",
    "load_request_payload",
    "parse_statuses",
    "render_dashboard",
]
