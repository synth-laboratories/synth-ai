"""Formatting helpers for experiment queue CLI output."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Iterable

from rich.console import Console
from rich.table import Table

from .models import Experiment, ExperimentJob, Trial


def _relative_time(dt: datetime | None) -> str:
    if not dt:
        return "-"
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = now - dt
    seconds = int(delta.total_seconds())
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days > 0:
        return f"{days}d {hours}h ago"
    if hours > 0:
        return f"{hours}h {minutes}m ago"
    if minutes > 0:
        return f"{minutes}m {seconds}s ago"
    return f"{seconds}s ago"


def experiments_table(title: str, experiments: Iterable[Experiment]) -> Table:
    table = Table(title=title)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Status", style="magenta")
    table.add_column("Jobs", justify="right")
    table.add_column("Trials", justify="right")
    table.add_column("Started/Completed", justify="right")

    for experiment in experiments:
        job_count = len(experiment.jobs)
        trial_count = len(experiment.trials)
        timestamp = experiment.started_at or experiment.completed_at or experiment.created_at
        table.add_row(
            experiment.experiment_id,
            experiment.name,
            getattr(experiment.status, "value", experiment.status),
            str(job_count),
            str(trial_count),
            _relative_time(timestamp),
        )
    return table


def experiment_jobs_table(jobs: Iterable[ExperimentJob]) -> Table:
    table = Table(title="Jobs")
    table.add_column("Job ID", style="cyan", no_wrap=True)
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Started", justify="right")
    table.add_column("Completed", justify="right")
    table.add_column("Error")

    for job in jobs:
        table.add_row(
            job.job_id,
            getattr(job.job_type, "value", job.job_type),
            getattr(job.status, "value", job.status),
            _relative_time(job.started_at),
            _relative_time(job.completed_at),
            (job.error or "")[:80],
        )
    return table


def experiment_trials_table(trials: Iterable[Trial]) -> Table:
    table = Table(title="Trials")
    table.add_column("Trial #", justify="right")
    table.add_column("System")
    table.add_column("Score", justify="right")
    table.add_column("Status")

    for trial in trials:
        table.add_row(
            str(trial.trial_number or "-"),
            trial.system_name or "-",
            f"{trial.aggregate_score:.4f}" if trial.aggregate_score is not None else "-",
            getattr(trial.status, "value", trial.status),
        )
    return table


def render_dashboard(console: Console, live: list[Experiment], recent: list[Experiment]) -> None:
    console.rule("[bold]Experiment Queue")
    if live:
        console.print(experiments_table("Live Experiments", live))
    else:
        console.print("No live experiments.")
    if recent:
        console.print(experiments_table("Recent Experiments", recent))
