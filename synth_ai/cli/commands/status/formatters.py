"""Rich-based formatting helpers for status commands."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _format_timestamp(value: Any) -> str:
    if value in (None, "", 0):
        return ""
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value)).isoformat()
        except Exception:
            return str(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return value
    return str(value)


def print_json(data: Any) -> None:
    console.print_json(data=data)


def jobs_table(jobs: Iterable[dict[str, Any]]) -> Table:
    table = Table(title="Training Jobs", box=box.SIMPLE, header_style="bold")
    table.add_column("ID", style="cyan", overflow="fold")
    table.add_column("Type", style="magenta")
    table.add_column("Status")
    table.add_column("Created", style="green")
    table.add_column("Updated", style="green")
    table.add_column("Model", style="yellow", overflow="fold")
    for job in jobs:
        status = job.get("status", "unknown")
        status_color = {
            "running": "green",
            "queued": "cyan",
            "succeeded": "bright_green",
            "failed": "red",
            "cancelled": "yellow",
        }.get(status, "white")
        table.add_row(
            str(job.get("job_id") or job.get("id", "")),
            str(job.get("training_type") or job.get("type", "")),
            f"[{status_color}]{status}[/{status_color}]",
            _format_timestamp(job.get("created_at")),
            _format_timestamp(job.get("updated_at")),
            str(job.get("model_id") or job.get("model", "")),
        )
    return table


def job_panel(job: dict[str, Any]) -> Panel:
    lines = [f"[bold cyan]Job[/bold cyan] {job.get('job_id') or job.get('id')}"]
    if job.get("name"):
        lines.append(f"Name: {job['name']}")
    lines.append(f"Type: {job.get('training_type', job.get('type', ''))}")
    lines.append(f"Status: {job.get('status', 'unknown')}")
    if job.get("model_id"):
        lines.append(f"Model: {job['model_id']}")
    if job.get("base_model"):
        lines.append(f"Base Model: {job['base_model']}")
    lines.append(f"Created: {_format_timestamp(job.get('created_at'))}")
    lines.append(f"Updated: {_format_timestamp(job.get('updated_at'))}")
    if config := job.get("config"):
        lines.append("")
        lines.append(f"[dim]{json.dumps(config, indent=2, sort_keys=True)}[/dim]")
    return Panel("\n".join(lines), title="Job Details", border_style="cyan")


def runs_table(runs: Iterable[dict[str, Any]]) -> Table:
    table = Table(title="Job Runs", box=box.SIMPLE, header_style="bold")
    table.add_column("Run #", justify="right")
    table.add_column("Engine")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Started")
    table.add_column("Ended")
    table.add_column("Duration", justify="right")
    for run in runs:
        table.add_row(
            str(run.get("run_number") or run.get("id", "")),
            str(run.get("engine", "")),
            str(run.get("status", "unknown")),
            _format_timestamp(run.get("created_at")),
            _format_timestamp(run.get("started_at")),
            _format_timestamp(run.get("ended_at")),
            str(run.get("duration_seconds") or run.get("duration", "")),
        )
    return table


def events_panel(events: Iterable[dict[str, Any]]) -> Panel:
    rendered = []
    for event in events:
        ts = _format_timestamp(event.get("timestamp") or event.get("created_at"))
        level = event.get("level") or event.get("severity", "info")
        message = event.get("message") or event.get("detail") or ""
        rendered.append(f"[dim]{ts}[/dim] [{level}] {message}")
    if not rendered:
        rendered.append("[dim]No events found.[/dim]")
    return Panel("\n".join(rendered), title="Job Events", border_style="green")


def metrics_table(metrics: dict[str, Any]) -> Table:
    table = Table(title="Job Metrics", box=box.SIMPLE, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        if isinstance(value, dict):
            table.add_row(key, Text(json.dumps(value), overflow="fold"))
        else:
            table.add_row(key, str(value))
    return table


def files_table(files: Iterable[dict[str, Any]]) -> Table:
    table = Table(title="Training Files", box=box.SIMPLE, header_style="bold")
    table.add_column("ID", overflow="fold")
    table.add_column("Purpose")
    table.add_column("Size", justify="right")
    table.add_column("Created")
    table.add_column("Filename", overflow="fold")
    for file in files:
        table.add_row(
            str(file.get("file_id") or file.get("id", "")),
            str(file.get("purpose", "")),
            str(file.get("bytes", "")),
            _format_timestamp(file.get("created_at")),
            str(file.get("filename", "")),
        )
    return table


def models_table(models: Iterable[dict[str, Any]]) -> Table:
    table = Table(title="Fine-tuned Models", box=box.SIMPLE, header_style="bold")
    table.add_column("ID", overflow="fold")
    table.add_column("Base")
    table.add_column("Created")
    table.add_column("Owner")
    table.add_column("Status")
    for model in models:
        table.add_row(
            str(model.get("id", model.get("name", ""))),
            str(model.get("base_model") or model.get("base", "")),
            _format_timestamp(model.get("created_at")),
            str(model.get("owner") or model.get("organization", "")),
            str(model.get("status", "")),
        )
    return table
