#!/usr/bin/env python3
"""
CLI: human-friendly manual for Synth AI commands and options.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


def _commands_table() -> Table:
    t = Table(title="Commands", box=box.SIMPLE, header_style="bold")
    t.add_column("Command")
    t.add_column("Summary")
    t.add_row(
        "balance",
        "Show remaining credit balance (USD) and a compact spend summary for last 24h and 7d.\n"
        "Options: --base-url, --api-key, --usage",
    )
    t.add_row(
        "traces",
        "List local trace DBs, trace counts, experiments, and per-system counts.\n"
        "Options: --root",
    )
    t.add_row(
        "experiments",
        "Snapshot table of experiments from the local traces DB.\n"
        "Options: --url, --limit",
    )
    t.add_row(
        "experiment <id>",
        "Details and sessions for an experiment (accepts partial ID).\n"
        "Options: --url",
    )
    t.add_row(
        "usage",
        "Model usage statistics (tokens, cost).\n"
        "Options: --url, --model",
    )
    t.add_row(
        "status",
        "DB stats, systems, and environment service health.\n"
        "Options: --url, --service-url",
    )
    t.add_row(
        "calc '<expr>'",
        "Evaluate a simple arithmetic expression (e.g., 2*(3+4)).",
    )
    t.add_row(
        "env list | env register | env unregister",
        "Manage environment registry via the service.\n"
        "Options vary; see examples.",
    )
    return t


def _env_table() -> Table:
    t = Table(title="Environment Variables", box=box.SIMPLE, header_style="bold")
    t.add_column("Variable")
    t.add_column("Used By")
    t.add_column("Purpose")
    t.add_row("SYNTH_BACKEND_BASE_URL", "balance", "Backend base URL (preferred) e.g. http://localhost:8000/api/v1")
    t.add_row("BACKEND_BASE_URL", "balance", "Fallback backend base URL")
    t.add_row("LOCAL_BACKEND_URL", "balance", "Another fallback backend base URL")
    t.add_row("SYNTH_BASE_URL", "balance", "Generic base URL (may point to Modal, guarded)")
    t.add_row("SYNTH_BACKEND_API_KEY", "balance", "Backend API key (preferred)")
    t.add_row("SYNTH_API_KEY", "balance, env*", "API key used if backend-specific key not set")
    t.add_row("DEFAULT_DEV_API_KEY", "balance", "Dev fallback key for local testing")
    t.add_row("SYNTH_TRACES_ROOT", "traces", "Root directory of local trace DBs (default ./synth_ai.db/dbs)")
    return t


def _examples_table() -> Table:
    t = Table(title="Examples", box=box.SIMPLE, header_style="bold")
    t.add_column("Command")
    t.add_column("Example")
    t.add_row("Balance (local backend)", "uvx . balance")
    t.add_row("Balance with URL+key", "uvx . balance --base-url http://localhost:8000 --api-key $SYNTH_API_KEY")
    t.add_row("Traces (default root)", "uvx . traces")
    t.add_row("Traces (custom root)", "uvx . traces --root /path/to/dbs")
    t.add_row("Experiments", "uvx . experiments --limit 20")
    t.add_row("Experiment detail", "uvx . experiment abcd1234")
    t.add_row("Usage by model", "uvx . usage --model gpt-4o-mini")
    t.add_row("Status", "uvx . status")
    t.add_row("Calc", "uvx . calc '2*(3+4)'")
    t.add_row("Env list", "uvx . env list --service-url http://localhost:8901")
    return t


def register(cli):
    @cli.command(name="man")
    def man():
        """Show Synth AI CLI manual with commands, options, env vars, and examples."""
        console = Console()
        console.print(Panel("Synth AI CLI Manual", border_style="cyan"))
        console.print(_commands_table())
        console.print(_env_table())
        console.print(_examples_table())

