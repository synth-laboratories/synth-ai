#!/usr/bin/env python3
"""
CLI: status of agent runs/versions and environment service.
"""

import asyncio
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import requests


async def _db_stats(db_url: str) -> dict:
    from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager

    db = AsyncSQLTraceManager(db_url)
    await db.initialize()
    try:
        out: dict = {}
        # Totals
        totals = await db.query_traces(
            """
            SELECT 
              (SELECT COUNT(*) FROM session_traces) AS sessions,
              (SELECT COUNT(*) FROM experiments) AS experiments,
              (SELECT COUNT(*) FROM events) AS events,
              (SELECT COUNT(*) FROM messages) AS messages,
              (SELECT COALESCE(SUM(CASE WHEN event_type='cais' THEN cost_usd ELSE 0 END),0)/100.0 FROM events) AS total_cost_usd,
              (SELECT COALESCE(SUM(CASE WHEN event_type='cais' THEN total_tokens ELSE 0 END),0) FROM events) AS total_tokens
        """
        )
        if not totals.empty:
            out["totals"] = totals.iloc[0].to_dict()
        else:
            out["totals"] = {}

        # Systems summary
        systems = await db.query_traces(
            """
            SELECT system_type, COUNT(*) as count FROM systems GROUP BY system_type
        """
        )
        out["systems"] = systems

        versions = await db.query_traces(
            """
            SELECT COUNT(*) as version_count FROM system_versions
        """
        )
        if not versions.empty:
            out["version_count"] = int(versions.iloc[0]["version_count"]) 
        else:
            out["version_count"] = 0
        return out
    finally:
        await db.close()


def register(cli):
    @cli.command()
    @click.option(
        "--url",
        "db_url",
        default="sqlite+aiosqlite:///./synth_ai.db/dbs/default/data",
        help="Database URL",
    )
    @click.option("--service-url", default="http://127.0.0.1:8901", help="Environment service URL")
    def status(db_url: str, service_url: str):
        """Show DB stats, agent/environment system counts, and env service health."""
        console = Console()

        async def _run():
            # DB
            stats = await _db_stats(db_url)

            # Env service
            health_text = "[red]unreachable[/red]"
            envs_list = []
            try:
                r = requests.get(f"{service_url}/health", timeout=2)
                if r.ok:
                    data = r.json()
                    health_text = "[green]ok[/green]"
                    envs_list = data.get("supported_environments", [])
                else:
                    health_text = f"[red]{r.status_code}[/red]"
            except Exception:
                pass

            # Render
            totals = stats.get("totals", {})
            lines = []
            lines.append(f"DB: [dim]{db_url}[/dim]")
            lines.append(
                f"Experiments: {int(totals.get('experiments', 0)):,}  "
                f"Sessions: {int(totals.get('sessions', 0)):,}  "
                f"Events: {int(totals.get('events', 0)):,}  "
                f"Messages: {int(totals.get('messages', 0)):,}"
            )
            lines.append(
                f"Cost: ${float(totals.get('total_cost_usd', 0.0) or 0.0):.4f}  "
                f"Tokens: {int(totals.get('total_tokens', 0)):,}"
            )
            lines.append("")
            lines.append(f"Env Service: {health_text}  [dim]{service_url}[/dim]")
            if envs_list:
                lines.append("Environments: " + ", ".join(sorted(envs_list)[:10]) + (" ..." if len(envs_list) > 10 else ""))

            panel_main = Panel("\n".join(lines), title="Synth AI Status", border_style="cyan")
            console.print(panel_main)

            # Systems table
            sys_df = stats.get("systems")
            if sys_df is not None and not sys_df.empty:
                tbl = Table(title=f"Systems (versions: {stats.get('version_count', 0)})", box=box.SIMPLE, header_style="bold")
                tbl.add_column("Type")
                tbl.add_column("Count", justify="right")
                for _, r in sys_df.iterrows():
                    tbl.add_row(str(r.get("system_type", "-")), f"{int(r.get('count', 0)):,}")
                console.print(tbl)

        asyncio.run(_run())
