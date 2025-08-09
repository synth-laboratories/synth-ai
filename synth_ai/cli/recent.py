#!/usr/bin/env python3
"""
CLI: experiments active in the last K hours with summary stats.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import box


def _fmt_int(v) -> str:
    try:
        return f"{int(v):,}"
    except Exception:
        return "0"


def _fmt_money(v) -> str:
    try:
        return f"${float(v or 0.0):.4f}"
    except Exception:
        return "$0.0000"


def _fmt_time(v) -> str:
    try:
        return str(v)
    except Exception:
        return "-"


async def _fetch_recent(db_url: str, hours: float):
    from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager

    start_time = datetime.now() - timedelta(hours=hours)

    db = AsyncSQLTraceManager(db_url)
    await db.initialize()
    try:
        query = """
            WITH windowed_sessions AS (
                SELECT *
                FROM session_traces
                WHERE created_at >= :start_time
            )
            SELECT 
                e.experiment_id,
                e.name,
                e.description,
                MIN(ws.created_at) AS window_start,
                MAX(ws.created_at) AS window_end,
                COUNT(DISTINCT ws.session_id) AS runs,
                COUNT(DISTINCT ev.id) AS events,
                COUNT(DISTINCT m.id) AS messages,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 AS cost_usd,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) AS tokens
            FROM windowed_sessions ws
            LEFT JOIN experiments e ON ws.experiment_id = e.experiment_id
            LEFT JOIN events ev ON ws.session_id = ev.session_id
            LEFT JOIN messages m ON ws.session_id = m.session_id
            GROUP BY e.experiment_id, e.name, e.description
            ORDER BY window_end DESC
        """
        df = await db.query_traces(query, {"start_time": start_time})
        return df
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
    @click.option("--hours", default=24.0, type=float, help="Look back window in hours")
    @click.option("--limit", default=20, type=int, help="Max experiments to display")
    def recent(db_url: str, hours: float, limit: int):
        """List experiments with activity in the last K hours with summary stats."""

        console = Console()

        async def _run():
            df = await _fetch_recent(db_url, hours)

            table = Table(title=f"Experiments in last {hours:g}h", header_style="bold", box=box.SIMPLE)
            for col in ["Experiment", "Runs", "First", "Last", "Events", "Msgs", "Cost", "Tokens"]:
                table.add_column(col, justify="right" if col in {"Runs","Events","Msgs","Tokens"} else "left")

            if df is None or df.empty:
                table.add_row("-", "0", "-", "-", "-", "-", "-", "-")
            else:
                count = 0
                for _, r in df.iterrows():
                    if count >= limit:
                        break
                    count += 1
                    name = r.get("name") or "Unnamed"
                    exp_disp = f"{name[:28]} [dim]({_short(r.get('experiment_id'))})[/dim]"
                    table.add_row(
                        exp_disp,
                        _fmt_int(r.get("runs", 0)),
                        _fmt_time(r.get("window_start")),
                        _fmt_time(r.get("window_end")),
                        _fmt_int(r.get("events", 0)),
                        _fmt_int(r.get("messages", 0)),
                        _fmt_money(r.get("cost_usd", 0.0)),
                        _fmt_int(r.get("tokens", 0)),
                    )

            console.print(table)

        def _short(exp_id) -> str:
            try:
                return str(exp_id)[:8]
            except Exception:
                return ""

        asyncio.run(_run())
