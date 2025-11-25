#!/usr/bin/env python3
"""
CLI visualizer: live watch + experiment listings.

Placed beside scope.txt for discoverability.
"""

import asyncio
from datetime import datetime
from typing import Any

import click
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from synth_ai.cli._internal.storage import load_storage


def _open_db(db_url: str):
    create_storage, storage_config = load_storage()
    return create_storage(storage_config(connection_string=db_url))


class _State:
    def __init__(self):
        self.view: str = "experiments"  # experiments | experiment | usage | traces | recent
        self.view_arg: str | None = None
        self.limit: int = 20
        self.hours: float = 24.0
        self.last_msg: str = "Type 'help' for commands. 'q' to quit."
        self.error: str | None = None
        # UI state for a visible, blinking cursor in the input field
        self.cursor_on: bool = True


def _short_id(exp_id: str) -> str:
    return exp_id[:8] if exp_id else ""


def _format_currency(value: float) -> str:
    try:
        return f"${value:.4f}"
    except Exception:
        return "$0.0000"


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


async def _fetch_experiments(db_url: str):
    db = _open_db(db_url)
    await db.initialize()
    try:
        df = await db.query_traces(
            """
            SELECT 
                e.experiment_id,
                e.name,
                e.description,
                e.created_at,
                COUNT(DISTINCT st.session_id) as num_sessions,
                COUNT(DISTINCT ev.id) as num_events,
                COUNT(DISTINCT m.id) as num_messages,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) as total_tokens
            FROM experiments e
            LEFT JOIN session_traces st ON e.experiment_id = st.experiment_id
            LEFT JOIN events ev ON st.session_id = ev.session_id
            LEFT JOIN messages m ON st.session_id = m.session_id
            GROUP BY e.experiment_id, e.name, e.description, e.created_at
            ORDER BY e.created_at DESC
            """
        )
        return df
    finally:
        await db.close()


def _experiments_table(df, limit: int | None = None) -> Table:
    table = Table(
        title="Synth AI Experiments",
        title_style="bold cyan",
        show_edge=False,
        box=box.SIMPLE,
        header_style="bold",
        pad_edge=False,
    )
    for col in ["ID", "Name", "Sessions", "Events", "Msgs", "Cost", "Tokens", "Created"]:
        table.add_column(
            col, justify="right" if col in {"Sessions", "Events", "Msgs", "Tokens"} else "left"
        )

    if df is not None and not df.empty:
        rows = df.itertuples(index=False)
        for count, row in enumerate(rows, start=1):
            if limit is not None and count > limit:
                break
            table.add_row(
                _short_id(getattr(row, "experiment_id", "")),
                str(getattr(row, "name", "Unnamed"))[:28],
                _format_int(getattr(row, "num_sessions", 0)),
                _format_int(getattr(row, "num_events", 0)),
                _format_int(getattr(row, "num_messages", 0)),
                _format_currency(float(getattr(row, "total_cost", 0.0) or 0.0)),
                _format_int(getattr(row, "total_tokens", 0)),
                str(getattr(row, "created_at", "")),
            )
    else:
        table.add_row("-", "No experiments found", "-", "-", "-", "-", "-", "-")

    return table


async def _experiment_detail(db_url: str, experiment_id: str) -> dict[str, Any]:
    db = _open_db(db_url)
    await db.initialize()
    try:
        exp_df = await db.query_traces(
            """
            SELECT * FROM experiments WHERE experiment_id LIKE :exp_id
            """,
            {"exp_id": f"{experiment_id}%"},
        )
        if exp_df.empty:
            return {"not_found": True}

        exp = exp_df.iloc[0]

        sessions = await db.get_sessions_by_experiment(exp["experiment_id"])
        stats_df = await db.query_traces(
            """
            SELECT 
                COUNT(DISTINCT ev.id) as total_events,
                COUNT(DISTINCT m.id) as total_messages,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost,
                SUM(CASE WHEN ev.event_type = 'cais' THEN ev.total_tokens ELSE 0 END) as total_tokens
            FROM session_traces st
            LEFT JOIN events ev ON st.session_id = ev.session_id
            LEFT JOIN messages m ON st.session_id = m.session_id
            WHERE st.experiment_id = :exp_id
            """,
            {"exp_id": exp["experiment_id"]},
        )
        stats = stats_df.iloc[0] if not stats_df.empty else None

        return {
            "experiment": exp,
            "sessions": sessions or [],
            "stats": stats,
        }
    finally:
        await db.close()


def _render_experiment_panel(detail: dict[str, Any]) -> Panel:
    if detail.get("not_found"):
        return Panel("No experiment found for given ID", title="Experiment", border_style="red")

    exp = detail["experiment"]
    stats = detail.get("stats")
    lines: list[str] = []
    lines.append(f"[bold]🧪 {exp['name']}[/bold]  ([dim]{exp['experiment_id']}[/dim])")
    if exp.get("description"):
        lines.append(exp["description"])
    lines.append("")
    if stats is not None:
        lines.append(
            f"[bold]Stats[/bold]  Events: {_format_int(stats['total_events'])}  "
            f"Messages: {_format_int(stats['total_messages'])}  "
            f"Cost: {_format_currency(float(stats['total_cost'] or 0.0))}  "
            f"Tokens: {_format_int(stats['total_tokens'])}"
        )
    lines.append(f"Created: {exp['created_at']}")
    lines.append("")
    sessions = detail.get("sessions", [])
    if sessions:
        lines.append("[bold]Sessions[/bold]")
        for s in sessions[:25]:
            lines.append(
                f"  - {s['session_id']}  [dim]{s['created_at']}[/dim]  "
                f"steps={s['num_timesteps']} events={s['num_events']} msgs={s['num_messages']}"
            )
    else:
        lines.append("No sessions found for experiment.")

    body = "\n".join(lines)
    return Panel(body, title="Experiment", border_style="cyan")


def register(cli):
    """Attach commands to the top-level click group."""

    # Note: The former interactive `watch` command has been removed in favor of
    # one-off commands (e.g., `synth-ai experiments`, `synth-ai recent`).

    @cli.command()
    @click.option(
        "--url",
        "db_url",
        default="sqlite+aiosqlite:///./synth_ai.db/dbs/default/data",
        help="Database URL",
    )
    @click.option("--limit", default=50, type=int, help="Max rows to display")
    def experiments(db_url: str, limit: int):
        """Print a snapshot table of experiments."""
        console = Console()

        async def _run():
            df = await _fetch_experiments(db_url)
            table = _experiments_table(df, limit)
            console.print(table)
            console.print(
                "\n[dim]Tip:[/dim] use [cyan]synth-ai experiment <id>[/cyan] for details, "
                "[cyan]synth-ai usage[/cyan] for model usage.",
                sep="",
            )

        asyncio.run(_run())

    @cli.command()
    @click.argument("experiment_id")
    @click.option(
        "--url",
        "db_url",
        default="sqlite+aiosqlite:///./synth_ai.db/dbs/default/data",
        help="Database URL",
    )
    def experiment(experiment_id: str, db_url: str):
        """Show details and sessions for an experiment (accepts partial ID)."""
        console = Console()

        async def _run():
            detail = await _experiment_detail(db_url, experiment_id)
            panel = _render_experiment_panel(detail)
            console.print(panel)

        asyncio.run(_run())

    @cli.command()
    @click.option(
        "--url",
        "db_url",
        default="sqlite+aiosqlite:///./synth_ai.db/dbs/default/data",
        help="Database URL",
    )
    @click.option("--model", "model_name", default=None, help="Filter by model name")
    def usage(db_url: str, model_name: str | None):
        """Show model usage statistics (tokens, cost)."""
        console = Console()

        async def _run():
            db = _open_db(db_url)
            await db.initialize()
            try:
                df = await db.get_model_usage(model_name=model_name)
            finally:
                await db.close()

            if df is None or df.empty:
                console.print("[dim]No model usage data found.[/dim]")
                return

            table = Table(title="Model Usage", header_style="bold", box=box.SIMPLE)
            for col in df.columns:
                table.add_column(str(col))
            for row in df.itertuples(index=False):
                table.add_row(*[str(cell) for cell in row])
            console.print(table)

        asyncio.run(_run())


async def _build_view(console: Console, db_url: str, state: _State):
    header = f"[bold]Synth AI[/bold] • DB: [dim]{db_url}[/dim] • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    help_bar = (
        "[dim]Commands:[/dim] "
        "[cyan]experiments[/cyan], [cyan]experiment <id>[/cyan], [cyan]usage [model][/cyan], "
        "[cyan]traces[/cyan], [cyan]recent [hours][/cyan], [cyan]help[/cyan], [cyan]q[/cyan]"
    )

    body: Any
    if state.view == "experiments":
        df = await _fetch_experiments(db_url)
        body = _experiments_table(df, state.limit)
    elif state.view == "experiment" and state.view_arg:
        detail = await _experiment_detail(db_url, state.view_arg)
        body = _render_experiment_panel(detail)
    elif state.view == "usage":
        body = await _usage_table(db_url, state.view_arg)
    elif state.view == "traces":
        body = await _traces_table(db_url, state.limit)
    elif state.view == "recent":
        body = await _recent_table(db_url, state.hours, state.limit)
    else:
        body = Panel("Unknown view", border_style="red")

    footer_lines = []
    if state.error:
        footer_lines.append(f"[red]Error:[/red] {state.error}")
    if state.last_msg:
        footer_lines.append(state.last_msg)
    footer = "\n".join(footer_lines) if footer_lines else help_bar

    # Render a visible input field with an outline and a blinking cursor so users
    # know where to type commands. This is a visual affordance only; input is
    # still read from stdin in the background.
    cursor_char = "█" if state.cursor_on else " "
    placeholder = "Type a command and press Enter"
    # Show a dim placeholder when idle; show last message above via subtitle
    input_text = f"›  [dim]{placeholder}[/dim]  {cursor_char}"
    input_panel = Panel(
        input_text,
        box=box.SQUARE,
        border_style="magenta",
        padding=(0, 1),
        title="Command",
        title_align="left",
    )

    combined = Group(
        Align.left(body),
        Align.left(input_panel),
    )

    return Panel(
        combined,
        title=header,
        border_style="green",
        subtitle=footer,
    )


async def _handle_command(cmd: str, state: _State):
    if not cmd:
        return
    parts = cmd.split()
    c = parts[0].lower()
    args = parts[1:]

    if c in {":q", "q", "quit", "exit"}:
        raise KeyboardInterrupt()
    if c in {"help", ":h", "h", "?"}:
        state.last_msg = (
            "experiments | experiment <id> | usage [model] | traces | recent [hours] | q"
        )
        return
    if c in {"experiments", "exp", "e"}:
        state.view = "experiments"
        state.view_arg = None
        state.last_msg = "Showing experiments"
        return
    if c in {"experiment", "x"} and args:
        state.view = "experiment"
        state.view_arg = args[0]
        state.last_msg = f"Experiment {args[0]}"
        return
    if c in {"usage", "u"}:
        state.view = "usage"
        state.view_arg = args[0] if args else None
        state.last_msg = f"Usage {state.view_arg or ''}"
        return
    if c in {"traces", "t"}:
        state.view = "traces"
        state.view_arg = None
        state.last_msg = "Recent sessions"
        return
    if c in {"recent", "r"}:
        hours = _parse_hours(args)
        if hours is not None:
            state.hours = hours
        state.view = "recent"
        state.last_msg = f"Recent {state.hours:g}h"
        return

    state.last_msg = f"Unknown command: {cmd}"


def _parse_hours(args: list[str]) -> float | None:
    if not args:
        return None
    # Accept formats: "6", "--6", "-6", "6h"
    token = args[0]
    token = token.lstrip("-")
    if token.endswith("h"):
        token = token[:-1]
    try:
        return float(token)
    except ValueError:
        return None


async def _usage_table(db_url: str, model_name: str | None):
    db = _open_db(db_url)
    await db.initialize()
    try:
        df = await db.get_model_usage(model_name=model_name)
    finally:
        await db.close()

    table = Table(title="Model Usage", header_style="bold", box=box.SIMPLE)
    if df is None or df.empty:
        table.add_column("Info")
        table.add_row("No model usage data found.")
        return table
    for col in df.columns:
        table.add_column(str(col))
    for row in df.itertuples(index=False):
        table.add_row(*[str(cell) for cell in row])
    return table


async def _traces_table(db_url: str, limit: int):
    db = _open_db(db_url)
    await db.initialize()
    try:
        df = await db.query_traces("SELECT * FROM session_summary ORDER BY created_at DESC")
    finally:
        await db.close()

    table = Table(title="Recent Sessions", box=box.SIMPLE, header_style="bold")
    for col in ["Session", "Experiment", "Events", "Msgs", "Timesteps", "Cost", "Created"]:
        table.add_column(col, justify="right" if col in {"Events", "Msgs", "Timesteps"} else "left")

    if df is None or df.empty:
        table.add_row("-", "No sessions found", "-", "-", "-", "-", "-")
    else:
        for count, (_, r) in enumerate(df.iterrows(), start=1):
            if count > limit:
                break
            table.add_row(
                str(r.get("session_id", ""))[:10],
                str(r.get("experiment_name", ""))[:24],
                f"{int(r.get('num_events', 0)):,}",
                f"{int(r.get('num_messages', 0)):,}",
                f"{int(r.get('num_timesteps', 0)):,}",
                f"${float(r.get('total_cost_usd', 0.0) or 0.0):.4f}",
                str(r.get("created_at", "")),
            )
    return table


async def _recent_table(db_url: str, hours: float, limit: int):
    # Inline the recent query to avoid cross-module coupling
    from datetime import timedelta

    start_time = datetime.now() - timedelta(hours=hours)
    db = _open_db(db_url)
    await db.initialize()
    try:
        query = """
            WITH windowed_sessions AS (
                SELECT * FROM session_traces WHERE created_at >= :start_time
            )
            SELECT 
                e.experiment_id,
                e.name,
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
            GROUP BY e.experiment_id, e.name
            ORDER BY window_end DESC
        """
        df = await db.query_traces(query, {"start_time": start_time})
    finally:
        await db.close()

    table = Table(title=f"Experiments in last {hours:g}h", header_style="bold", box=box.SIMPLE)
    for col in ["Experiment", "Runs", "First", "Last", "Events", "Msgs", "Cost", "Tokens"]:
        table.add_column(
            col, justify="right" if col in {"Runs", "Events", "Msgs", "Tokens"} else "left"
        )

    if df is None or df.empty:
        table.add_row("-", "0", "-", "-", "-", "-", "-", "-")
    else:
        for count, (_, r) in enumerate(df.iterrows(), start=1):
            if count > limit:
                break
            name = r.get("name") or "Unnamed"
            exp_disp = f"{name[:28]} [dim]({(str(r.get('experiment_id', ''))[:8])})[/dim]"
            table.add_row(
                exp_disp,
                f"{int(r.get('runs', 0)):,}",
                str(r.get("window_start", "")),
                str(r.get("window_end", "")),
                f"{int(r.get('events', 0)):,}",
                f"{int(r.get('messages', 0)):,}",
                f"${float(r.get('cost_usd', 0.0) or 0.0):.4f}",
                f"{int(r.get('tokens', 0)):,}",
            )
    return table
