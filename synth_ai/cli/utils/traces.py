#!/usr/bin/env python3
"""
CLI: basic info about traces (runs).
"""

import asyncio
import os

import click
from rich import box
from rich.console import Console
from rich.table import Table

from synth_ai.cli._internal.storage import load_storage


def register(cli):
    @cli.command()
    @click.option(
        "--url",
        "db_url",
        default="sqlite+aiosqlite:///./synth_ai.db/dbs/default/data",
        help="Database URL",
    )
    @click.option("--limit", default=25, type=int, help="Max sessions to display")
    def traces(db_url: str, limit: int):
        """Show local trace DBs, traces per DB, and per-system counts."""
        console = Console()

        async def _run():
            # Discover DBs under ./synth_ai.db/dbs (or override via env)
            root = os.getenv("SYNTH_TRACES_ROOT", "./synth_ai.db/dbs")
            if not os.path.isdir(root):
                console.print(f"[red]No DB root found:[/red] {root}")
                return

            entries: list[tuple[str, str]] = []
            for name in sorted(os.listdir(root)):
                path = os.path.join(root, name)
                data_path = os.path.join(path, "data")
                if os.path.isdir(path) and os.path.isfile(data_path):
                    entries.append((name, os.path.abspath(path)))

            if not entries:
                console.print("[dim]No trace databases found.[/dim]")
                return

            def _dir_size_bytes(dir_path: str) -> int:
                total = 0
                for dp, _, files in os.walk(dir_path):
                    for fn in files:
                        fp = os.path.join(dp, fn)
                        import contextlib

                        with contextlib.suppress(OSError):
                            total += os.path.getsize(fp)
                return total

            async def db_counts(db_dir: str) -> tuple[int, dict[str, int], int, str | None, int]:
                data_file = os.path.join(db_dir, "data")
                create_storage, storage_config = load_storage()
                mgr = create_storage(storage_config(connection_string=f"sqlite+aiosqlite:///{data_file}"))
                await mgr.initialize()
                try:
                    traces_df = await mgr.query_traces("SELECT COUNT(*) AS c FROM session_traces")
                    traces_count = (
                        int(traces_df.iloc[0]["c"])
                        if traces_df is not None and not traces_df.empty
                        else 0
                    )
                    try:
                        systems_df = await mgr.query_traces(
                            "SELECT system_type, COUNT(*) AS c FROM systems GROUP BY system_type"
                        )
                        system_counts = (
                            {
                                str(r["system_type"] or "-"): int(r["c"] or 0)
                                for _, r in systems_df.iterrows()
                            }
                            if systems_df is not None
                            and hasattr(systems_df, "iterrows")
                            and not systems_df.empty
                            else {}
                        )
                    except Exception:
                        system_counts = {}
                    try:
                        exps_df = await mgr.query_traces("SELECT COUNT(*) AS c FROM experiments")
                        exps_count = (
                            int(exps_df.iloc[0]["c"])
                            if exps_df is not None and not exps_df.empty
                            else 0
                        )
                    except Exception:
                        exps_count = 0
                    try:
                        last_df = await mgr.query_traces(
                            "SELECT MAX(created_at) AS last_created_at FROM session_traces"
                        )
                        last_created = (
                            str(last_df.iloc[0]["last_created_at"])
                            if last_df is not None and not last_df.empty
                            else None
                        )
                    except Exception:
                        last_created = None
                    size_bytes = _dir_size_bytes(db_dir)
                    return traces_count, system_counts, exps_count, last_created, size_bytes
                finally:
                    await mgr.close()

            results = []
            for name, db_dir in entries:
                try:
                    counts = await db_counts(db_dir)
                except Exception:
                    counts = (0, {}, 0, None, 0)
                results.append((name, counts))

            # DB summary table
            summary = Table(title="Trace Databases", box=box.SIMPLE, header_style="bold")
            for col in ["DB", "Traces", "Experiments", "Last Activity", "Size (GB)"]:
                summary.add_column(
                    col, justify="right" if col in {"Traces", "Experiments"} else "left"
                )

            aggregate_systems: dict[str, int] = {}
            total_bytes = 0
            for name, (
                traces_count,
                system_counts,
                experiments_count,
                last_created_at,
                size_bytes,
            ) in results:
                total_bytes += int(size_bytes or 0)
                gb = int(size_bytes or 0) / (1024**3)
                summary.add_row(
                    name,
                    f"{traces_count:,}",
                    f"{experiments_count:,}",
                    str(last_created_at or "-"),
                    f"{gb:.2f}",
                )
                for k, v in system_counts.items():
                    aggregate_systems[k] = aggregate_systems.get(k, 0) + int(v)
            console.print(summary)

            # Total storage line
            total_gb = total_bytes / (1024**3)
            console.print(f"[dim]Total storage across DBs:[/dim] [bold]{total_gb:.2f} GB[/bold]")

            # Per-system aggregate across DBs
            if aggregate_systems:
                st = Table(title="Per-System (all DBs)", box=box.SIMPLE, header_style="bold")
                st.add_column("System")
                st.add_column("Count", justify="right")
                for sys_name, count in sorted(
                    aggregate_systems.items(), key=lambda x: (-x[1], x[0])
                ):
                    st.add_row(sys_name or "-", f"{int(count):,}")
                console.print(st)

        asyncio.run(_run())
