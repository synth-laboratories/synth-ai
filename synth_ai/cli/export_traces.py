"""CLI helper for export_trace_sft.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
from synth_ai._utils.print_next_step import print_next_step

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = PROJECT_ROOT / "examples" / "warming_up_to_rl" / "export_trace_sft.py"


@click.command("export-traces")
@click.argument("script_args", nargs=-1)
def export_command(script_args: tuple[str, ...]) -> None:
    """Run the Warming Up to RL trace exporter."""

    if not EXPORT_SCRIPT.exists():
        raise click.ClickException(f"Export script not found: {EXPORT_SCRIPT}")

    env = os.environ.copy()
    cmd = [sys.executable, str(EXPORT_SCRIPT), *script_args]
    try:
        result = subprocess.run(cmd, env=env)
    except OSError as exc:
        raise click.ClickException(f"Failed to execute {EXPORT_SCRIPT}: {exc}") from exc

    if result.returncode != 0:
        raise click.ClickException(f"Exporter exited with code {result.returncode}")

    print_next_step(
        "kick off training with the exported dataset",
        [
            "synth-ai train",
        ],
    )
