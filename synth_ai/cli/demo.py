#!/usr/bin/env python3
"""
CLI: interactive launcher for example demos.

Finds all `run_demo.sh` scripts under `examples/` and lets the user pick one
to run. Intended to be used as: `uvx synth-ai demo` or `synth-ai demo`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List

import click


def _find_demo_scripts(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("run_demo.sh") if p.is_file()])


def register(cli):
    @cli.command()
    @click.option("--list", "list_only", is_flag=True, help="List available demos and exit")
    @click.option("-f", "filter_term", default="", help="Filter demos by substring")
    def demo(list_only: bool, filter_term: str):
        """Launch an interactive demo from examples/"""
        repo_root = Path(os.getcwd())
        examples_dir = repo_root / "examples"
        demos = _find_demo_scripts(examples_dir)
        if filter_term:
            demos = [p for p in demos if filter_term.lower() in str(p).lower()]

        if not demos:
            click.echo("No run_demo.sh scripts found under examples/.")
            return

        if list_only:
            click.echo("Available demos:")
            for p in demos:
                click.echo(f" - {p.relative_to(repo_root)}")
            return

        click.echo("Available demos:")
        for idx, p in enumerate(demos, start=1):
            click.echo(f" {idx}. {p.relative_to(repo_root)}")
        click.echo("")

        def _validate_choice(val: str) -> int:
            try:
                i = int(val)
            except Exception:
                raise click.BadParameter("Enter a number from the list")
            if i < 1 or i > len(demos):
                raise click.BadParameter(f"Choose a number between 1 and {len(demos)}")
            return i

        choice = click.prompt("Select a demo to run", value_proc=_validate_choice)
        script = demos[choice - 1]

        click.echo("")
        click.echo(f"🚀 Running {script.relative_to(repo_root)}\n")

        # Run via bash to avoid relying on executable bit; inherit environment
        try:
            subprocess.run(["bash", str(script)], check=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Demo exited with non-zero status: {e.returncode}")
        except KeyboardInterrupt:
            click.echo("\n🛑 Demo interrupted by user")

