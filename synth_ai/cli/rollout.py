"""Helper CLI for running example rollout scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import click
from synth_ai._utils.print_next_step import print_next_step

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _discover_rollout_scripts() -> list[Path]:
    """Return rollout scripts shipped with the SDK examples."""

    search_roots: Iterable[Path] = [
        PROJECT_ROOT / "examples",
    ]

    candidates: list[Path] = []
    seen: set[Path] = set()

    for base in search_roots:
        if not base.exists():
            continue
        for path in base.rglob("run_*rollout*.py"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)

    candidates.sort()
    return candidates


def _prompt_for_script() -> Path:
    scripts = _discover_rollout_scripts()
    if not scripts:
        raise click.ClickException("No rollout scripts found under examples/.")

    click.echo("Select a rollout helper:")
    for idx, script in enumerate(scripts, start=1):
        click.echo(f"  {idx}) {script.name}")
    click.echo("  0) Abort")

    choice = click.prompt("Enter choice", type=int, default=1)
    if choice == 0:
        raise click.ClickException("Aborted by user")
    if choice < 0 or choice > len(scripts):
        raise click.ClickException("Invalid selection")
    return scripts[choice - 1]


@click.command("rollout")
@click.option(
    "--script",
    "script_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to rollout script (defaults to selecting from examples).",
)
@click.argument("script_args", nargs=-1)
def rollout_command(script_path: Path | None, script_args: tuple[str, ...]) -> None:
    """Invoke an example rollout helper."""

    target = script_path or _prompt_for_script()

    if not target.exists():
        raise click.ClickException(f"Script not found: {target}")

    env = os.environ.copy()
    cmd = [sys.executable, str(target), *script_args]
    try:
        result = subprocess.run(cmd, env=env)
    except OSError as exc:
        raise click.ClickException(f"Failed to execute {target}: {exc}") from exc

    if result.returncode != 0:
        raise click.ClickException(f"Rollout script exited with code {result.returncode}")

    print_next_step(
        "export the collected traces",
        ["synth-ai export-traces"],
    )
