"""Utility CLI command for managing Turso sqld binaries."""

from __future__ import annotations

import subprocess

import click

from synth_ai.cli.root import (  # type: ignore[import-untyped]
    SQLD_VERSION,
    find_sqld_binary,
    install_sqld,
)


def register(cli: click.Group) -> None:
    """Register the turso command on the main CLI group."""

    cli.add_command(turso)


def _get_sqld_version(binary: str) -> str | None:
    """Return the version string reported by the sqld binary."""

    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None

    output = result.stdout.strip() or result.stderr.strip()
    return output or None


@click.command()
@click.option(
    "--force",
    is_flag=True,
    help="Reinstall the pinned sqld build even if one is already available.",
)
def turso(force: bool) -> None:
    """Ensure the Turso sqld binary required for tracing v3 is installed."""

    existing_path = find_sqld_binary()

    if existing_path and not force:
        version_info = _get_sqld_version(existing_path)
        click.echo(f"✅ Turso sqld detected at {existing_path}.")
        if version_info:
            click.echo(f"   Reported version: {version_info}")
        if version_info and SQLD_VERSION not in version_info:
            click.echo(
                f"⚠️ Pinned version is {SQLD_VERSION}. Run with --force to install the supported build."
            )
        else:
            click.echo("No action taken. Use --force to reinstall the pinned build.")
        return

    if existing_path and force:
        click.echo(f"♻️ Reinstalling Turso sqld {SQLD_VERSION} (previously at {existing_path}).")
    else:
        click.echo(f"📦 Installing Turso sqld {SQLD_VERSION}…")

    try:
        installed_path = install_sqld()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surfaced as Click error
        raise click.ClickException(
            f"sqld installation failed (exit code {exc.returncode})."
        ) from exc

    click.echo(f"✅ sqld installed to {installed_path}")
    click.echo("Ensure ~/.local/bin is on your PATH before running Synth AI services.")
