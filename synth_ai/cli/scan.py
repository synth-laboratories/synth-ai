"""Scan command."""

import asyncio
from pathlib import Path

import click


@click.command()
@click.option(
    "--port-range",
    type=str,
    default="8000:8100",
    help="Port range to scan for local apps (format: START:END)",
)
@click.option(
    "--timeout",
    type=float,
    default=2.0,
    show_default=True,
    help="Health check timeout in seconds",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    envvar="ENVIRONMENT_API_KEY",
    help="API key for health checks (default: from ENVIRONMENT_API_KEY env var)",
)
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.option("--verbose", is_flag=True, help="Show detailed scanning progress")
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="(Deprecated) Not used - tunnels are discovered from running processes",
    hidden=True,
)
def scan(
    port_range: str,
    timeout: float,
    api_key: str | None,
    output_json: bool,
    verbose: bool,
    env_file: Path | None,
) -> None:
    """Scan for active Cloudflare and local task apps."""
    from synth_ai.core.scanning import format_app_json, format_app_table, run_scan

    # Parse port range
    try:
        if ":" in port_range:
            start_str, end_str = port_range.split(":", 1)
            start_port = int(start_str.strip())
            end_port = int(end_str.strip())
        else:
            start_port = int(port_range.strip())
            end_port = start_port
    except ValueError as e:
        raise click.BadParameter(
            f"Invalid port range format: {port_range}. Use START:END (e.g., 8000:8100)"
        ) from e

    if start_port < 1 or end_port > 65535 or start_port > end_port:
        raise click.BadParameter(f"Invalid port range: {start_port}-{end_port}")

    verbose_callback = None
    if verbose:

        def verbose_callback(msg: str) -> None:
            click.echo(msg, err=True)

    apps = asyncio.run(
        run_scan((start_port, end_port), timeout, api_key, env_file, verbose_callback)
    )

    if output_json:
        click.echo(format_app_json(apps))
    else:
        click.echo(format_app_table(apps))
