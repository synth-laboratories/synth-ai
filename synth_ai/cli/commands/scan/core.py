"""Main scan command implementation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import click

from synth_ai.cli.commands.scan.cloudflare_scanner import scan_cloudflare_apps
from synth_ai.cli.commands.scan.local_scanner import scan_local_ports, scan_registry
from synth_ai.cli.commands.scan.models import ScannedApp
from synth_ai.cli.lib.env import resolve_env_var


def format_app_table(apps: list[ScannedApp]) -> str:
    """Format apps as a human-readable table."""
    if not apps:
        return "No active task apps found."

    # Calculate column widths
    name_width = max(len(app.name) for app in apps if app.name) if apps else 4
    port_width = 5  # "Port" header
    status_width = 10  # Status with icon
    type_width = max(len(app.type) for app in apps) if apps else 4
    via_width = max(len(app.discovered_via) for app in apps) if apps else 13
    
    # Additional metadata columns
    app_id_width = max(len(app.app_id or "") for app in apps) if apps else 0
    app_id_width = max(app_id_width, 7)  # "App ID" header
    
    version_width = max(len(app.version or "") for app in apps) if apps else 0
    version_width = max(version_width, 7)  # "Version" header

    # Ensure minimum widths
    name_width = max(name_width, 4)
    type_width = max(type_width, 4)
    via_width = max(via_width, 13)

    lines = [f"Found {len(apps)} active task app{'s' if len(apps) != 1 else ''}:\n"]
    
    # Header row
    header_parts = [
        f"{'Name':<{name_width}}",
        f"{'Port':<{port_width}}",
        f"{'Status':<{status_width}}",
        f"{'Type':<{type_width}}",
    ]
    if app_id_width > 0:
        header_parts.append(f"{'App ID':<{app_id_width}}")
    if version_width > 0:
        header_parts.append(f"{'Version':<{version_width}}")
    header_parts.append(f"{'Discovered Via':<{via_width}}")
    
    lines.append(" ".join(header_parts))
    
    # Separator
    total_width = sum(len(p) for p in header_parts) + len(header_parts) - 1
    lines.append("─" * total_width)

    for app in apps:
        status_icon = "✅" if app.health_status == "healthy" else "⚠️ " if app.health_status == "unhealthy" else "❓"
        status_display = f"{status_icon} {app.health_status}"
        
        port_str = str(app.port) if app.port else "-"
        
        row_parts = [
            f"{app.name:<{name_width}}",
            f"{port_str:<{port_width}}",
            f"{status_display:<{status_width}}",
            f"{app.type:<{type_width}}",
        ]
        if app_id_width > 0:
            row_parts.append(f"{(app.app_id or '-'):<{app_id_width}}")
        if version_width > 0:
            row_parts.append(f"{(app.version or '-'):<{version_width}}")
        row_parts.append(f"{app.discovered_via:<{via_width}}")
        
        lines.append(" ".join(row_parts))

    return "\n".join(lines)


def format_app_json(apps: list[ScannedApp]) -> str:
    """Format apps as JSON."""
    apps_data = []
    for app in apps:
        apps_data.append(
            {
                "name": app.name,
                "url": app.url,
                "type": app.type,
                "health_status": app.health_status,
                "port": app.port,
                "tunnel_mode": app.tunnel_mode,
                "tunnel_hostname": app.tunnel_hostname,
                "app_id": app.app_id,
                "task_name": app.task_name,
                "dataset_id": app.dataset_id,
                "version": app.version,
                "metadata": app.metadata,
                "discovered_via": app.discovered_via,
            }
        )

    summary = {
        "total_found": len(apps),
        "healthy": sum(1 for app in apps if app.health_status == "healthy"),
        "unhealthy": sum(1 for app in apps if app.health_status == "unhealthy"),
        "local_count": sum(1 for app in apps if app.type == "local"),
        "cloudflare_count": sum(1 for app in apps if app.type == "cloudflare"),
    }

    return json.dumps({"apps": apps_data, "scan_summary": summary}, indent=2)


async def run_scan(
    port_range: tuple[int, int],
    timeout: float,
    api_key: str | None,
    env_file: Path | None,
    verbose: bool,
) -> list[ScannedApp]:
    """Run the scan operation.

    Args:
        port_range: Tuple of (start_port, end_port)
        timeout: Health check timeout
        api_key: API key for health checks
        env_file: Specific .env file to check
        verbose: Show detailed progress

    Returns:
        List of discovered apps
    """
    start_port, end_port = port_range
    all_apps: list[ScannedApp] = []

    # Resolve API key
    env_api_key = api_key
    if not env_api_key:
        try:
            env_api_key = resolve_env_var("ENVIRONMENT_API_KEY")
        except Exception:
            env_api_key = os.getenv("ENVIRONMENT_API_KEY")

    if verbose:
        click.echo(f"Scanning ports {start_port}-{end_port}...", err=True)

    # Scan local ports
    local_apps = await scan_local_ports(start_port, end_port, env_api_key, timeout)
    all_apps.extend(local_apps)

    if verbose:
        click.echo(f"Found {len(local_apps)} local app(s)", err=True)

    # Scan Cloudflare apps
    synth_api_key = os.getenv("SYNTH_API_KEY")
    cloudflare_apps = await scan_cloudflare_apps(synth_api_key, env_api_key, env_file, timeout)
    all_apps.extend(cloudflare_apps)

    if verbose:
        click.echo(f"Found {len(cloudflare_apps)} Cloudflare app(s)", err=True)

    # Scan service records (local services deployed via synth-ai)
    from synth_ai.cli.commands.scan.local_scanner import scan_service_records
    service_record_apps = await scan_service_records(env_api_key, timeout)
    all_apps.extend(service_record_apps)

    if verbose:
        click.echo(f"Found {len(service_record_apps)} service record(s)", err=True)

    # Scan registry (for reference, but these don't have URLs)
    registry_apps = scan_registry()
    # Only add registry apps that weren't already discovered
    discovered_app_ids = {app.app_id for app in all_apps if app.app_id}
    new_registry_apps = [app for app in registry_apps if app.app_id and app.app_id not in discovered_app_ids]
    all_apps.extend(new_registry_apps)

    if verbose:
        click.echo(f"Found {len(new_registry_apps)} registry app(s) not yet running", err=True)

    # Deduplicate apps by URL (prefer better sources and metadata)
    seen_urls: dict[str, ScannedApp] = {}
    for app in all_apps:
        if app.url and app.url in seen_urls:
            # Prefer apps with better discovery method and metadata
            existing = seen_urls[app.url]
            
            # Priority order for discovery methods:
            # 1. service_records / tunnel_records (have metadata like app_id, task_app_path)
            # 2. cloudflared_process / backend_api (tunnels)
            # 3. port_scan (basic discovery)
            # 4. registry (no URL)
            
            priority_order = {
                "service_records": 1,
                "tunnel_records": 1,
                "backend_api": 2,
                "cloudflared_process": 2,
                "port_scan": 3,
                "registry": 4,
            }
            
            app_priority = priority_order.get(app.discovered_via, 99)
            existing_priority = priority_order.get(existing.discovered_via, 99)
            
            # Prefer higher priority discovery method
            if app_priority < existing_priority:
                seen_urls[app.url] = app
            elif app_priority == existing_priority:
                # Same priority - prefer better health status or better metadata
                if app.health_status != "unknown" and existing.health_status == "unknown":
                    seen_urls[app.url] = app
                elif app.app_id and not existing.app_id:
                    # Prefer app with app_id
                    seen_urls[app.url] = app
                elif app.discovered_via != "registry" and existing.discovered_via == "registry":
                    seen_urls[app.url] = app
        elif app.url:
            seen_urls[app.url] = app
        elif app.discovered_via == "registry":
            # Registry apps without URLs - add them separately
            seen_urls[f"registry:{app.app_id}"] = app

    return list(seen_urls.values())


@click.command("scan")
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
def scan_command(
    port_range: str,
    timeout: float,
    api_key: str | None,
    output_json: bool,
    verbose: bool,
    env_file: Path | None,
) -> None:
    """Scan for active Cloudflare and local task apps.

    Discovers and performs health checks on running task applications deployed
    locally or via Cloudflare tunnels. Returns structured information in table
    or JSON format, suitable for terminal use or programmatic consumption by
    CLI agents and automation tools.

    Discovery Methods:
        - Port scanning: Scans specified port range for local HTTP servers
        - Service records: Reads deployed local services from persistent records
        - Tunnel records: Reads deployed Cloudflare tunnels from persistent records
        - Process scanning: Inspects running cloudflared processes for tunnel URLs
        - Backend API: Queries backend for managed tunnel information
        - Registry: Checks task app registry for registered apps

    Health Checks:
        - Performs HTTP GET requests to /health endpoints
        - Extracts metadata from /info endpoints (app_id, version, task_name, etc.)
        - Supports API key authentication via X-API-Key header

    Output Formats:
        - Table (default): Human-readable table with columns for name, port, status,
          type, app ID, version, and discovery method
        - JSON (--json): Machine-readable JSON with full metadata and scan summary

    Examples:
        # Scan default port range (8000-8100) and show table
        $ synth-ai scan

        # Scan specific port range with verbose output
        $ synth-ai scan --port-range 8000:9000 --verbose

        # Get JSON output for programmatic use
        $ synth-ai scan --json

        # Use custom API key and timeout
        $ synth-ai scan --api-key YOUR_KEY --timeout 5.0

    Args:
        port_range: Port range to scan (format: START:END, e.g., "8000:8100")
        timeout: Health check timeout in seconds (default: 2.0)
        api_key: API key for health checks (default: from ENVIRONMENT_API_KEY env var)
        output_json: Output results as JSON instead of table
        verbose: Show detailed scanning progress
        env_file: (Deprecated) Not used - tunnels are discovered from processes

    Returns:
        None (outputs to stdout)

    Raises:
        click.BadParameter: If port range format is invalid or out of valid range
    """
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
        raise click.BadParameter(f"Invalid port range format: {port_range}. Use START:END (e.g., 8000:8100)") from e

    if start_port < 1 or end_port > 65535 or start_port > end_port:
        raise click.BadParameter(f"Invalid port range: {start_port}-{end_port}")

    # Run scan
    apps = asyncio.run(run_scan((start_port, end_port), timeout, api_key, env_file, verbose))

    # Output results
    if output_json:
        click.echo(format_app_json(apps))
    else:
        click.echo(format_app_table(apps))

