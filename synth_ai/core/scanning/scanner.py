"""Main scan logic."""

import json
import os
from pathlib import Path
from typing import Callable

from synth_ai.core.scanning.cloudflare_scanner import scan_cloudflare_apps
from synth_ai.core.scanning.local_scanner import (
    scan_local_ports,
    scan_registry,
    scan_service_records,
)
from synth_ai.core.scanning.models import ScannedApp


def format_app_table(apps: list[ScannedApp]) -> str:
    """Format apps as a human-readable table."""
    if not apps:
        return "No active task apps found."

    name_width = max(len(app.name) for app in apps if app.name) if apps else 4
    port_width = 5
    status_width = 10
    type_width = max(len(app.type) for app in apps) if apps else 4
    via_width = max(len(app.discovered_via) for app in apps) if apps else 13

    app_id_width = max(len(app.app_id or "") for app in apps) if apps else 0
    app_id_width = max(app_id_width, 7)

    version_width = max(len(app.version or "") for app in apps) if apps else 0
    version_width = max(version_width, 7)

    name_width = max(name_width, 4)
    type_width = max(type_width, 4)
    via_width = max(via_width, 13)

    lines = [f"Found {len(apps)} active task app{'s' if len(apps) != 1 else ''}:\n"]

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

    total_width = sum(len(p) for p in header_parts) + len(header_parts) - 1
    lines.append("─" * total_width)

    for app in apps:
        status_icon = (
            "✅"
            if app.health_status == "healthy"
            else "⚠️ "
            if app.health_status == "unhealthy"
            else "❓"
        )
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
    verbose_callback: Callable[[str], None] | None = None,
) -> list[ScannedApp]:
    """Run the scan operation.

    Args:
        port_range: Tuple of (start_port, end_port)
        timeout: Health check timeout
        api_key: API key for health checks
        env_file: Specific .env file to check
        verbose_callback: Optional callback for verbose output

    Returns:
        List of discovered apps
    """
    start_port, end_port = port_range
    all_apps: list[ScannedApp] = []

    env_api_key = api_key
    if not env_api_key:
        try:
            from synth_ai.core.env_utils import resolve_env_var

            env_api_key = resolve_env_var("ENVIRONMENT_API_KEY")
        except Exception:
            env_api_key = os.getenv("ENVIRONMENT_API_KEY")

    if verbose_callback:
        verbose_callback(f"Scanning ports {start_port}-{end_port}...")

    local_apps = await scan_local_ports(start_port, end_port, env_api_key, timeout)
    all_apps.extend(local_apps)

    if verbose_callback:
        verbose_callback(f"Found {len(local_apps)} local app(s)")

    synth_api_key = os.getenv("SYNTH_API_KEY")
    cloudflare_apps = await scan_cloudflare_apps(synth_api_key, env_api_key, env_file, timeout)
    all_apps.extend(cloudflare_apps)

    if verbose_callback:
        verbose_callback(f"Found {len(cloudflare_apps)} Cloudflare app(s)")

    service_record_apps = await scan_service_records(env_api_key, timeout)
    all_apps.extend(service_record_apps)

    if verbose_callback:
        verbose_callback(f"Found {len(service_record_apps)} service record(s)")

    registry_apps = scan_registry()
    discovered_app_ids = {app.app_id for app in all_apps if app.app_id}
    new_registry_apps = [
        app for app in registry_apps if app.app_id and app.app_id not in discovered_app_ids
    ]
    all_apps.extend(new_registry_apps)

    if verbose_callback:
        verbose_callback(f"Found {len(new_registry_apps)} registry app(s) not yet running")

    # Deduplicate apps by URL
    seen_urls: dict[str, ScannedApp] = {}
    for app in all_apps:
        if app.url and app.url in seen_urls:
            existing = seen_urls[app.url]

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

            if app_priority < existing_priority or (
                app_priority == existing_priority
                and (
                    (app.health_status != "unknown" and existing.health_status == "unknown")
                    or (app.app_id and not existing.app_id)
                    or (app.discovered_via != "registry" and existing.discovered_via == "registry")
                )
            ):
                seen_urls[app.url] = app
        elif app.url:
            seen_urls[app.url] = app
        elif app.discovered_via == "registry":
            seen_urls[f"registry:{app.app_id}"] = app

    return list(seen_urls.values())
