"""Local app discovery for scan command."""

from __future__ import annotations

import asyncio
import socket

from synth_ai.cli.commands.scan.health_checker import check_app_health, extract_app_info
from synth_ai.cli.commands.scan.models import ScannedApp


async def scan_local_ports(
    start_port: int,
    end_port: int,
    api_key: str | None,
    timeout: float = 2.0,
    max_concurrent: int = 20,
) -> list[ScannedApp]:
    """Scan local ports for running task apps.

    Scans a range of local ports to discover running task applications by:
    1. Finding open ports using socket connections
    2. Checking each open port for HTTP responses
    3. Performing health checks on `/health` endpoints
    4. Extracting metadata from `/info` endpoints

    The function checks both `localhost` and `127.0.0.1` for each port, but
    only adds an app once if found on either hostname.

    Args:
        start_port: Start of port range (inclusive)
        end_port: End of port range (inclusive)
        api_key: API key for health checks via X-API-Key header. If None,
            requests are made without authentication.
        timeout: Health check timeout in seconds per request. Default is 2.0.
        max_concurrent: Maximum number of concurrent port checks and health
            checks. Default is 20. Increase for faster scanning, decrease to
            reduce resource usage.

    Returns:
        List of ScannedApp instances discovered via port scanning. Only apps
        that respond to health checks (status != "unknown") are included.

    Examples:
        >>> import asyncio
        >>> apps = asyncio.run(scan_local_ports(8000, 8100, "api_key"))
        >>> for app in apps:
        ...     print(f"{app.name} on port {app.port}: {app.health_status}")

    Note:
        Ports that don't respond to HTTP requests or don't have `/health`
        endpoints are not included in the results.
    """
    # First, find open ports
    open_ports = await _find_open_ports(start_port, end_port, max_concurrent)

    if not open_ports:
        return []

    # Check each open port for task app health
    apps: list[ScannedApp] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_port(port: int) -> None:
        async with semaphore:
            for base_url in [f"http://localhost:{port}", f"http://127.0.0.1:{port}"]:
                try:
                    health_status, metadata = await check_app_health(base_url, api_key, timeout)
                    # Only add if it's actually a task app (healthy or responds to /health)
                    if health_status != "unknown":
                        app_id, task_name, dataset_id, version = extract_app_info(metadata)
                        name = app_id or task_name or f"localhost:{port}"

                        apps.append(
                            ScannedApp(
                                name=name,
                                url=base_url,
                                type="local",
                                health_status=health_status,
                                port=port,
                                tunnel_mode=None,
                                tunnel_hostname=None,
                                app_id=app_id,
                                task_name=task_name,
                                dataset_id=dataset_id,
                                version=version,
                                metadata=metadata,
                                discovered_via="port_scan",
                            )
                        )
                        break  # Found on this URL, don't check the other
                except Exception:
                    continue

    await asyncio.gather(*[check_port(port) for port in open_ports], return_exceptions=True)

    return apps


async def _find_open_ports(start_port: int, end_port: int, max_concurrent: int = 20) -> list[int]:
    """Find open ports in the specified range.

    Uses socket.connect_ex() to check if ports are accepting connections.
    Checks both `127.0.0.1` and `localhost` for each port, but only adds
    each port once to the result list.

    Args:
        start_port: Start of port range (inclusive)
        end_port: End of port range (inclusive)
        max_concurrent: Maximum concurrent port checks. Default is 20.

    Returns:
        Sorted list of open port numbers. Empty list if no ports are open.

    Note:
        This function only checks if ports are accepting connections, not
        whether they're serving HTTP or task apps. Use scan_local_ports()
        for full task app discovery.
    """
    open_ports: list[int] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_port(port: int) -> None:
        async with semaphore:
            for host in ["127.0.0.1", "localhost"]:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.5)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    if result == 0:
                        open_ports.append(port)
                        break
                except Exception:
                    continue

    ports_to_check = list(range(start_port, end_port + 1))
    await asyncio.gather(*[check_port(port) for port in ports_to_check], return_exceptions=True)

    return sorted(set(open_ports))


def scan_registry() -> list[ScannedApp]:
    """Scan task app registry for registered apps.

    Queries the task app registry to find registered task applications.
    These apps may not have active deployments, so they're marked with
    "unknown" health status and empty URLs.

    Returns:
        List of ScannedApp instances from registry. Each app has:
        - name: App ID from registry
        - url: Empty string (no active deployment)
        - health_status: "unknown" (not checked)
        - discovered_via: "registry"
        - metadata: Registry entry details (description, aliases)

    Note:
        Registry apps are typically only included if they weren't already
        discovered via other methods (port scan, service records, etc.)
        to avoid duplicates.
    """
    apps: list[ScannedApp] = []

    try:
        from synth_ai.sdk.task.apps import registry

        for entry in registry.list():
            # Try to construct a URL - we don't know the port, so we'll mark it as unknown
            apps.append(
                ScannedApp(
                    name=entry.app_id,
                    url="",  # Unknown URL from registry alone
                    type="local",
                    health_status="unknown",
                    port=None,
                    tunnel_mode=None,
                    tunnel_hostname=None,
                    app_id=entry.app_id,
                    task_name=None,
                    dataset_id=None,
                    version=None,
                    metadata={"description": entry.description, "aliases": list(entry.aliases)},
                    discovered_via="registry",
                )
            )
    except Exception:
        pass

    return apps


async def scan_service_records(
    api_key: str | None,
    timeout: float = 2.0,
) -> list[ScannedApp]:
    """Scan service records file for local services and check their health.

    Reads persistent service records from `~/.synth-ai/services.json` (created
    when deploying local services via `synth-ai deploy --runtime local`),
    performs health checks on each service, and returns ScannedApp instances.

    The function:
    1. Cleans up stale records (processes that are no longer running)
    2. Loads active service records
    3. Performs health checks on each service
    4. Extracts metadata from `/info` endpoints
    5. Merges record metadata with endpoint metadata

    Args:
        api_key: API key for health checks via X-API-Key header. If None,
            requests are made without authentication.
        timeout: Health check timeout in seconds. Default is 2.0.

    Returns:
        List of ScannedApp instances from service records. Each app includes:
        - Full metadata from both records and `/info` endpoint
        - Health status from health check
        - App ID from metadata (preferred) or record
        - discovered_via: "service_records"

    Examples:
        >>> import asyncio
        >>> apps = asyncio.run(scan_service_records("api_key"))
        >>> for app in apps:
        ...     print(f"{app.name} ({app.port}): {app.health_status}")

    Note:
        Only local services (type="local") are included. Tunnel services
        are handled by scan_cloudflare_apps().
    """
    apps: list[ScannedApp] = []

    try:
        from synth_ai.cli.commands.scan.health_checker import check_app_health, extract_app_info
        from synth_ai.cli.lib.tunnel_records import cleanup_stale_records, load_service_records
        
        # Clean up stale records first
        cleanup_stale_records()
        
        records = load_service_records()
        for _, record in records.items():
            service_type = record.get("type", "local")
            if service_type == "local":
                url = record.get("url")
                port = record.get("port")
                app_id = record.get("app_id")
                
                if url and port:
                    # Check health
                    health_status, metadata = await check_app_health(url, api_key, timeout)
                    record_app_id, task_name, dataset_id, version = extract_app_info(metadata)
                    
                    # Use app_id from metadata if available, otherwise from record
                    final_app_id = record_app_id or app_id
                    
                    apps.append(
                        ScannedApp(
                            name=final_app_id or task_name or f"localhost:{port}",
                            url=url,
                            type="local",
                            health_status=health_status,
                            port=port,
                            tunnel_mode=None,
                            tunnel_hostname=None,
                            app_id=final_app_id,
                            task_name=task_name,
                            dataset_id=dataset_id,
                            version=version,
                            metadata={**record, **metadata},
                            discovered_via="service_records",
                        )
                    )
    except Exception:
        pass

    return apps

