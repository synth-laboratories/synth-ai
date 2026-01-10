"""Local app discovery for scanning."""

import asyncio
import socket

from synth_ai.core.scanning.health_checker import check_app_health, extract_app_info
from synth_ai.core.scanning.models import ScannedApp


async def scan_local_ports(
    start_port: int,
    end_port: int,
    api_key: str | None,
    timeout: float = 2.0,
    max_concurrent: int = 20,
) -> list[ScannedApp]:
    """Scan local ports for running task apps.

    Args:
        start_port: Start of port range (inclusive)
        end_port: End of port range (inclusive)
        api_key: API key for health checks
        timeout: Health check timeout in seconds
        max_concurrent: Maximum concurrent checks

    Returns:
        List of ScannedApp instances discovered via port scanning
    """
    open_ports = await _find_open_ports(start_port, end_port, max_concurrent)

    if not open_ports:
        return []

    apps: list[ScannedApp] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def check_port(port: int) -> None:
        async with semaphore:
            for base_url in [f"http://localhost:{port}", f"http://127.0.0.1:{port}"]:
                try:
                    health_status, metadata = await check_app_health(base_url, api_key, timeout)
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
                        break
                except Exception:
                    continue

    await asyncio.gather(*[check_port(port) for port in open_ports], return_exceptions=True)

    return apps


async def _find_open_ports(start_port: int, end_port: int, max_concurrent: int = 20) -> list[int]:
    """Find open ports in the specified range."""
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

    Returns:
        List of ScannedApp instances from registry
    """
    apps: list[ScannedApp] = []

    try:
        from synth_ai.sdk.task.apps import registry

        for entry in registry.list():
            apps.append(
                ScannedApp(
                    name=entry.app_id,
                    url="",
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

    Args:
        api_key: API key for health checks
        timeout: Health check timeout in seconds

    Returns:
        List of ScannedApp instances from service records
    """
    apps: list[ScannedApp] = []

    try:
        from synth_ai.core.service_records import cleanup_stale_records, load_service_records

        cleanup_stale_records()

        records = load_service_records()
        for _, record in records.items():
            service_type = record.get("type", "local")
            if service_type == "local":
                url = record.get("url")
                port = record.get("port")
                app_id = record.get("app_id")

                if url and port:
                    health_status, metadata = await check_app_health(url, api_key, timeout)
                    record_app_id, task_name, dataset_id, version = extract_app_info(metadata)

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
