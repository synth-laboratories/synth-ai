"""Cloudflare app discovery for scanning."""

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import httpx

from synth_ai.core.scanning.health_checker import check_app_health, extract_app_info
from synth_ai.core.scanning.models import ScannedApp
from synth_ai.core.urls import BACKEND_URL_BASE

_QUICK_TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


def scan_cloudflare_processes() -> list[dict[str, Any]]:
    """Scan running cloudflared processes to find tunnels.

    Returns:
        List of tunnel info dicts with keys: port, mode, pid, cmdline, url
    """
    tunnels: list[dict[str, Any]] = []

    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if proc.info["name"] and "cloudflared" in proc.info["name"].lower():
                    cmdline = proc.info.get("cmdline") or []
                    pid = proc.info.get("pid")

                    for i, arg in enumerate(cmdline):
                        if arg == "--url" and i + 1 < len(cmdline):
                            local_url = cmdline[i + 1]
                            port_match = re.search(r":(\d+)$", local_url)
                            if port_match:
                                port = int(port_match.group(1))
                                tunnel_url = _try_get_quick_tunnel_url(pid, port)
                                tunnels.append(
                                    {
                                        "port": port,
                                        "mode": "quick",
                                        "pid": pid,
                                        "cmdline": cmdline,
                                        "url": tunnel_url,
                                        "local_url": local_url,
                                    }
                                )
                            break

                        if arg == "run" and i > 0 and cmdline[i - 1] == "tunnel":
                            tunnels.append(
                                {
                                    "port": None,
                                    "mode": "managed",
                                    "pid": pid,
                                    "cmdline": cmdline,
                                    "url": None,
                                }
                            )
                            break
            except Exception:
                continue
    except ImportError:
        try:
            result = subprocess.run(
                ["pgrep", "-fl", "cloudflared"],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    parts = line.split(None, 1)
                    if len(parts) < 2:
                        continue
                    pid_str, rest = parts
                    try:
                        pid = int(pid_str)
                    except ValueError:
                        continue

                    url_match = re.search(r"--url\s+(\S+)", rest)
                    if url_match:
                        local_url = url_match.group(1)
                        port_match = re.search(r":(\d+)$", local_url)
                        if port_match:
                            port = int(port_match.group(1))
                            tunnel_url = _try_get_quick_tunnel_url(pid, port)
                            tunnels.append(
                                {
                                    "port": port,
                                    "mode": "quick",
                                    "pid": pid,
                                    "cmdline": rest.split(),
                                    "url": tunnel_url,
                                    "local_url": local_url,
                                }
                            )
                    elif "tunnel run" in rest or "tunnel run --token" in rest:
                        tunnels.append(
                            {
                                "port": None,
                                "mode": "managed",
                                "pid": pid,
                                "cmdline": rest.split(),
                                "url": None,
                            }
                        )
        except Exception:
            pass

    return tunnels


def _try_get_quick_tunnel_url(pid: int | None, port: int) -> str | None:
    """Try to get quick tunnel URL from process stdout/stderr or log files."""
    if pid is None:
        return None

    log_paths = [
        Path(f"/tmp/cloudflared_{port}.log"),
        Path.home() / f".cloudflared_{port}.log",
        Path(f"/tmp/cloudflared_{pid}.log"),
    ]

    for log_path in log_paths:
        try:
            if log_path.exists():
                with log_path.open("r") as f:
                    content = f.read()
                    match = _QUICK_TUNNEL_URL_RE.search(content)
                    if match:
                        return match.group(0)
        except Exception:
            continue

    try:
        import psutil

        proc = psutil.Process(pid)
        for fd in [proc.stdout, proc.stderr]:
            if fd and hasattr(fd, "read"):
                try:
                    content = fd.read(8192) if hasattr(fd, "read") else None
                    if content:
                        if isinstance(content, bytes):
                            content = content.decode("utf-8", errors="ignore")
                        match = _QUICK_TUNNEL_URL_RE.search(content)
                        if match:
                            return match.group(0)
                except Exception:
                    pass
    except Exception:
        pass

    return None


async def fetch_managed_tunnels(api_key: str | None) -> list[dict[str, Any]]:
    """Fetch active managed tunnels from backend API.

    Args:
        api_key: SYNTH_API_KEY for authentication

    Returns:
        List of tunnel metadata dicts
    """
    if not api_key:
        return []

    try:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=5.0) as client:
            url = f"{BACKEND_URL_BASE.rstrip('/')}/api/v1/tunnels/"
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return [t for t in data if isinstance(t, dict) and t.get("status") == "active"]
    except Exception:
        pass

    return []


def get_tunnel_processes() -> dict[int, Any]:
    """Get tunnel processes from global state."""
    try:
        from synth_ai.core.integrations.cloudflare import _TUNNEL_PROCESSES

        return _TUNNEL_PROCESSES.copy()
    except Exception:
        return {}


async def scan_cloudflare_apps(
    api_key: str | None,
    env_api_key: str | None,
    env_file: Path | None = None,
    timeout: float = 2.0,
) -> list[ScannedApp]:
    """Scan for Cloudflare tunnel apps using process-based discovery.

    Args:
        api_key: SYNTH_API_KEY for backend API
        env_api_key: ENVIRONMENT_API_KEY for health checks
        env_file: Not used (kept for API compatibility)
        timeout: Health check timeout

    Returns:
        List of discovered ScannedApp instances
    """
    apps: list[ScannedApp] = []
    seen_urls: set[str] = set()

    # Method 1: Check cloudflared processes
    process_tunnels = scan_cloudflare_processes()
    for tunnel_info in process_tunnels:
        tunnel_url = tunnel_info.get("url")
        port = tunnel_info.get("port")
        mode = tunnel_info.get("mode")
        pid = tunnel_info.get("pid")

        if mode == "quick" and tunnel_url and tunnel_url not in seen_urls:
            seen_urls.add(tunnel_url)
            health_status, metadata = await check_app_health(tunnel_url, env_api_key, timeout)
            app_id, task_name, dataset_id, version = extract_app_info(metadata)

            tunnel_hostname = tunnel_url.replace("https://", "").split("/")[0]
            name = app_id or task_name or tunnel_hostname or tunnel_url

            apps.append(
                ScannedApp(
                    name=name,
                    url=tunnel_url,
                    type="cloudflare",
                    health_status=health_status,
                    port=port,
                    tunnel_mode="quick",
                    tunnel_hostname=tunnel_hostname,
                    app_id=app_id,
                    task_name=task_name,
                    dataset_id=dataset_id,
                    version=version,
                    metadata={**metadata, "pid": pid, "local_port": port},
                    discovered_via="cloudflared_process",
                )
            )

    # Method 2: Query backend API for managed tunnels
    synth_api_key = api_key or os.getenv("SYNTH_API_KEY")
    managed_tunnels = await fetch_managed_tunnels(synth_api_key)
    for tunnel in managed_tunnels:
        hostname = tunnel.get("hostname")
        local_port = tunnel.get("local_port")
        if hostname:
            url = f"https://{hostname}"
            if url not in seen_urls:
                seen_urls.add(url)
                health_status, metadata = await check_app_health(url, env_api_key, timeout)
                app_id, task_name, dataset_id, version = extract_app_info(metadata)

                name = app_id or task_name or hostname.split(".")[0] or url

                apps.append(
                    ScannedApp(
                        name=name,
                        url=url,
                        type="cloudflare",
                        health_status=health_status,
                        port=local_port,
                        tunnel_mode="managed",
                        tunnel_hostname=hostname,
                        app_id=app_id,
                        task_name=task_name,
                        dataset_id=dataset_id,
                        version=version,
                        metadata={**metadata, **tunnel},
                        discovered_via="backend_api",
                    )
                )

    # Method 3: Check global tunnel processes state
    tunnel_procs = get_tunnel_processes()
    for port, proc in tunnel_procs.items():
        if proc and proc.poll() is None:
            tunnel_url = _try_get_quick_tunnel_url(proc.pid, port)
            if tunnel_url and tunnel_url not in seen_urls:
                seen_urls.add(tunnel_url)
                health_status, metadata = await check_app_health(tunnel_url, env_api_key, timeout)
                app_id, task_name, dataset_id, version = extract_app_info(metadata)

                tunnel_hostname = tunnel_url.replace("https://", "").split("/")[0]
                name = app_id or task_name or tunnel_hostname or tunnel_url

                apps.append(
                    ScannedApp(
                        name=name,
                        url=tunnel_url,
                        type="cloudflare",
                        health_status=health_status,
                        port=port,
                        tunnel_mode="quick",
                        tunnel_hostname=tunnel_hostname,
                        app_id=app_id,
                        task_name=task_name,
                        dataset_id=dataset_id,
                        version=version,
                        metadata={**metadata, "pid": proc.pid},
                        discovered_via="cloudflared_process",
                    )
                )

    # Method 4: Check service records file
    try:
        from synth_ai.core.service_records import cleanup_stale_records, load_service_records

        cleanup_stale_records()

        records = load_service_records()
        records = {k: v for k, v in records.items() if v.get("type") == "tunnel"}
        for _, record in records.items():
            tunnel_url = record.get("url")
            port = record.get("port")
            mode = record.get("mode", "quick")
            hostname = record.get("hostname")

            if tunnel_url and tunnel_url not in seen_urls:
                seen_urls.add(tunnel_url)
                health_status, metadata = await check_app_health(tunnel_url, env_api_key, timeout)
                app_id, task_name, dataset_id, version = extract_app_info(metadata)

                if not hostname and tunnel_url.startswith("https://"):
                    hostname = tunnel_url.replace("https://", "").split("/")[0]

                name = app_id or task_name or hostname or tunnel_url

                apps.append(
                    ScannedApp(
                        name=name,
                        url=tunnel_url,
                        type="cloudflare",
                        health_status=health_status,
                        port=port,
                        tunnel_mode=mode,
                        tunnel_hostname=hostname,
                        app_id=app_id,
                        task_name=task_name,
                        dataset_id=dataset_id,
                        version=version,
                        metadata={**metadata, **record},
                        discovered_via="tunnel_records",
                    )
                )
    except Exception:
        pass

    return apps
