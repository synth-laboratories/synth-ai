"""Tunnel and local service record management for tracking deployed services."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_ai.core.user_config import CONFIG_DIR


def _get_records_path() -> Path:
    """Get the path to service records file."""
    return CONFIG_DIR / "services.json"


def record_service(
    url: str,
    port: int,
    service_type: str,
    pid: int | None = None,
    hostname: str | None = None,
    local_host: str = "127.0.0.1",
    task_app_path: str | None = None,
    app_id: str | None = None,
    mode: str | None = None,  # For tunnels: "quick" or "managed"
) -> None:
    """Record a service deployment (tunnel or local).

    Args:
        url: Service URL (public tunnel URL or local URL)
        port: Local port
        service_type: "tunnel" or "local"
        pid: Process ID (optional)
        hostname: Tunnel hostname (optional, for tunnels)
        local_host: Local host (default: 127.0.0.1)
        task_app_path: Path to task app file (optional)
        app_id: Task app ID (optional)
        mode: Tunnel mode - "quick" or "managed" (optional, for tunnels)
    """
    records_path = _get_records_path()
    
    # Load existing records
    records = load_service_records()
    
    # Create new record
    record: dict[str, Any] = {
        "url": url,
        "port": port,
        "type": service_type,
        "local_host": local_host,
        "created_at": datetime.now(UTC).isoformat(),
    }
    
    if pid is not None:
        record["pid"] = pid
    if hostname is not None:
        record["hostname"] = hostname
    if task_app_path is not None:
        record["task_app_path"] = str(task_app_path)
    if app_id is not None:
        record["app_id"] = app_id
    if mode is not None:
        record["mode"] = mode
    
    # Use port as key (one service per port)
    records[str(port)] = record
    
    # Write back
    try:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w") as f:
            json.dump(records, f, indent=2)
    except Exception:
        pass  # Fail silently - records are optional


def load_service_records() -> dict[str, dict[str, Any]]:
    """Load service records from disk.

    Returns:
        Dict mapping port (as string) -> record dict
    """
    records_path = _get_records_path()
    
    if not records_path.exists():
        return {}
    
    try:
        with records_path.open("r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    
    return {}


def remove_service_record(port: int) -> None:
    """Remove a service record.

    Args:
        port: Port of service to remove
    """
    records = load_service_records()
    port_str = str(port)
    
    if port_str in records:
        del records[port_str]
        
        records_path = _get_records_path()
        try:
            with records_path.open("w") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass


def cleanup_stale_records() -> None:
    """Remove records for services that are no longer running.
    
    Checks if processes are still alive and removes dead ones.
    Also checks if ports are still in use (for records without PID).
    """
    records = load_service_records()
    updated = False
    
    for port_str, record in list(records.items()):
        port = record.get("port")
        pid = record.get("pid")
        
        # Method 1: Check PID if available
        if pid is not None:
            try:
                import psutil  # type: ignore[import-untyped]
                proc = psutil.Process(pid)
                if not proc.is_running():
                    # Process is dead, remove record
                    del records[port_str]
                    updated = True
                    continue
            except ImportError:
                # psutil unavailable - skip PID check
                pass
            except Exception:
                # Process doesn't exist or access denied
                del records[port_str]
                updated = True
                continue
        
        # Method 2: Check if port is still in use (for records without PID or as fallback)
        if port is not None:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                sock.connect_ex(("127.0.0.1", port))
                sock.close()
                # If connection succeeds, port is in use (service might be running)
                # If connection fails, port might be free (but could also be a different service)
                # We'll be conservative and only remove if we're sure the process is dead
            except Exception:
                pass
    
    if updated:
        records_path = _get_records_path()
        try:
            with records_path.open("w") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass


# Backward compatibility aliases
def record_tunnel(
    url: str,
    port: int,
    mode: str,
    pid: int | None = None,
    hostname: str | None = None,
    local_host: str = "127.0.0.1",
    task_app_path: str | None = None,
) -> None:
    """Record a tunnel deployment (backward compatibility)."""
    record_service(
        url=url,
        port=port,
        service_type="tunnel",
        pid=pid,
        hostname=hostname,
        local_host=local_host,
        task_app_path=task_app_path,
        mode=mode,
    )


def load_tunnel_records() -> dict[str, dict[str, Any]]:
    """Load tunnel records (backward compatibility - returns all service records)."""
    return load_service_records()


def remove_tunnel_record(port: int) -> None:
    """Remove a tunnel record (backward compatibility)."""
    remove_service_record(port)

