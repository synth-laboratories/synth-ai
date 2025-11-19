"""Data structures for scan command.

This module defines the core data structures used by the scan command to represent
discovered task applications and their metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ScannedApp:
    """Information about a discovered task app.

    Represents a single task application discovered during scanning, including
    its location (URL), health status, metadata, and how it was discovered.
    This structure is used both for internal processing and for output formatting
    (table and JSON).

    Attributes:
        name: App identifier, typically from /info endpoint (service.task.id or
            service.task.name), or fallback like "localhost:PORT"
        url: Full URL to access the app (http://localhost:PORT for local apps,
            https://tunnel-url for Cloudflare tunnels)
        type: Type of deployment - "local" for local servers, "cloudflare" for tunnels
        health_status: Health check result - "healthy" (200 OK with valid response),
            "unhealthy" (error or invalid response), or "unknown" (timeout or no response)
        port: Local port number (for local apps) or tunnel target port (for tunnels)
        tunnel_mode: Tunnel mode for Cloudflare apps - "quick" (ephemeral) or
            "managed" (stable). None for local apps.
        tunnel_hostname: Tunnel hostname for Cloudflare apps (e.g., "abc123.trycloudflare.com").
            None for local apps.
        app_id: Task app identifier from /info endpoint (service.task.id)
        task_name: Human-readable task name from /info endpoint (service.task.name)
        dataset_id: Dataset identifier from /info endpoint (dataset.id), if available
        version: App version from /info endpoint (service.task.version)
        discovered_via: Discovery method used to find this app. One of:
            - "port_scan": Found by scanning ports
            - "service_records": Found in persistent service records (local deployments)
            - "tunnel_records": Found in persistent tunnel records (tunnel deployments)
            - "cloudflared_process": Found by inspecting running cloudflared processes
            - "backend_api": Found by querying backend API (managed tunnels)
            - "registry": Found in task app registry (no active deployment)
        metadata: Additional metadata dictionary containing:
            - Full /info endpoint response (service, dataset, rubrics, inference, etc.)
            - Deployment information (PID, created_at, task_app_path, etc.)
            - Error information (if health check failed)
            - Any other context-specific data

    Examples:
        >>> app = ScannedApp(
        ...     name="banking77",
        ...     url="http://127.0.0.1:8000",
        ...     type="local",
        ...     health_status="healthy",
        ...     port=8000,
        ...     tunnel_mode=None,
        ...     tunnel_hostname=None,
        ...     app_id="banking77",
        ...     task_name="Banking77 Intent Classification",
        ...     dataset_id=None,
        ...     version="1.0.0",
        ...     discovered_via="service_records",
        ...     metadata={"pid": 12345, "created_at": "2025-01-01T00:00:00Z"}
        ... )
    """

    name: str
    url: str
    type: Literal["local", "cloudflare"]
    health_status: Literal["healthy", "unhealthy", "unknown"]
    port: int | None
    tunnel_mode: Literal["quick", "managed"] | None
    tunnel_hostname: str | None
    app_id: str | None
    task_name: str | None
    dataset_id: str | None
    version: str | None
    discovered_via: str
    metadata: dict[str, Any] = field(default_factory=dict)

