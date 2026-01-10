"""Data structures for scanning."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ScannedApp:
    """Information about a discovered task app.

    Represents a single task application discovered during scanning, including
    its location (URL), health status, metadata, and how it was discovered.
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
