"""Type definitions for the tunnel system.

This module defines the core data types used by the lease-based tunnel system.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class LeaseState(str, Enum):
    """State of a tunnel lease."""

    PENDING = "pending"
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    FAILED = "failed"


class ConnectorState(str, Enum):
    """State of the cloudflared connector."""

    STOPPED = "stopped"
    STARTING = "starting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class GatewayState(str, Enum):
    """State of the local gateway."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass(slots=True)
class LeaseInfo:
    """Information about an active tunnel lease."""

    lease_id: str
    managed_tunnel_id: str
    hostname: str
    route_prefix: str
    public_url: str
    local_host: str
    local_port: int
    expires_at: datetime
    tunnel_token: str
    access_client_id: Optional[str] = None
    access_client_secret: Optional[str] = None
    gateway_port: int = 8016
    state: LeaseState = LeaseState.PENDING
    diagnostics_hint: str = ""

    @property
    def is_active(self) -> bool:
        """Check if the lease is currently active."""
        return self.state in (LeaseState.PENDING, LeaseState.ACTIVE)

    @property
    def short_id(self) -> str:
        """Get a short version of the lease ID for display."""
        return self.lease_id[:8] if self.lease_id else ""


@dataclass(slots=True)
class ManagedTunnelInfo:
    """Information about a managed tunnel identity."""

    id: str
    hostname: str
    client_instance_id: str
    gateway_port: int
    health_status: str = "unknown"
    last_connected_at: Optional[datetime] = None


@dataclass(slots=True)
class ConnectorStatus:
    """Status of the cloudflared connector process."""

    state: ConnectorState
    pid: Optional[int] = None
    connected_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self.state == ConnectorState.CONNECTED


@dataclass(slots=True)
class GatewayStatus:
    """Status of the local gateway."""

    state: GatewayState
    port: int = 8016
    routes: dict[str, tuple[str, int]] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self.state == GatewayState.RUNNING


@dataclass(slots=True)
class TunnelHandle:
    """Handle to an active tunnel session.

    This is the primary interface for users to interact with tunnels.
    It provides:
    - The public URL for the tunnel
    - Methods to check health and close the tunnel
    - Access to the underlying lease and connector info
    """

    url: str
    hostname: str
    local_port: int
    lease: LeaseInfo
    connector: ConnectorStatus
    gateway: GatewayStatus

    # Internal state
    _closed: bool = False

    @property
    def is_ready(self) -> bool:
        """Check if the tunnel is fully ready for traffic."""
        return self.lease.is_active and self.connector.is_connected and self.gateway.is_running

    @property
    def route_prefix(self) -> str:
        """Get the route prefix for this tunnel."""
        return self.lease.route_prefix


@dataclass(slots=True)
class Diagnostics:
    """Diagnostic information for troubleshooting."""

    lease_id: str
    tunnel_id: str
    client_instance_id: str
    hostname: str
    connector_state: str
    gateway_state: str
    lease_state: str
    last_error: Optional[str] = None
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "lease_id": self.lease_id,
            "tunnel_id": self.tunnel_id,
            "client_instance_id": self.client_instance_id,
            "hostname": self.hostname,
            "connector_state": self.connector_state,
            "gateway_state": self.gateway_state,
            "lease_state": self.lease_state,
            "last_error": self.last_error,
            "logs": self.logs[-50:],  # Last 50 log entries
        }


try:  # Prefer Rust-backed classes when available
    import synth_ai_py as _rust_models  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for tunnel models.") from exc

with contextlib.suppress(AttributeError):
    LeaseInfo = _rust_models.LeaseInfo  # noqa: F811
    ConnectorStatus = _rust_models.ConnectorStatus  # noqa: F811
    GatewayStatus = _rust_models.GatewayStatus  # noqa: F811
    Diagnostics = _rust_models.Diagnostics  # noqa: F811
    TunnelHandle = _rust_models.TunnelHandle  # noqa: F811
