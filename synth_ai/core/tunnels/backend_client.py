"""HTTP client for the tunnel lease API.

This module provides a typed client for communicating with the backend's
lease-based tunnel API.

.. deprecated::
    This module is deprecated. Use `TunneledLocalAPI` instead, which uses
    the Rust core internally for better performance and reliability.
    Direct lease management will be removed in a future version.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from datetime import datetime
from typing import Any, Optional

from .errors import TunnelAPIError
from .types import LeaseInfo, LeaseState

try:
    import synth_ai_py  # type: ignore
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for tunnel leases.") from exc

logger = logging.getLogger(__name__)

# Default backend URL
DEFAULT_BACKEND_URL = "https://api.usesynth.ai"


class LeaseClient:
    """Client for the tunnel lease API.

    This client handles:
    - Creating/attaching to leases
    - Sending heartbeats
    - Releasing leases
    - Listing active leases

    Token safety: The client never logs tunnel tokens.
    """

    def __init__(
        self,
        api_key: str,
        *,
        backend_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize the lease client.

        Args:
            api_key: Synth API key for authentication
            backend_url: Backend URL (defaults to production)
            timeout: Request timeout in seconds

        .. deprecated::
            Use TunneledLocalAPI instead for tunnel management.
        """
        warnings.warn(
            "LeaseClient is deprecated. Use TunneledLocalAPI instead, which uses "
            "the Rust core internally for better performance and reliability.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.api_key = api_key
        self.backend_url = (
            backend_url or os.getenv("SYNTH_BACKEND_URL", DEFAULT_BACKEND_URL)
        ).rstrip("/")
        self.timeout = timeout
        self._rust = synth_ai_py.LeaseClient(self.api_key, self.backend_url, int(self.timeout))

    async def close(self) -> None:
        """Close the HTTP client."""
        return

    async def __aenter__(self) -> LeaseClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _coerce_lease_info(self, lease: Any) -> LeaseInfo:
        if isinstance(lease, LeaseInfo):
            return lease
        data = None
        if hasattr(lease, "to_dict"):
            data = lease.to_dict()
        elif isinstance(lease, dict):
            data = lease
        if not isinstance(data, dict):
            raise TunnelAPIError("Invalid lease payload returned from rust client.")
        state_value = data.get("state")
        if isinstance(state_value, LeaseState):
            state = state_value
        else:
            try:
                state = LeaseState(str(state_value))
            except Exception:
                state = LeaseState.PENDING
        expires_raw = data.get("expires_at") or ""
        if isinstance(expires_raw, str):
            expires_at = datetime.fromisoformat(expires_raw.replace("Z", "+00:00"))
        else:
            expires_at = datetime.utcnow()
        return LeaseInfo(
            lease_id=str(data.get("lease_id") or ""),
            managed_tunnel_id=str(data.get("managed_tunnel_id") or ""),
            hostname=str(data.get("hostname") or ""),
            route_prefix=str(data.get("route_prefix") or ""),
            public_url=str(data.get("public_url") or ""),
            local_host=str(data.get("local_host") or ""),
            local_port=int(data.get("local_port") or 0),
            expires_at=expires_at,
            tunnel_token=str(data.get("tunnel_token") or ""),
            access_client_id=data.get("access_client_id"),
            access_client_secret=data.get("access_client_secret"),
            gateway_port=int(data.get("gateway_port") or 8016),
            state=state,
            diagnostics_hint=str(data.get("diagnostics_hint") or ""),
        )

    async def create_lease(
        self,
        *,
        client_instance_id: str,
        local_host: str = "127.0.0.1",
        local_port: int,
        app_name: Optional[str] = None,
        requested_ttl_seconds: int = 3600,
        reuse_connector: bool = True,
        idempotency_key: Optional[str] = None,
    ) -> LeaseInfo:
        """Create or attach to a tunnel lease.

        Args:
            client_instance_id: Stable identifier for this machine
            local_host: Local host to forward to
            local_port: Local port to forward to
            app_name: Optional name for the app
            requested_ttl_seconds: Requested lease TTL
            reuse_connector: Whether to reuse existing connector
            idempotency_key: Optional key for idempotent requests

        Returns:
            LeaseInfo with tunnel details and token
        """
        lease = await asyncio.to_thread(
            self._rust.create_lease,
            client_instance_id,
            local_host,
            local_port,
            app_name,
            requested_ttl_seconds,
            reuse_connector,
            idempotency_key,
        )
        return self._coerce_lease_info(lease)

    async def heartbeat(
        self,
        lease_id: str,
        *,
        connected_to_edge: bool = False,
        gateway_ready: bool = False,
        local_ready: bool = False,
        last_error: Optional[str] = None,
    ) -> tuple[str, int]:
        """Send heartbeat for an active lease.

        Args:
            lease_id: The lease ID
            connected_to_edge: Whether connector is connected
            gateway_ready: Whether gateway is ready
            local_ready: Whether local app is ready
            last_error: Last error message (if any)

        Returns:
            Tuple of (action, next_heartbeat_seconds)
        """
        return await asyncio.to_thread(
            self._rust.heartbeat,
            lease_id,
            connected_to_edge,
            gateway_ready,
            local_ready,
            last_error,
        )

    async def release(self, lease_id: str) -> None:
        """Release an active lease.

        Args:
            lease_id: The lease ID to release
        """
        await asyncio.to_thread(self._rust.release, lease_id)
        logger.debug("[LEASE_CLIENT] Released lease: lease_id=%s", lease_id[:8])

    async def list_leases(
        self,
        *,
        client_instance_id: Optional[str] = None,
        include_expired: bool = False,
    ) -> list[LeaseInfo]:
        """List active leases.

        Note: This does NOT return tunnel tokens. Use create_lease to get tokens.

        Args:
            client_instance_id: Filter by client instance
            include_expired: Whether to include expired leases

        Returns:
            List of lease info (without tokens)
        """
        leases = await asyncio.to_thread(
            self._rust.list_leases,
            client_instance_id,
            include_expired,
        )
        return [self._coerce_lease_info(item) for item in leases]


def get_lease_client(
    api_key: Optional[str] = None,
    backend_url: Optional[str] = None,
) -> LeaseClient:
    """Get a lease client with default configuration.

    Args:
        api_key: Synth API key (defaults to SYNTH_API_KEY env var)
        backend_url: Backend URL (defaults to production)

    Returns:
        LeaseClient instance
    """
    key = api_key or os.getenv("SYNTH_API_KEY")
    if not key:
        raise TunnelAPIError(
            "No API key provided",
            hint="Set the SYNTH_API_KEY environment variable or pass api_key parameter.",
        )

    return LeaseClient(key, backend_url=backend_url)
