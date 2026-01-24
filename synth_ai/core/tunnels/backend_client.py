"""HTTP client for the tunnel lease API.

This module provides a typed client for communicating with the backend's
lease-based tunnel API.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

import httpx

from .errors import LeaseNotFoundError, TunnelAPIError
from .types import LeaseInfo, LeaseState

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
        """
        self.api_key = api_key
        self.backend_url = (
            backend_url or os.getenv("SYNTH_BACKEND_URL", DEFAULT_BACKEND_URL)
        ).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "synth-ai-sdk/1.0",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> LeaseClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make an API request."""
        import time

        client = await self._get_client()
        url = f"/api/v1/tunnels{path}"

        logger.debug("[LEASE_CLIENT] --> %s %s", method, url)
        start_time = time.monotonic()

        try:
            response = await client.request(
                method,
                url,
                json=json,
                params=params,
            )
            elapsed = time.monotonic() - start_time
            logger.debug(
                "[LEASE_CLIENT] <-- %s %s status=%d elapsed=%.3fs",
                method,
                url,
                response.status_code,
                elapsed,
            )
        except httpx.TimeoutException as e:
            elapsed = time.monotonic() - start_time
            logger.error(
                "[LEASE_CLIENT] <-- %s %s TIMEOUT elapsed=%.3fs",
                method,
                url,
                elapsed,
            )
            raise TunnelAPIError(
                f"Request timed out: {method} {url}",
                hint="The backend may be slow. Try again.",
            ) from e
        except httpx.RequestError as e:
            elapsed = time.monotonic() - start_time
            logger.error(
                "[LEASE_CLIENT] <-- %s %s ERROR=%s elapsed=%.3fs",
                method,
                url,
                type(e).__name__,
                elapsed,
            )
            raise TunnelAPIError(
                f"Request failed: {method} {url}: {e}",
                hint="Check your network connection.",
            ) from e

        if response.status_code == 404:
            raise LeaseNotFoundError(path.split("/")[-1] if "/" in path else "unknown")

        if response.status_code >= 400:
            try:
                body = response.json()
                detail = body.get("detail", response.text)
            except Exception:
                detail = response.text

            raise TunnelAPIError(
                f"API error: {detail}",
                status_code=response.status_code,
                response_body=response.text,
            )

        return response.json()

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
        data = await self._request(
            "POST",
            "/lease",
            json={
                "client_instance_id": client_instance_id,
                "local_host": local_host,
                "local_port": local_port,
                "app_name": app_name,
                "requested_ttl_seconds": requested_ttl_seconds,
                "reuse_connector": reuse_connector,
                "idempotency_key": idempotency_key,
            },
        )

        logger.debug(
            "[LEASE_CLIENT] Created lease: lease_id=%s hostname=%s route=%s",
            data["lease_id"][:8],
            data["hostname"],
            data["route_prefix"],
        )

        return LeaseInfo(
            lease_id=data["lease_id"],
            managed_tunnel_id=data["managed_tunnel_id"],
            hostname=data["hostname"],
            route_prefix=data["route_prefix"],
            public_url=data["public_url"],
            local_host=local_host,
            local_port=local_port,
            expires_at=datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")),
            tunnel_token=data["tunnel_token"],
            access_client_id=data.get("access_client_id"),
            access_client_secret=data.get("access_client_secret"),
            gateway_port=data.get("gateway_port", 8016),
            state=LeaseState.PENDING,
            diagnostics_hint=data.get("diagnostics_hint", ""),
        )

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
        data = await self._request(
            "POST",
            f"/lease/{lease_id}/heartbeat",
            json={
                "connected_to_edge": connected_to_edge,
                "gateway_ready": gateway_ready,
                "local_ready": local_ready,
                "last_error": last_error[:1000] if last_error else None,
            },
        )

        return data.get("action", "none"), data.get("next_heartbeat_seconds", 30)

    async def release(self, lease_id: str) -> None:
        """Release an active lease.

        Args:
            lease_id: The lease ID to release
        """
        await self._request("POST", f"/lease/{lease_id}/release")
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
        params: dict[str, Any] = {}
        if client_instance_id:
            params["client_instance_id"] = client_instance_id
        if include_expired:
            params["include_expired"] = "true"

        data = await self._request("GET", "/lease", params=params)

        return [
            LeaseInfo(
                lease_id=item["lease_id"],
                managed_tunnel_id=item["managed_tunnel_id"],
                hostname=item["hostname"],
                route_prefix=item["route_prefix"],
                public_url=item["public_url"],
                local_host="127.0.0.1",  # Not returned in list
                local_port=0,  # Not returned in list
                expires_at=datetime.fromisoformat(item["expires_at"].replace("Z", "+00:00")),
                tunnel_token="",  # Not returned in list
                gateway_port=item.get("gateway_port", 8016),
                state=LeaseState.ACTIVE,  # Assumed active if listed
                diagnostics_hint=item.get("diagnostics_hint", ""),
            )
            for item in data
        ]


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
