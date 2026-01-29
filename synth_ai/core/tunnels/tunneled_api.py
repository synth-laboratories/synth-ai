"""High-level tunnel management for exposing local APIs.

This module provides a clean abstraction for setting up Cloudflare tunnels
to expose local APIs to the internet.

Example:
    from synth_ai.core.tunnels import TunneledLocalAPI, TunnelBackend

    # Default: Lease-based managed tunnel (fast, reusable)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        api_key="sk_live_...",
    )

    # Quick tunnel (random subdomain, no API key needed)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.CloudflareQuickTunnel,
    )

    print(f"Local API exposed at: {tunnel.url}")

    # Use the URL for remote jobs
    job = PromptLearningJob.from_dict(
        config_dict={...},
        task_app_url=tunnel.url,
    )

    # Clean up when done
    tunnel.close()

See Also:
    - `synth_ai.core.tunnels`: Lower-level tunnel functions
    - `synth_ai.core.tunnels.cloudflare`: Core tunnel implementation
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TunnelBackend(str, Enum):
    """Supported tunnel backends for exposing local APIs.

    Attributes:
        CloudflareManagedLease: NEW - Lease-based managed tunnel (RECOMMENDED).
            - Stable hostnames that persist across sessions
            - Fast reconnection (~1-5s after first run)
            - Automatic tunnel reuse without reprovisioning
            - Requires Synth API key
            - Best for production use

        CloudflareManagedTunnel: Legacy managed tunnel via Synth backend.
            - Creates new tunnel each time (slower)
            - Stable subdomains (e.g., task-1234-5678.usesynth.ai)
            - Requires Synth API key
            - Use CloudflareManagedLease instead for better performance

        CloudflareQuickTunnel: Anonymous tunnel via trycloudflare.com.
            - Random subdomains that change each time
            - No API key required
            - Not associated with any organization
            - Best for quick local testing

        Localhost: No tunnel, use localhost directly.
            - Uses http://localhost:{port}
            - No API key required
            - Best for local backend development
    """

    CloudflareManagedLease = "cloudflare_managed_lease"  # NEW - recommended
    CloudflareManagedTunnel = "cloudflare_managed"  # Legacy
    CloudflareQuickTunnel = "cloudflare_quick"
    Localhost = "localhost"


@dataclass
class TunneledLocalAPI:
    """A managed tunnel exposing a local API to the internet.

    This class provides a clean interface for:
    1. Provisioning a Cloudflare tunnel (managed or quick)
    2. Starting the cloudflared process
    3. Verifying DNS resolution and connectivity
    4. Tracking the process for cleanup

    Use `TunneledLocalAPI.create()` with a `TunnelBackend` to provision a tunnel.

    Attributes:
        url: Public HTTPS URL for the tunnel (e.g., "https://task-1234-5678.usesynth.ai")
        hostname: Hostname without protocol (e.g., "task-1234-5678.usesynth.ai")
        local_port: Local port being tunneled
        backend: The tunnel backend used (CloudflareManagedTunnel or CloudflareQuickTunnel)
        process: The cloudflared subprocess (for advanced use)

    Example:
        >>> from synth_ai.core.tunnels import TunneledLocalAPI
        >>> tunnel = await TunneledLocalAPI.create(
        ...     local_port=8001,
        ...     api_key="sk_live_...",
        ... )
        >>> print(tunnel.url)
        https://mt-xxxx.usesynth.ai/s/abc123
        >>> tunnel.close()
    """

    url: str
    hostname: str
    local_port: int
    backend: TunnelBackend
    process: Optional[Any] = None
    tunnel_token: Optional[str] = field(default=None, repr=False)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)
    # New lease-based fields
    _lease_id: Optional[str] = field(default=None, repr=False)
    _manager: Any = field(default=None, repr=False)  # Reserved for future use
    _handle: Any = field(default=None, repr=False)  # Rust handle for lease-based tunnels

    @classmethod
    async def create(
        cls,
        local_port: int,
        backend: TunnelBackend = TunnelBackend.CloudflareManagedLease,
        *,
        api_key: Optional[str] = None,
        env_api_key: Optional[str] = None,
        backend_url: Optional[str] = None,
        verify_dns: bool = True,
        progress: bool = False,
        reason: str | None = None,
    ) -> TunneledLocalAPI:
        """Create a tunnel to expose a local API.

        This is the main entry point for creating tunnels. It handles:
        1. Requesting a tunnel (from Synth backend for managed, or trycloudflare for quick)
        2. Starting the cloudflared process
        3. Waiting for DNS propagation
        4. Verifying HTTP connectivity
        5. Registering for automatic cleanup

        Args:
            local_port: Local port to tunnel (e.g., 8001)
            backend: Tunnel backend to use. Defaults to CloudflareManagedLease.
                - CloudflareManagedLease: Fast, reusable tunnels (recommended)
                - CloudflareManagedTunnel: Legacy managed tunnel (slower)
                - CloudflareQuickTunnel: Random subdomain, no api_key needed
            api_key: Synth API key for authentication (required for managed tunnels).
                If not provided, will be read from SYNTH_API_KEY environment variable.
            env_api_key: API key for the local task app (for health checks).
                Defaults to ENVIRONMENT_API_KEY env var.
            backend_url: Optional backend URL (defaults to production, managed only)
            verify_dns: Whether to verify DNS resolution after creating tunnel.
                Set to False if you're sure DNS will work (e.g., reusing subdomain).
            progress: If True, print status updates during setup

        Returns:
            TunneledLocalAPI instance with .url, .hostname, .close(), etc.

        Raises:
            ValueError: If api_key is missing for managed tunnels
            RuntimeError: If tunnel creation or verification fails
        """
        import os

        # Auto-detect API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("SYNTH_API_KEY")

        from synth_ai.sdk.localapi.auth import ensure_localapi_auth

        if backend == TunnelBackend.Localhost:
            url = f"http://localhost:{local_port}"
            return cls(
                url=url,
                hostname="localhost",
                local_port=local_port,
                backend=backend,
                process=None,
                tunnel_token=None,
                _raw={},
            )

        # Resolve env_api_key from environment if not provided
        if env_api_key is None:
            env_api_key = ensure_localapi_auth(
                backend_base=backend_url,
                synth_api_key=api_key,
            )

        return await cls._create_via_rust(
            local_port=local_port,
            backend=backend,
            api_key=api_key,
            env_api_key=env_api_key,
            backend_url=backend_url,
            verify_dns=verify_dns,
            progress=progress,
        )

    @classmethod
    async def _create_via_rust(
        cls,
        local_port: int,
        backend: TunnelBackend,
        api_key: Optional[str],
        env_api_key: Optional[str],
        backend_url: Optional[str],
        verify_dns: bool,
        progress: bool,
    ) -> TunneledLocalAPI:
        """Internal: Create a tunnel using Rust core."""
        import synth_ai_py

        backend_map = {
            TunnelBackend.CloudflareManagedLease: "cloudflare_managed_lease",
            TunnelBackend.CloudflareManagedTunnel: "cloudflare_managed",
            TunnelBackend.CloudflareQuickTunnel: "cloudflare_quick",
        }
        backend_key = backend_map.get(backend)
        if backend_key is None:
            raise ValueError(f"Unsupported tunnel backend: {backend}")

        # Quick tunnels have unreliable DNS verification - skip it
        if backend == TunnelBackend.CloudflareQuickTunnel:
            verify_dns = False

        handle = await asyncio.to_thread(
            synth_ai_py.tunnel_open,
            backend_key,
            local_port,
            api_key,
            backend_url,
            env_api_key,
            False,
            verify_dns,
            progress,
        )

        return cls(
            url=handle.url,
            hostname=handle.hostname,
            local_port=local_port,
            backend=backend,
            process=None,
            tunnel_token=None,
            _raw={
                "lease_id": getattr(handle, "lease_id", None),
                "process_id": getattr(handle, "process_id", None),
            },
            _lease_id=getattr(handle, "lease_id", None),
            _manager=None,
            _handle=handle,
        )

    def close(self) -> None:
        """Close the tunnel and terminate the cloudflared process.

        This is called automatically when the process exits (via atexit),
        but you can call it explicitly for earlier cleanup.
        """
        logger.info("[TUNNELED_API] close() called")
        if self._handle:
            try:
                self._handle.close()
            except Exception as e:
                logger.warning("[TUNNELED_API] Close failed: %s", e)
            self._handle = None
            self._lease_id = None
        elif self.process:
            try:
                import synth_ai_py

                synth_ai_py.stop_tunnel(self.process)
            except Exception:
                with contextlib.suppress(Exception):
                    self.process.terminate()
            self.process = None
        else:
            logger.debug("[TUNNELED_API] close() - nothing to close")

    def __enter__(self) -> TunneledLocalAPI:
        """Context manager entry (for sync use after async creation)."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - closes tunnel."""
        self.close()

    @classmethod
    async def create_for_app(
        cls,
        app: Any,
        local_port: int | None = None,
        backend: TunnelBackend = TunnelBackend.CloudflareManagedLease,
        *,
        api_key: Optional[str] = None,
        backend_url: Optional[str] = None,
        verify_dns: bool = True,
        progress: bool = False,
    ) -> TunneledLocalAPI:
        """Create a tunnel for a FastAPI/ASGI app, handling server startup automatically.

        This is a convenience method that:
        1. Finds an available port (or uses the provided one)
        2. Kills any process using that port
        3. Starts the app server in the background
        4. Waits for health check
        5. Creates and returns the tunnel

        Args:
            app: FastAPI or ASGI application to tunnel
            local_port: Port to use (defaults to finding an available port starting from 8001)
            backend: Tunnel backend to use
            api_key: Synth API key (defaults to SYNTH_API_KEY env var)
            backend_url: Backend URL (defaults to production)
            verify_dns: Whether to verify DNS resolution
            progress: If True, print status updates

        Returns:
            TunneledLocalAPI instance with .url, .hostname, .close(), etc.

        Example:
            >>> from fastapi import FastAPI
            >>> app = FastAPI()
            >>> tunnel = await TunneledLocalAPI.create_for_app(app)
            >>> print(f"App exposed at: {tunnel.url}")
        """
        import os

        from synth_ai.sdk.localapi._impl.server import run_server_background

        from .rust import find_available_port, kill_port, wait_for_health_check

        if api_key is None:
            api_key = os.environ.get("SYNTH_API_KEY") or None

        # Find or use port
        if local_port is None:
            local_port = find_available_port(8001)
            if progress:
                print(f"Auto-selected port: {local_port}")
        else:
            # Kill any process using the port
            kill_port(local_port)

        # Start the server
        if progress:
            print(f"Starting server on port {local_port}...")
        run_server_background(app, local_port)

        # Wait for health check
        if progress:
            print("Waiting for server health check...")
        await wait_for_health_check("localhost", local_port, timeout=30.0)

        # Create tunnel
        return await cls.create(
            local_port=local_port,
            backend=backend,
            api_key=api_key,
            backend_url=backend_url,
            verify_dns=verify_dns,
            progress=progress,
        )


__all__ = ["TunneledLocalAPI", "TunnelBackend"]
