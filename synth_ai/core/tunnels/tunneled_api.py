"""High-level tunnel management for exposing local APIs.

This module provides a clean abstraction for setting up tunnels to expose
local task apps to the internet for use with Synth optimization jobs.

Two backends are available:

- **SynthTunnel** (default): Relay-based HTTPS tunnel via Synth's servers.
  No ``cloudflared`` binary needed. Supports 128 concurrent in-flight
  requests with dynamic memory-based budgeting. Uses ``worker_token``
  for job authentication.

- **Cloudflare**: Tunnel via Cloudflare's network. Requires ``cloudflared``.
  Uses ``task_app_api_key`` for job authentication.

Example — SynthTunnel (recommended):

    from synth_ai.core.tunnels import TunneledLocalAPI

    tunnel = await TunneledLocalAPI.create(local_port=8001, api_key="sk_live_...")
    print(tunnel.url)            # https://dev.st.usesynth.ai/s/rt_...
    print(tunnel.worker_token)   # pass this to your job config
    tunnel.close()

Example — Cloudflare quick tunnel:

    from synth_ai.core.tunnels import TunneledLocalAPI, TunnelBackend

    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.CloudflareQuickTunnel,
    )
    print(tunnel.url)  # https://random-words.trycloudflare.com
    tunnel.close()

Using with optimization jobs::

    # SynthTunnel
    job = PromptLearningJob.from_dict(config,
        task_app_url=tunnel.url,
        task_app_worker_token=tunnel.worker_token,
    )

    # Cloudflare
    job = PromptLearningJob.from_dict(config,
        task_app_url=tunnel.url,
        task_app_api_key=env_api_key,
    )

See Also:
    - `synth_ai.core.tunnels`: Module-level docs with comparison table
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

    **SynthTunnel vs Cloudflare:**

    Use **SynthTunnel** (default) for optimization jobs (GEPA, MiPRO).
    It requires no external binary, supports 128 concurrent in-flight
    requests with dynamic memory budgeting, and authenticates via
    ``worker_token``.

    Use **Cloudflare** when you need a stable subdomain that persists
    across sessions, or for quick anonymous testing without an API key.
    Requires the ``cloudflared`` binary and authenticates via
    ``task_app_api_key``.

    Attributes:
        SynthTunnel: Relay-based HTTPS tunnel (default, recommended).
            - No ``cloudflared`` binary required
            - Up to 128 concurrent in-flight requests (dynamic cap)
            - Authenticate jobs with ``worker_token``
            - URLs: ``https://st.usesynth.ai/s/rt_...``

        CloudflareManagedLease: Lease-based Cloudflare tunnel.
            - Stable hostnames that persist across sessions
            - Fast reconnection (~1-5s after first run)
            - Requires ``cloudflared`` and Synth API key
            - Authenticate jobs with ``task_app_api_key``

        CloudflareManagedTunnel: Legacy managed tunnel (use ManagedLease instead).

        CloudflareQuickTunnel: Anonymous Cloudflare tunnel.
            - Random subdomains via trycloudflare.com
            - No API key required
            - Subject to Cloudflare rate limits

        Localhost: No tunnel, use ``http://localhost:{port}`` directly.
    """

    SynthTunnel = "synthtunnel"
    CloudflareManagedLease = "cloudflare_managed_lease"  # NEW - recommended
    CloudflareManagedTunnel = "cloudflare_managed"  # Legacy
    CloudflareQuickTunnel = "cloudflare_quick"
    Localhost = "localhost"


@dataclass
class TunneledLocalAPI:
    """A managed tunnel exposing a local API to the internet.

    This class provides a clean interface for:
    1. Provisioning a tunnel (SynthTunnel or Cloudflare)
    2. Starting the cloudflared process
    3. Verifying DNS resolution and connectivity
    4. Tracking the process for cleanup

    Use `TunneledLocalAPI.create()` with a `TunnelBackend` to provision a tunnel.

    Attributes:
        url: Public HTTPS URL for the tunnel (e.g., "https://st.usesynth.ai/s/rt_...")
        hostname: Hostname without protocol (e.g., "task-1234-5678.usesynth.ai")
        local_port: Local port being tunneled
        backend: The tunnel backend used
        process: The cloudflared subprocess (for advanced use)
        worker_token: SynthTunnel worker token (if using SynthTunnel)

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
    worker_token: Optional[str] = field(default=None, repr=False)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)
    # New lease-based fields
    _lease_id: Optional[str] = field(default=None, repr=False)
    _manager: Any = field(default=None, repr=False)  # Reserved for future use
    _handle: Any = field(default=None, repr=False)  # Rust handle for lease-based tunnels
    _synth_session: Any = field(default=None, repr=False)

    @classmethod
    async def create(
        cls,
        local_port: int,
        backend: TunnelBackend = TunnelBackend.SynthTunnel,
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
            backend: Tunnel backend to use. Defaults to SynthTunnel.
                - SynthTunnel: Relay-based HTTPS tunnel (recommended)
                - CloudflareManagedLease: Fast, reusable tunnels (legacy)
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

        if backend == TunnelBackend.SynthTunnel:
            if not api_key:
                raise ValueError("api_key is required for SynthTunnel")
            if env_api_key is None:
                env_api_key = os.environ.get("ENVIRONMENT_API_KEY") or os.environ.get(
                    "DEV_ENVIRONMENT_API_KEY"
                )

            # Step 1: Create lease via Python (simple HTTP POST, works fine)
            from .synth_tunnel import (
                SynthTunnelClient,
                _collect_local_api_keys,
                get_client_instance_id,
                hostname_from_url,
            )

            client = SynthTunnelClient(api_key, backend_url=backend_url)
            lease = await client.create_lease(
                client_instance_id=get_client_instance_id(),
                local_host="127.0.0.1",
                local_port=local_port,
            )

            # Step 2: Start Rust WS agent (runs in its own tokio runtime)
            import synth_ai_py

            local_api_keys = _collect_local_api_keys(env_api_key)
            max_inflight = int(lease.limits.get("max_inflight", 128))
            agent = await asyncio.to_thread(
                synth_ai_py.synth_tunnel_start,
                lease.agent_url,
                lease.agent_token,
                lease.lease_id,
                "127.0.0.1",
                local_port,
                lease.public_url,
                lease.worker_token,
                local_api_keys,
                max_inflight,
            )

            url = lease.public_url
            return cls(
                url=url,
                hostname=hostname_from_url(url),
                local_port=local_port,
                backend=backend,
                process=None,
                tunnel_token=None,
                worker_token=lease.worker_token,
                _raw={
                    "lease_id": lease.lease_id,
                    "route_token": lease.route_token,
                    "agent_url": lease.agent_url,
                },
                _lease_id=lease.lease_id,
                _manager=None,
                _handle=None,
                _synth_session={"agent": agent, "client": client, "lease": lease},
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
        if self.backend == TunnelBackend.SynthTunnel and self._synth_session:
            session = self._synth_session
            try:
                # Stop Rust WS agent
                if isinstance(session, dict):
                    agent = session.get("agent")
                    if agent is not None:
                        agent.stop()
                    # Close lease via Python client
                    client = session.get("client")
                    lease = session.get("lease")
                    if client is not None and lease is not None:
                        try:
                            import asyncio

                            loop = asyncio.get_running_loop()
                            loop.create_task(client.close_lease(lease.lease_id))
                        except RuntimeError:
                            import asyncio

                            asyncio.run(client.close_lease(lease.lease_id))
                else:
                    # Legacy SynthTunnelSession fallback
                    session.close()
            except Exception as e:
                logger.warning("[TUNNELED_API] SynthTunnel close failed: %s", e)
            self._synth_session = None
            self._lease_id = None
            self.worker_token = None
        elif self._handle:
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
        backend: TunnelBackend = TunnelBackend.SynthTunnel,
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
            backend: Tunnel backend to use (defaults to SynthTunnel)
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
