"""High-level tunnel management for exposing local APIs.

This module provides a clean abstraction for setting up Cloudflare tunnels
to expose local APIs to the internet.

Example:
    from synth_ai.sdk.tunnels import TunneledLocalAPI, TunnelBackend

    # Managed tunnel (stable subdomain, requires API key)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.CloudflareManagedTunnel,
        api_key="sk_live_...",
        env_api_key="env_key_...",
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
    - `synth_ai.sdk.tunnels`: Lower-level tunnel functions
    - `synth_ai.core.integrations.cloudflare`: Core tunnel implementation
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from synth_ai.core.telemetry import log_info


class TunnelBackend(str, Enum):
    """Supported tunnel backends for exposing local APIs.

    Attributes:
        CloudflareManagedTunnel: Managed tunnel via Synth backend.
            - Stable subdomains (e.g., task-1234-5678.usesynth.ai)
            - Requires Synth API key
            - Associated with your organization
            - Best for production jobs

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

    CloudflareManagedTunnel = "cloudflare_managed"
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
        >>> from synth_ai.sdk.tunnels import TunneledLocalAPI, TunnelBackend
        >>> tunnel = await TunneledLocalAPI.create(
        ...     local_port=8001,
        ...     backend=TunnelBackend.CloudflareManagedTunnel,
        ...     api_key="sk_live_...",
        ...     env_api_key="env_key_...",
        ... )
        >>> print(tunnel.url)
        https://task-1234-5678.usesynth.ai
        >>> tunnel.close()
    """

    url: str
    hostname: str
    local_port: int
    backend: TunnelBackend
    process: Optional[subprocess.Popen] = None
    tunnel_token: Optional[str] = field(default=None, repr=False)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    async def create(
        cls,
        local_port: int,
        backend: TunnelBackend = TunnelBackend.CloudflareManagedTunnel,
        *,
        api_key: Optional[str] = None,
        env_api_key: Optional[str] = None,
        backend_url: Optional[str] = None,
        verify_dns: bool = True,
        progress: bool = False,
    ) -> "TunneledLocalAPI":
        """Create a tunnel to expose a local API.

        This is the main entry point for creating tunnels. It handles:
        1. Requesting a tunnel (from Synth backend for managed, or trycloudflare for quick)
        2. Starting the cloudflared process
        3. Waiting for DNS propagation
        4. Verifying HTTP connectivity
        5. Registering for automatic cleanup

        Args:
            local_port: Local port to tunnel (e.g., 8001)
            backend: Tunnel backend to use. Defaults to CloudflareManagedTunnel.
                - CloudflareManagedTunnel: Stable subdomain, requires api_key
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
        
        from .cleanup import track_process
        from synth_ai.sdk.localapi.auth import ensure_localapi_auth

        if backend == TunnelBackend.Localhost:
            url = f"http://localhost:{local_port}"
            log_info(
                "TunneledLocalAPI.create: localhost passthrough",
                ctx={"local_port": local_port},
            )
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

        if backend == TunnelBackend.CloudflareManagedTunnel:
            return await cls._create_managed(
                local_port=local_port,
                api_key=api_key,
                env_api_key=env_api_key,
                backend_url=backend_url,
                verify_dns=verify_dns,
                progress=progress,
                track_process=track_process,
            )
        elif backend == TunnelBackend.CloudflareQuickTunnel:
            return await cls._create_quick(
                local_port=local_port,
                env_api_key=env_api_key,
                progress=progress,
                track_process=track_process,
            )
        else:
            raise ValueError(f"Unsupported tunnel backend: {backend}")

    @classmethod
    async def _create_managed(
        cls,
        local_port: int,
        api_key: Optional[str],
        env_api_key: Optional[str],
        backend_url: Optional[str],
        verify_dns: bool,
        progress: bool,
        track_process,
    ) -> "TunneledLocalAPI":
        """Internal: Create a managed tunnel via Synth backend."""
        from synth_ai.core.integrations.cloudflare import (
            open_managed_tunnel_with_connection_wait,
            rotate_tunnel,
            verify_tunnel_dns_resolution,
        )

        if not api_key:
            raise ValueError(
                "api_key is required for CloudflareManagedTunnel. "
                "Use CloudflareQuickTunnel for anonymous tunnels."
            )

        # Step 1: Provision tunnel from backend
        if progress:
            print(f"Provisioning managed tunnel for port {local_port}...")

        tunnel_data = await rotate_tunnel(
            api_key,
            local_port,
            backend_url=backend_url,
        )

        hostname = tunnel_data["hostname"]
        tunnel_token = tunnel_data["tunnel_token"]
        url = f"https://{hostname}"

        log_info(
            "TunneledLocalAPI.create: managed tunnel provisioned",
            ctx={"hostname": hostname, "local_port": local_port},
        )

        # Step 2: Start cloudflared and WAIT for it to connect
        # This is critical - DNS only resolves after cloudflared connects to Cloudflare's edge
        if progress:
            print(f"Starting cloudflared for {hostname}...")
            print("Waiting for cloudflared to connect to Cloudflare edge...")

        proc = await open_managed_tunnel_with_connection_wait(
            tunnel_token,
            timeout_seconds=30.0,
        )
        track_process(proc)

        log_info(
            "TunneledLocalAPI.create: cloudflared connected",
            ctx={"hostname": hostname},
        )

        # Step 3: Verify DNS resolution and connectivity (if requested)
        # DNS should now resolve quickly since cloudflared is connected
        if verify_dns:
            dns_verified = tunnel_data.get("dns_verified", False)
            if not dns_verified:
                if progress:
                    print("Verifying DNS propagation...")

                await verify_tunnel_dns_resolution(
                    url,
                    name="tunnel",
                    timeout_seconds=60.0,  # Reduced from 90s since cloudflared is already connected
                    api_key=env_api_key,
                )

        if progress:
            print(f"Tunnel ready: {url}")

        return cls(
            url=url,
            hostname=hostname,
            local_port=local_port,
            backend=TunnelBackend.CloudflareManagedTunnel,
            process=proc,
            tunnel_token=tunnel_token,
            _raw=tunnel_data,
        )

    @classmethod
    async def _create_quick(
        cls,
        local_port: int,
        env_api_key: Optional[str],
        progress: bool,
        track_process,
    ) -> "TunneledLocalAPI":
        """Internal: Create a quick (anonymous) tunnel via trycloudflare.com."""
        from synth_ai.core.integrations.cloudflare import (
            open_quick_tunnel_with_dns_verification,
        )

        if progress:
            print(f"Starting quick tunnel for port {local_port}...")

        url, proc = await open_quick_tunnel_with_dns_verification(
            port=local_port,
            api_key=env_api_key,
        )

        track_process(proc)

        # Extract hostname from URL
        hostname = url.replace("https://", "").replace("http://", "").rstrip("/")

        log_info(
            "TunneledLocalAPI.create: quick tunnel ready",
            ctx={"hostname": hostname, "local_port": local_port},
        )

        if progress:
            print(f"Tunnel ready: {url}")

        return cls(
            url=url,
            hostname=hostname,
            local_port=local_port,
            backend=TunnelBackend.CloudflareQuickTunnel,
            process=proc,
            tunnel_token=None,
            _raw={},
        )

    def close(self) -> None:
        """Close the tunnel and terminate the cloudflared process.

        This is called automatically when the process exits (via atexit),
        but you can call it explicitly for earlier cleanup.
        """
        from synth_ai.core.integrations.cloudflare import stop_tunnel

        if self.process:
            stop_tunnel(self.process)
            self.process = None
            log_info(
                "TunneledLocalAPI.close: tunnel closed",
                ctx={"hostname": self.hostname, "backend": self.backend.value},
            )

    def __enter__(self) -> "TunneledLocalAPI":
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
        backend: TunnelBackend = TunnelBackend.CloudflareManagedTunnel,
        *,
        api_key: Optional[str] = None,
        backend_url: Optional[str] = None,
        verify_dns: bool = True,
        progress: bool = False,
    ) -> "TunneledLocalAPI":
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

        from synth_ai.sdk.tunnels.ports import find_available_port, kill_port
        from synth_ai.sdk.task.server import run_server_background
        from synth_ai.core.integrations.cloudflare import wait_for_health_check

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
