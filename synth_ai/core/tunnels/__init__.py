"""Cloudflare tunnel helpers for exposing local APIs.

This module provides high-level and low-level tunnel management for
exposing local task apps to the internet via Cloudflare tunnels.

**Recommended:** Use `TunneledLocalAPI` for a clean, one-liner experience:

    from synth_ai.core.tunnels import TunneledLocalAPI, TunnelBackend

    # SynthTunnel (relay-based, default)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.SynthTunnel,
        api_key="sk_live_...",
    )

    # Managed tunnel (stable subdomain, requires API key)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.CloudflareManagedTunnel,
        api_key="sk_live_...",
        env_api_key="env_key_...",
        progress=True,  # Print status updates
    )

    # Quick tunnel (random subdomain, no API key needed)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.CloudflareQuickTunnel,
        progress=True,
    )

    # Localhost passthrough (no tunnel)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.Localhost,
    )

    print(f"Local API exposed at: {tunnel.url}")

    # Use the URL for remote jobs (SynthTunnel requires worker token)
    job = PromptLearningJob.from_dict(
        config,
        task_app_url=tunnel.url,
        task_app_worker_token=tunnel.worker_token,
    )

    # Clean up when done
    tunnel.close()

Note:
    For Cloudflare tunnels, omit task_app_worker_token and use task_app_api_key as usual.

**Low-level:** For more control, use the individual functions:

    from synth_ai.core.tunnels import (
        rotate_tunnel,
        open_managed_tunnel,
        track_process,
        verify_tunnel_dns_resolution,
    )

    # Get a managed tunnel from backend
    tunnel = await rotate_tunnel(API_KEY, port=8001)

    # Start cloudflared with the token
    proc = track_process(open_managed_tunnel(tunnel['tunnel_token']))

    # Verify the tunnel is ready
    await verify_tunnel_dns_resolution(f"https://{tunnel['hostname']}")

Note:
    Processes registered with track_process() are automatically cleaned up
    when Python exits (via atexit). You can also call cleanup_all() manually.
"""

from __future__ import annotations

from typing import Any

from .cleanup import cleanup_all, track_process, tracked_processes
from .errors import (
    ConnectorError,
    ConnectorNotInstalledError,
    GatewayError,
    LeaseError,
    LeaseExpiredError,
    LocalAppError,
    TunnelAPIError,
    TunnelConfigurationError,
    TunnelError,
)
from .ports import PortConflictBehavior, PortInUseError
from .rust import (
    acquire_port,
    create_tunnel,
    ensure_cloudflared_installed,
    find_available_port,
    get_cloudflared_path,
    is_port_available,
    kill_port,
    open_managed_tunnel,
    open_managed_tunnel_with_connection_wait,
    open_quick_tunnel,
    open_quick_tunnel_with_dns_verification,
    require_cloudflared,
    rotate_tunnel,
    stop_tunnel,
    verify_tunnel_dns_resolution,
    wait_for_health_check,
)

# New: high-level tunnel abstraction
from .tunneled_api import TunnelBackend, TunneledLocalAPI
from .types import (
    ConnectorState,
    Diagnostics,
    GatewayState,
    LeaseInfo,
    LeaseState,
    TunnelHandle,
)


# Convenience function for creating tunnels from apps
async def create_tunneled_api(
    app: Any,
    local_port: int | None = None,
    backend: TunnelBackend = TunnelBackend.SynthTunnel,
    *,
    api_key: str | None = None,
    backend_url: str | None = None,
    verify_dns: bool = True,
    progress: bool = False,
) -> TunneledLocalAPI:
    """Create a tunnel for a FastAPI/ASGI app, handling server startup automatically.

    This is a convenience function that handles the common pattern of:
    1. Finding an available port (or using the provided one)
    2. Starting the app server
    3. Waiting for health check
    4. Creating the tunnel

    Args:
        app: FastAPI or ASGI application to tunnel
        local_port: Port to use (defaults to auto-finding an available port from 8001)
        backend: Tunnel backend to use
        api_key: Synth API key (defaults to SYNTH_API_KEY env var)
        backend_url: Backend URL (defaults to production)
        verify_dns: Whether to verify DNS resolution
        progress: If True, print status updates

    Returns:
        TunneledLocalAPI instance

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> tunnel = await create_tunneled_api(app)
        >>> print(f"App exposed at: {tunnel.url}")
    """
    return await TunneledLocalAPI.create_for_app(
        app=app,
        local_port=local_port,
        backend=backend,
        api_key=api_key,
        backend_url=backend_url,
        verify_dns=verify_dns,
        progress=progress,
    )


__all__ = [
    # High-level (RECOMMENDED)
    "TunneledLocalAPI",
    "TunnelBackend",
    "create_tunneled_api",
    # Lease-based system types
    "TunnelHandle",
    "LeaseInfo",
    "LeaseState",
    "Diagnostics",
    "ConnectorState",
    "GatewayState",
    # Errors
    "TunnelError",
    "TunnelConfigurationError",
    "TunnelAPIError",
    "LeaseError",
    "LeaseExpiredError",
    "ConnectorError",
    "ConnectorNotInstalledError",
    "GatewayError",
    "LocalAppError",
    # Legacy: Tunnel lifecycle
    "rotate_tunnel",
    "create_tunnel",
    "open_managed_tunnel",
    "open_managed_tunnel_with_connection_wait",
    "open_quick_tunnel",
    "open_quick_tunnel_with_dns_verification",
    "stop_tunnel",
    # Verification
    "verify_tunnel_dns_resolution",
    "wait_for_health_check",
    # Installation
    "require_cloudflared",
    "ensure_cloudflared_installed",
    "get_cloudflared_path",
    # Process tracking
    "track_process",
    "cleanup_all",
    "tracked_processes",
    # Port management
    "kill_port",
    "is_port_available",
    "find_available_port",
    "acquire_port",
    "PortConflictBehavior",
    "PortInUseError",
]
