"""Cloudflare tunnel helpers for exposing local APIs.

This module provides high-level and low-level tunnel management for
exposing local task apps to the internet via Cloudflare tunnels.

**Recommended:** Use `TunneledLocalAPI` for a clean, one-liner experience:

    from synth_ai.sdk.tunnels import TunneledLocalAPI, TunnelBackend

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

    # Use the URL for remote jobs
    job = PromptLearningJob.from_dict(config, task_app_url=tunnel.url)

    # Clean up when done
    tunnel.close()

**Low-level:** For more control, use the individual functions:

    from synth_ai.sdk.tunnels import (
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

# Re-export from cloudflare.py (no wrappers - these are the actual functions)
from synth_ai.core.integrations.cloudflare import (
    # Tunnel lifecycle
    create_tunnel,
    open_managed_tunnel,
    open_quick_tunnel,
    open_quick_tunnel_with_dns_verification,
    rotate_tunnel,
    stop_tunnel,
    # Verification
    verify_tunnel_dns_resolution,
    wait_for_health_check,
    # Installation
    ensure_cloudflared_installed,
    get_cloudflared_path,
    require_cloudflared,
    # Discovery
    fetch_managed_tunnels,
    ManagedTunnelRecord,
    # Health checks (NEW)
    HealthResult,
    check_tunnel_health,
    check_tunnel_health_sync,
    check_all_tunnels_health,
    check_all_tunnels_health_sync,
)

# New: process tracking with atexit cleanup
from synth_ai.sdk.tunnels.cleanup import cleanup_all, track_process, tracked_processes

# New: port management utilities (was private in in_process.py)
from synth_ai.sdk.tunnels.ports import (
    find_available_port,
    is_port_available,
    kill_port,
    acquire_port,
    PortConflictBehavior,
    PortInUseError,
)

# New: high-level tunnel abstraction
from synth_ai.sdk.tunnels.tunneled_api import TunneledLocalAPI, TunnelBackend

# Convenience function for creating tunnels from apps
async def create_tunneled_api(
    app: Any,
    local_port: int | None = None,
    backend: TunnelBackend = TunnelBackend.CloudflareManagedTunnel,
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
    # Tunnel lifecycle
    "rotate_tunnel",
    "create_tunnel",
    "open_managed_tunnel",
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
    # Discovery
    "fetch_managed_tunnels",
    "ManagedTunnelRecord",
    # Health checks (NEW)
    "HealthResult",
    "check_tunnel_health",
    "check_tunnel_health_sync",
    "check_all_tunnels_health",
    "check_all_tunnels_health_sync",
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
