"""Tunnel helpers for exposing local APIs to Synth's training infrastructure.

This module provides high-level and low-level tunnel management for
exposing local task apps to the internet. Two tunnel backends are available:

- **SynthTunnel** (default, recommended): Relay-based HTTPS tunnel. Traffic
  flows through Synth's relay servers via WebSocket. No ``cloudflared``
  binary required. Supports up to 128 concurrent in-flight requests (with
  dynamic memory-based budgeting for large payloads). Uses ``worker_token``
  for authentication.

- **Cloudflare**: Tunnels via Cloudflare's network. Requires the
  ``cloudflared`` binary. Available in managed (stable subdomain) and
  quick (random subdomain, no API key) variants. Uses ``task_app_api_key``
  for authentication.

**Recommended:** Use ``TunneledLocalAPI`` for a clean, one-liner experience:

    from synth_ai.core.tunnels import TunneledLocalAPI, TunnelBackend

    # SynthTunnel (relay-based, default — recommended)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        api_key="sk_live_...",
    )
    print(tunnel.url)           # https://dev.st.usesynth.ai/s/rt_...
    print(tunnel.worker_token)  # pass to job config

    # Cloudflare quick tunnel (no API key, random subdomain)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.CloudflareQuickTunnel,
    )

    # Localhost passthrough (no tunnel, local dev only)
    tunnel = await TunneledLocalAPI.create(
        local_port=8001,
        backend=TunnelBackend.Localhost,
    )

    tunnel.close()

**SynthTunnel vs Cloudflare — when to use which:**

+---------------------+----------------------------+----------------------------+
| Concern             | SynthTunnel                | Cloudflare                 |
+---------------------+----------------------------+----------------------------+
| Setup               | Zero config, no binary     | Requires ``cloudflared``   |
| Concurrency         | 128 in-flight (dynamic)    | Cloudflare limits apply    |
| Auth                | ``worker_token``           | ``task_app_api_key``       |
| URL stability       | Per-lease (session-lived)  | Managed = stable subdomain |
| Best for            | SDK jobs, GEPA, MiPRO      | Long-lived production apps |
+---------------------+----------------------------+----------------------------+

**Using tunnels with optimization jobs:**

    # SynthTunnel — pass worker_token
    job = PromptLearningJob.from_dict(
        config,
        task_app_url=tunnel.url,
        task_app_worker_token=tunnel.worker_token,
    )

    # Cloudflare — pass task_app_api_key instead
    job = PromptLearningJob.from_dict(
        config,
        task_app_url=tunnel.url,
        task_app_api_key=env_api_key,
    )

**Low-level:** For more control, use the individual functions:

    from synth_ai.core.tunnels import (
        rotate_tunnel,
        open_managed_tunnel,
        track_process,
        verify_tunnel_dns_resolution,
    )

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
