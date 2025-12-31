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
    tunnel = await rotate_tunnel(API_KEY, port=8001, reason="demo")

    # Start cloudflared with the token
    proc = track_process(open_managed_tunnel(tunnel['tunnel_token']))

    # Verify the tunnel is ready
    await verify_tunnel_dns_resolution(f"https://{tunnel['hostname']}")

Note:
    Processes registered with track_process() are automatically cleaned up
    when Python exits (via atexit). You can also call cleanup_all() manually.
"""

from __future__ import annotations

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
)

# New: process tracking with atexit cleanup
from synth_ai.sdk.tunnels.cleanup import cleanup_all, track_process, tracked_processes

# New: port management utilities (was private in in_process.py)
from synth_ai.sdk.tunnels.ports import find_available_port, is_port_available, kill_port

# New: high-level tunnel abstraction
from synth_ai.sdk.tunnels.tunneled_api import TunneledLocalAPI, TunnelBackend

__all__ = [
    # High-level (RECOMMENDED)
    "TunneledLocalAPI",
    "TunnelBackend",
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
    # Process tracking (NEW)
    "track_process",
    "cleanup_all",
    "tracked_processes",
    # Port management (NEW - was private)
    "kill_port",
    "is_port_available",
    "find_available_port",
]
