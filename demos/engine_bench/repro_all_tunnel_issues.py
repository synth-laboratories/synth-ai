#!/usr/bin/env python3
"""
Minimal repro scripts for all tunnel issues described in tunnel_stability_coding_agent.txt

Issues:
1. CloudflareManagedLease: "LeaseError: lease not found"
2. CloudflareManagedTunnel: "TunnelAPIError: rotate failed"  
3. CloudflareQuickTunnel: "timed out waiting for quick tunnel URL"

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai
    .venv/bin/python demos/engine_bench/repro_all_tunnel_issues.py
"""

import asyncio
import os
import sys
import traceback
from dataclasses import dataclass


@dataclass
class ReproResult:
    backend: str
    success: bool
    url: str | None
    error: str | None
    error_type: str | None


async def repro_managed_lease(api_key: str, env_key: str | None, port: int = 9091) -> ReproResult:
    """Repro Issue #1: CloudflareManagedLease - 'LeaseError: lease not found'"""
    from synth_ai.core.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI

    try:
        print("\n" + "=" * 60)
        print("ISSUE #1: CloudflareManagedLease")
        print("Expected error: 'LeaseError: lease not found'")
        print("=" * 60)
        
        tunnel = await asyncio.wait_for(
            TunneledLocalAPI.create(
                local_port=port,
                backend=TunnelBackend.CloudflareManagedLease,
                api_key=api_key,
                env_api_key=env_key,
                progress=True,
            ),
            timeout=60.0,
        )
        
        print(f"SUCCESS: Tunnel created at {tunnel.url}")
        tunnel.close()
        return ReproResult("CloudflareManagedLease", True, tunnel.url, None, None)
        
    except asyncio.TimeoutError:
        print("REPRODUCED: asyncio.TimeoutError - tunnel creation timed out")
        return ReproResult("CloudflareManagedLease", False, None, "Timeout after 60s", "TimeoutError")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"REPRODUCED: {error_type}: {error_msg}")
        return ReproResult("CloudflareManagedLease", False, None, error_msg, error_type)


async def repro_managed_tunnel(api_key: str, env_key: str | None, port: int = 9092) -> ReproResult:
    """Repro Issue #2: CloudflareManagedTunnel - 'TunnelAPIError: rotate failed'"""
    from synth_ai.core.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI

    try:
        print("\n" + "=" * 60)
        print("ISSUE #2: CloudflareManagedTunnel")
        print("Expected error: 'TunnelAPIError: rotate failed'")
        print("=" * 60)
        
        tunnel = await asyncio.wait_for(
            TunneledLocalAPI.create(
                local_port=port,
                backend=TunnelBackend.CloudflareManagedTunnel,
                api_key=api_key,
                env_api_key=env_key,
                progress=True,
            ),
            timeout=60.0,
        )
        
        print(f"SUCCESS: Tunnel created at {tunnel.url}")
        tunnel.close()
        return ReproResult("CloudflareManagedTunnel", True, tunnel.url, None, None)
        
    except asyncio.TimeoutError:
        print("REPRODUCED: asyncio.TimeoutError - tunnel creation timed out")
        return ReproResult("CloudflareManagedTunnel", False, None, "Timeout after 60s", "TimeoutError")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"REPRODUCED: {error_type}: {error_msg}")
        return ReproResult("CloudflareManagedTunnel", False, None, error_msg, error_type)


async def repro_quick_tunnel(api_key: str | None, port: int = 9093) -> ReproResult:
    """Repro Issue #3: CloudflareQuickTunnel - 'timed out waiting for quick tunnel URL'"""
    from synth_ai.core.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI

    try:
        print("\n" + "=" * 60)
        print("ISSUE #3: CloudflareQuickTunnel")
        print("Expected error: 'timed out waiting for quick tunnel URL'")
        print("=" * 60)
        
        tunnel = await asyncio.wait_for(
            TunneledLocalAPI.create(
                local_port=port,
                backend=TunnelBackend.CloudflareQuickTunnel,
                api_key=api_key,  # Not required but may be used for DNS verification
                progress=True,
            ),
            timeout=30.0,  # Quick tunnel should be faster
        )
        
        print(f"SUCCESS: Tunnel created at {tunnel.url}")
        tunnel.close()
        return ReproResult("CloudflareQuickTunnel", True, tunnel.url, None, None)
        
    except asyncio.TimeoutError:
        print("REPRODUCED: asyncio.TimeoutError - tunnel creation timed out after 30s")
        return ReproResult("CloudflareQuickTunnel", False, None, "Timeout after 30s", "TimeoutError")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"REPRODUCED: {error_type}: {error_msg}")
        return ReproResult("CloudflareQuickTunnel", False, None, error_msg, error_type)


async def repro_quick_tunnel_via_rust_direct(port: int = 9094) -> ReproResult:
    """Repro Issue #3 (direct): Call synth_ai_py.open_quick_tunnel directly"""
    import synth_ai_py

    try:
        print("\n" + "=" * 60)
        print("ISSUE #3 (direct Rust call): open_quick_tunnel")
        print("Expected: timeout or failure parsing cloudflared output")
        print("=" * 60)
        
        # Call the Rust function directly with a short timeout
        url, proc = await asyncio.to_thread(
            synth_ai_py.open_quick_tunnel,
            port,
            15.0,  # wait_s
        )
        
        print(f"SUCCESS: Quick tunnel URL = {url}")
        synth_ai_py.stop_tunnel(proc)
        return ReproResult("open_quick_tunnel (Rust)", True, url, None, None)
        
    except asyncio.TimeoutError:
        print("REPRODUCED: asyncio.TimeoutError")
        return ReproResult("open_quick_tunnel (Rust)", False, None, "Timeout", "TimeoutError")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"REPRODUCED: {error_type}: {error_msg}")
        return ReproResult("open_quick_tunnel (Rust)", False, None, error_msg, error_type)


async def repro_tunnel_open_unified(backend_key: str, api_key: str | None, env_key: str | None, port: int) -> ReproResult:
    """Repro via tunnel_open (the unified Rust entry point)"""
    import synth_ai_py

    backend_names = {
        "cloudflare_managed_lease": "CloudflareManagedLease",
        "cloudflare_managed": "CloudflareManagedTunnel",
        "cloudflare_quick": "CloudflareQuickTunnel",
    }
    name = backend_names.get(backend_key, backend_key)

    try:
        print(f"\n--- tunnel_open({backend_key}) ---")
        
        handle = await asyncio.wait_for(
            asyncio.to_thread(
                synth_ai_py.tunnel_open,
                backend_key,    # backend
                port,           # local_port
                api_key,        # api_key
                None,           # backend_url
                env_key,        # env_api_key
                False,          # use_http2
                True,           # verify_dns
                True,           # progress
            ),
            timeout=45.0,
        )
        
        print(f"SUCCESS: {name} -> {handle.url}")
        handle.close()
        return ReproResult(f"tunnel_open({backend_key})", True, handle.url, None, None)
        
    except asyncio.TimeoutError:
        print(f"REPRODUCED: {name} timed out after 45s")
        return ReproResult(f"tunnel_open({backend_key})", False, None, "Timeout after 45s", "TimeoutError")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"REPRODUCED: {name} -> {error_type}: {error_msg}")
        return ReproResult(f"tunnel_open({backend_key})", False, None, error_msg, error_type)


async def main():
    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("Getting demo API key...")
        from synth_ai.core.env import mint_demo_api_key
        api_key = mint_demo_api_key()
    
    print(f"API Key: {api_key[:20]}...")

    # Get env key
    env_key = None
    try:
        from synth_ai.sdk.localapi.auth import ensure_localapi_auth
        env_key = ensure_localapi_auth(synth_api_key=api_key)
        print(f"Env Key: {env_key[:20]}...")
    except Exception as e:
        print(f"Warning: Could not get env key: {e}")

    results: list[ReproResult] = []

    # Test all three backends via the high-level API
    results.append(await repro_managed_lease(api_key, env_key))
    results.append(await repro_managed_tunnel(api_key, env_key))
    results.append(await repro_quick_tunnel(api_key))
    
    # Also test the direct Rust calls
    results.append(await repro_quick_tunnel_via_rust_direct())

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Tunnel Issue Reproduction Results")
    print("=" * 70)
    
    for r in results:
        status = "✅ SUCCESS" if r.success else "❌ REPRODUCED"
        print(f"\n{status}: {r.backend}")
        if r.success:
            print(f"   URL: {r.url}")
        else:
            print(f"   Error type: {r.error_type}")
            print(f"   Error: {r.error[:100]}..." if r.error and len(r.error) > 100 else f"   Error: {r.error}")

    failures = [r for r in results if not r.success]
    successes = [r for r in results if r.success]
    
    print("\n" + "-" * 70)
    print(f"Total: {len(successes)}/{len(results)} succeeded, {len(failures)}/{len(results)} failed (reproduced)")
    
    if failures:
        print("\nFailed backends (issues reproduced):")
        for r in failures:
            print(f"  - {r.backend}: {r.error_type}")
    
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
