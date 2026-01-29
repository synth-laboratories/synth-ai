#!/usr/bin/env python3
"""
Repro Issue #3: CloudflareQuickTunnel - "timed out waiting for quick tunnel URL"

This backend does NOT require an API key but still fails.
The cloudflared CLI itself works fine when run manually.

Expected output:
    REPRODUCED: ... timed out waiting for quick tunnel URL
    OR
    REPRODUCED: TimeoutError

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai
    .venv/bin/python demos/engine_bench/repro_issue3_quick_timeout.py
"""

import asyncio
import os
import sys


async def test_via_high_level_api():
    """Test via TunneledLocalAPI.create()"""
    print("\n--- Test 1: TunneledLocalAPI.create() ---")
    
    from synth_ai.core.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI

    try:
        tunnel = await asyncio.wait_for(
            TunneledLocalAPI.create(
                local_port=9093,
                backend=TunnelBackend.CloudflareQuickTunnel,
                api_key=None,  # Not required for quick tunnel
                progress=True,
            ),
            timeout=30.0,
        )
        
        print(f"✅ SUCCESS: {tunnel.url}")
        tunnel.close()
        return True
        
    except asyncio.TimeoutError:
        print("❌ REPRODUCED: TimeoutError after 30s")
        return False
    except Exception as e:
        print(f"❌ REPRODUCED: {type(e).__name__}: {e}")
        return False


async def test_via_rust_direct():
    """Test via synth_ai_py.open_quick_tunnel directly"""
    print("\n--- Test 2: synth_ai_py.open_quick_tunnel() ---")
    
    import synth_ai_py

    try:
        url, proc = await asyncio.wait_for(
            asyncio.to_thread(
                synth_ai_py.open_quick_tunnel,
                9094,   # port
                15.0,   # wait_s (default timeout in Rust)
            ),
            timeout=30.0,
        )
        
        print(f"✅ SUCCESS: {url}")
        synth_ai_py.stop_tunnel(proc)
        return True
        
    except asyncio.TimeoutError:
        print("❌ REPRODUCED: TimeoutError after 30s")
        return False
    except Exception as e:
        print(f"❌ REPRODUCED: {type(e).__name__}: {e}")
        return False


async def test_via_tunnel_open():
    """Test via synth_ai_py.tunnel_open (unified entry point)"""
    print("\n--- Test 3: synth_ai_py.tunnel_open('cloudflare_quick', ...) ---")
    
    import synth_ai_py

    try:
        handle = await asyncio.wait_for(
            asyncio.to_thread(
                synth_ai_py.tunnel_open,
                "cloudflare_quick",  # backend
                9095,                # local_port
                None,                # api_key (not needed)
                None,                # backend_url
                None,                # env_api_key
                False,               # use_http2
                False,               # verify_dns (skip to isolate issue)
                True,                # progress
            ),
            timeout=30.0,
        )
        
        print(f"✅ SUCCESS: {handle.url}")
        handle.close()
        return True
        
    except asyncio.TimeoutError:
        print("❌ REPRODUCED: TimeoutError after 30s")
        return False
    except Exception as e:
        print(f"❌ REPRODUCED: {type(e).__name__}: {e}")
        return False


async def main():
    print("=" * 60)
    print("Testing CloudflareQuickTunnel")
    print("Expected: 'timed out waiting for quick tunnel URL'")
    print("=" * 60)

    results = []
    results.append(("TunneledLocalAPI.create()", await test_via_high_level_api()))
    results.append(("synth_ai_py.open_quick_tunnel()", await test_via_rust_direct()))
    results.append(("synth_ai_py.tunnel_open()", await test_via_tunnel_open()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed - issue NOT reproduced")
        return 0
    else:
        print("\nAt least one test failed - issue REPRODUCED")
        print("\nNote: The cloudflared CLI works fine when run directly:")
        print("  cloudflared tunnel --url http://localhost:8080")
        print("This suggests the issue is in synth_ai_py Rust wrapper.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
