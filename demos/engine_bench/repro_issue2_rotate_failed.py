#!/usr/bin/env python3
"""
Repro Issue #2: CloudflareManagedTunnel - "TunnelAPIError: rotate failed"

This is the LEGACY managed tunnel backend.

Expected output:
    REPRODUCED: TunnelAPIError: rotate failed
    OR
    REPRODUCED: RuntimeError: ... rotate ...

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai
    .venv/bin/python demos/engine_bench/repro_issue2_rotate_failed.py
"""

import asyncio
import os
import sys


async def main():
    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("No SYNTH_API_KEY set, minting demo key...")
        from synth_ai.core.env import mint_demo_api_key
        api_key = mint_demo_api_key()
    
    print(f"API Key: {api_key[:20]}...")

    # Get env key
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth
    try:
        env_key = ensure_localapi_auth(synth_api_key=api_key)
        print(f"Env Key: {env_key[:20]}...")
    except Exception as e:
        print(f"Warning: Could not get env key: {e}")
        env_key = None

    print("\n" + "=" * 60)
    print("Testing CloudflareManagedTunnel backend (legacy)")
    print("Expected: 'TunnelAPIError: rotate failed'")
    print("=" * 60 + "\n")

    from synth_ai.core.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI

    try:
        tunnel = await asyncio.wait_for(
            TunneledLocalAPI.create(
                local_port=9092,
                backend=TunnelBackend.CloudflareManagedTunnel,
                api_key=api_key,
                env_api_key=env_key,
                progress=True,
            ),
            timeout=60.0,
        )
        
        print(f"\n✅ SUCCESS: Tunnel created at {tunnel.url}")
        print("Issue NOT reproduced - tunnel works!")
        tunnel.close()
        return 0
        
    except asyncio.TimeoutError:
        print(f"\n❌ REPRODUCED: TimeoutError - creation timed out after 60s")
        return 1
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Check for the expected error
        if "rotate" in error_msg.lower() or "TunnelAPIError" in error_type:
            print(f"\n❌ REPRODUCED (expected error): {error_type}: {error_msg}")
        else:
            print(f"\n❌ REPRODUCED (unexpected error): {error_type}: {error_msg}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
