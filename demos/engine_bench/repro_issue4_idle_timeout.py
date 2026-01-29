#!/usr/bin/env python3
"""
Repro Issue #4: Managed Tunnel Idle Timeout

PREREQUISITE: One of the tunnel backends must work first!
              Currently all backends fail (issues #1-3).

This script tests whether a working tunnel survives an idle period.
The symptom in real GEPA runs:
- Candidate 1 rollouts complete successfully (~13 requests over 10 min)
- 30-60s idle gap while algorithm does mutation/crossover
- Candidate 2 rollouts fail: "operation timed out"

The hypothesis is that Cloudflare reclaims the tunnel session during idle.

Expected (if tunnel creation worked):
    Phase 1: 6/6 succeeded
    [60s idle]
    Phase 2: 0/4 succeeded -> REPRODUCED

Currently expected:
    ERROR: Tunnel creation failed (see issues #1-3)

Usage:
    cd /Users/joshpurtell/Documents/GitHub/synth-ai
    .venv/bin/python demos/engine_bench/repro_issue4_idle_timeout.py --idle-seconds 60

This script reuses the existing repro_tunnel_drop.py but adds more diagnostics.
"""

import asyncio
import os
import sys
import time


async def create_tunnel(backend_name: str, api_key: str, env_key: str | None, port: int):
    """Try to create a tunnel with the specified backend."""
    from synth_ai.core.tunnels.tunneled_api import TunnelBackend, TunneledLocalAPI
    
    backend_map = {
        "lease": TunnelBackend.CloudflareManagedLease,
        "managed": TunnelBackend.CloudflareManagedTunnel,
        "quick": TunnelBackend.CloudflareQuickTunnel,
    }
    
    backend = backend_map.get(backend_name)
    if not backend:
        raise ValueError(f"Unknown backend: {backend_name}")
    
    return await TunneledLocalAPI.create(
        local_port=port,
        backend=backend,
        api_key=api_key,
        env_api_key=env_key,
        progress=True,
    )


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Repro tunnel idle timeout issue")
    parser.add_argument("--idle-seconds", type=int, default=60,
                        help="Idle duration between phases (default: 60)")
    parser.add_argument("--backend", choices=["lease", "managed", "quick"], default="lease",
                        help="Tunnel backend to test (default: lease)")
    parser.add_argument("--phase1-requests", type=int, default=6)
    parser.add_argument("--phase2-requests", type=int, default=4)
    parser.add_argument("--port", type=int, default=9096)
    args = parser.parse_args()

    print("=" * 70)
    print("ISSUE #4: Managed Tunnel Idle Timeout")
    print("=" * 70)
    print(f"\nBackend: {args.backend}")
    print(f"Idle duration: {args.idle_seconds}s")
    print(f"Phase 1 requests: {args.phase1_requests}")
    print(f"Phase 2 requests: {args.phase2_requests}")
    
    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("\nMinting demo API key...")
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

    # Start a minimal HTTP server
    print("\nStarting minimal HTTP server...")
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    
    async def health(request):
        return JSONResponse({"status": "ok"})
    
    async def echo(request):
        body = await request.json()
        await asyncio.sleep(0.5)  # Small delay
        return JSONResponse({"echo": body, "ts": time.time()})
    
    app = Starlette(routes=[
        Route("/health", health, methods=["GET"]),
        Route("/echo", echo, methods=["POST"]),
    ])
    
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)
    print(f"Server running on http://127.0.0.1:{args.port}")

    # Try to create tunnel
    print(f"\nCreating {args.backend} tunnel...")
    try:
        tunnel = await asyncio.wait_for(
            create_tunnel(args.backend, api_key, env_key, args.port),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        print(f"\n❌ BLOCKED: Tunnel creation timed out after 60s")
        print("Cannot test idle timeout - tunnel creation itself fails (issues #1-3)")
        server.should_exit = True
        return 1
    except Exception as e:
        print(f"\n❌ BLOCKED: Tunnel creation failed: {type(e).__name__}: {e}")
        print("Cannot test idle timeout - tunnel creation itself fails (issues #1-3)")
        server.should_exit = True
        return 1

    tunnel_url = tunnel.url
    print(f"Tunnel URL: {tunnel_url}")

    # Test function
    import httpx
    
    async def send_request(client: httpx.AsyncClient, label: str) -> bool:
        try:
            r = await client.post(
                f"{tunnel_url}/echo",
                json={"label": label, "ts": time.time()},
                timeout=30.0,
            )
            print(f"  [{label}] status={r.status_code}")
            return r.status_code == 200
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            return False

    # Phase 1: Active traffic
    print(f"\n{'='*50}")
    print(f"PHASE 1: Sending {args.phase1_requests} requests")
    print(f"{'='*50}")
    
    phase1_ok = 0
    async with httpx.AsyncClient() as client:
        for i in range(args.phase1_requests):
            if await send_request(client, f"phase1-{i}"):
                phase1_ok += 1
    
    print(f"\nPhase 1 result: {phase1_ok}/{args.phase1_requests} succeeded")

    # Idle period
    print(f"\n{'='*50}")
    print(f"IDLE: Waiting {args.idle_seconds}s (simulating GEPA inter-candidate gap)")
    print(f"{'='*50}")
    
    for elapsed in range(0, args.idle_seconds, 10):
        remaining = args.idle_seconds - elapsed
        print(f"  ... {remaining}s remaining")
        await asyncio.sleep(min(10, remaining))
    print("  Idle complete.")

    # Phase 2: Resume traffic
    print(f"\n{'='*50}")
    print(f"PHASE 2: Sending {args.phase2_requests} requests (testing if tunnel survived)")
    print(f"{'='*50}")
    
    phase2_ok = 0
    async with httpx.AsyncClient() as client:
        for i in range(args.phase2_requests):
            if await send_request(client, f"phase2-{i}"):
                phase2_ok += 1
    
    print(f"\nPhase 2 result: {phase2_ok}/{args.phase2_requests} succeeded")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Phase 1 (before idle): {phase1_ok}/{args.phase1_requests}")
    print(f"  Idle duration: {args.idle_seconds}s")
    print(f"  Phase 2 (after idle):  {phase2_ok}/{args.phase2_requests}")
    
    if phase2_ok == 0 and phase1_ok > 0:
        print(f"\n  ❌ REPRODUCED: Tunnel dropped during {args.idle_seconds}s idle period")
        print("     Phase 1 worked, Phase 2 failed = idle timeout issue")
        exit_code = 1
    elif phase2_ok < args.phase2_requests:
        print(f"\n  ⚠️  PARTIAL: {args.phase2_requests - phase2_ok} requests failed after idle")
        exit_code = 1
    else:
        print(f"\n  ✅ NOT REPRODUCED: Tunnel survived {args.idle_seconds}s idle")
        print("     Try increasing --idle-seconds to stress-test further")
        exit_code = 0

    # Cleanup
    tunnel.close()
    server.should_exit = True
    await server_task
    
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
