#!/usr/bin/env python3
"""
Reproduce managed tunnel idle timeout issue.

This script:
1. Starts a minimal HTTP server on localhost
2. Creates a managed tunnel to expose it
3. Sends periodic requests through the tunnel (simulating rollouts)
4. Pauses for 30-90s (simulating GEPA inter-candidate computation)
5. Tries to reach the tunnel again — this is where it fails

Usage:
    python repro_tunnel_drop.py [--idle-seconds 60]
"""

import argparse
import asyncio
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from synth_ai.core.env import mint_demo_api_key
from synth_ai.core.tunnels.tunneled_api import TunneledLocalAPI, TunnelBackend
from synth_ai.core.urls import BACKEND_URL_BASE
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port


# --- Minimal test server ---

async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})

async def echo(request: Request) -> JSONResponse:
    body = await request.json()
    # Simulate work (like an OpenCode agent running cargo test)
    await asyncio.sleep(2)
    return JSONResponse({"echo": body, "ts": time.time()})

app = Starlette(routes=[
    Route("/health", health, methods=["GET"]),
    Route("/echo", echo, methods=["POST"]),
])


# --- Reproduction logic ---

async def send_request(client: httpx.AsyncClient, url: str, label: str) -> bool:
    """Send a test request through the tunnel. Returns True if successful."""
    try:
        r = await client.post(
            f"{url}/echo",
            json={"label": label, "ts": time.time()},
            timeout=30.0,
        )
        print(f"  [{label}] status={r.status_code} latency={r.elapsed.total_seconds():.1f}s")
        return r.status_code == 200
    except Exception as e:
        print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
        return False


async def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce managed tunnel idle timeout")
    parser.add_argument("--idle-seconds", type=int, default=60,
                        help="How long to idle between phases (default: 60)")
    parser.add_argument("--phase1-requests", type=int, default=6,
                        help="Number of requests in phase 1 (default: 6)")
    parser.add_argument("--phase2-requests", type=int, default=4,
                        help="Number of requests in phase 2 (default: 4)")
    parser.add_argument("--concurrent", type=int, default=2,
                        help="Concurrent requests per batch (default: 2)")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    backend_url = BACKEND_URL_BASE
    print(f"Backend: {backend_url}")

    # Mint demo key
    api_key = mint_demo_api_key(backend_url=backend_url)
    env_key = ensure_localapi_auth(backend_base=backend_url, synth_api_key=api_key)
    print(f"API Key: {api_key[:20]}...")

    # Start local server
    port = acquire_port(args.port, on_conflict=PortConflictBehavior.FIND_NEW)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)
    print(f"Local server: http://127.0.0.1:{port}")

    # Create managed tunnel
    print("Creating managed tunnel...")
    tunnel = await TunneledLocalAPI.create(
        local_port=port,
        backend=TunnelBackend.CloudflareManagedLease,
        api_key=api_key,
        env_api_key=env_key,
        progress=True,
    )
    tunnel_url = tunnel.url
    print(f"Tunnel URL: {tunnel_url}")

    # Verify tunnel works
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{tunnel_url}/health", timeout=10.0)
        print(f"Health check: {r.status_code}")
        if r.status_code != 200:
            print("FAIL: Tunnel health check failed immediately")
            tunnel.close()
            return 1

    # --- Phase 1: Active traffic (simulates candidate 1 rollouts) ---
    print(f"\n{'='*60}")
    print(f"PHASE 1: Sending {args.phase1_requests} requests (concurrent={args.concurrent})")
    print(f"{'='*60}")
    phase1_ok = 0
    async with httpx.AsyncClient() as client:
        for batch_start in range(0, args.phase1_requests, args.concurrent):
            batch_end = min(batch_start + args.concurrent, args.phase1_requests)
            tasks = [
                send_request(client, tunnel_url, f"phase1-{i}")
                for i in range(batch_start, batch_end)
            ]
            results = await asyncio.gather(*tasks)
            phase1_ok += sum(results)

    print(f"\nPhase 1 results: {phase1_ok}/{args.phase1_requests} succeeded")

    # --- Idle period (simulates GEPA mutation/crossover computation) ---
    print(f"\n{'='*60}")
    print(f"IDLE: Waiting {args.idle_seconds}s (simulating inter-candidate computation)")
    print(f"{'='*60}")
    for elapsed in range(0, args.idle_seconds, 10):
        remaining = args.idle_seconds - elapsed
        print(f"  ... {remaining}s remaining")
        await asyncio.sleep(min(10, remaining))
    print("  Idle period complete.")

    # --- Phase 2: Resume traffic (simulates candidate 2 rollouts) ---
    print(f"\n{'='*60}")
    print(f"PHASE 2: Sending {args.phase2_requests} requests (concurrent={args.concurrent})")
    print(f"  If the tunnel dropped, these will fail/timeout.")
    print(f"{'='*60}")
    phase2_ok = 0
    async with httpx.AsyncClient() as client:
        for batch_start in range(0, args.phase2_requests, args.concurrent):
            batch_end = min(batch_start + args.concurrent, args.phase2_requests)
            tasks = [
                send_request(client, tunnel_url, f"phase2-{i}")
                for i in range(batch_start, batch_end)
            ]
            results = await asyncio.gather(*tasks)
            phase2_ok += sum(results)

    print(f"\nPhase 2 results: {phase2_ok}/{args.phase2_requests} succeeded")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Phase 1 (before idle): {phase1_ok}/{args.phase1_requests}")
    print(f"  Idle duration: {args.idle_seconds}s")
    print(f"  Phase 2 (after idle):  {phase2_ok}/{args.phase2_requests}")
    if phase2_ok == 0 and phase1_ok > 0:
        print("\n  ❌ REPRODUCED: Tunnel dropped during idle period.")
        print("     Phase 1 worked, Phase 2 failed after idle gap.")
    elif phase2_ok < args.phase2_requests:
        print(f"\n  ⚠️  PARTIAL: {args.phase2_requests - phase2_ok} requests failed after idle.")
    else:
        print(f"\n  ✅ NOT REPRODUCED: Tunnel survived {args.idle_seconds}s idle.")
        print("     Try increasing --idle-seconds.")

    tunnel.close()
    server.should_exit = True
    await server_task
    return 0 if phase2_ok == args.phase2_requests else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
