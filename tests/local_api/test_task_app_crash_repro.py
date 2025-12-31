#!/usr/bin/env python3
"""Reproduce task app crashes by analyzing the aiohttp client behavior under load.

The task app crash manifests as "Server disconnected without sending a response"
which typically happens when:
1. Connection pool exhaustion
2. Memory pressure
3. Unhandled exceptions killing request handlers

This script tests the task app's HTTP client resilience by:
1. Hitting the health endpoint rapidly (baseline)
2. Simulating concurrent rollout requests that fail fast (no LLM call)
3. Monitoring for connection issues
"""

import asyncio
import time
import httpx
import sys
from concurrent.futures import ThreadPoolExecutor

TASK_APP_URL = "http://localhost:8001"
API_KEY = "sk_env_30c78a787bac223c716918181209f263"


async def stress_health(num_requests: int, concurrent: int) -> dict:
    """Rapid-fire health checks to stress the server."""
    print(f"\n=== HEALTH ENDPOINT STRESS: {num_requests} requests, {concurrent} concurrent ===")

    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "X-API-Key": API_KEY,
        }

        successes = 0
        failures = 0
        start = time.time()

        semaphore = asyncio.Semaphore(concurrent)

        async def make_request(i: int):
            nonlocal successes, failures
            async with semaphore:
                try:
                    resp = await client.get(
                        f"{TASK_APP_URL}/health",
                        headers=headers,
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        successes += 1
                    else:
                        failures += 1
                except Exception as e:
                    failures += 1
                    print(f"  [ERROR] Request {i}: {type(e).__name__}: {e}")

        await asyncio.gather(*[make_request(i) for i in range(num_requests)])

        elapsed = time.time() - start
        rps = num_requests / elapsed

        print(f"  Completed: {successes}/{num_requests} in {elapsed:.2f}s ({rps:.1f} req/s)")
        print(f"  Failures: {failures}")

        return {"successes": successes, "failures": failures, "rps": rps}


async def stress_rollout_fast_fail(num_requests: int, concurrent: int) -> dict:
    """Rollout requests that fail fast (invalid payload or missing endpoint).

    This stresses the HTTP path without making LLM calls.
    """
    print(f"\n=== ROLLOUT FAST-FAIL STRESS: {num_requests} requests, {concurrent} concurrent ===")

    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
        }

        successes_4xx = 0  # Expected failures (validation errors, 404s)
        successes_200 = 0
        connection_errors = 0
        other_errors = 0
        start = time.time()

        semaphore = asyncio.Semaphore(concurrent)

        async def make_request(i: int):
            nonlocal successes_4xx, successes_200, connection_errors, other_errors
            async with semaphore:
                # Use a minimal payload that will validate but fail at LLM call
                # (since interceptor isn't set up)
                payload = {
                    "run_id": f"stress-{i}",
                    "mode": "rl",
                    "env": {
                        "env_name": "banking77",
                        "seed": i % 100,
                        "config": {"split": "train"},
                    },
                    "policy": {
                        "config": {
                            "model": "gpt-4.1-nano",
                            "provider": "openai",
                            "inference_url": "http://localhost:8000/api/interceptor/v1",  # Will 404
                        }
                    },
                }

                try:
                    resp = await client.post(
                        f"{TASK_APP_URL}/rollout",
                        json=payload,
                        headers=headers,
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        successes_200 += 1
                    elif 400 <= resp.status_code < 500:
                        successes_4xx += 1  # Expected - means server is handling requests
                    else:
                        other_errors += 1
                        print(f"  [UNEXPECTED] Request {i}: status={resp.status_code}")
                except httpx.ConnectError as e:
                    connection_errors += 1
                    print(f"  [CONNECT_ERROR] Request {i}: {e}")
                except httpx.RemoteProtocolError as e:
                    connection_errors += 1
                    print(f"  [PROTOCOL_ERROR] Request {i}: {e}")
                except Exception as e:
                    other_errors += 1
                    if i % 100 == 0:  # Sample
                        print(f"  [ERROR] Request {i}: {type(e).__name__}: {e}")

        await asyncio.gather(*[make_request(i) for i in range(num_requests)])

        elapsed = time.time() - start
        rps = num_requests / elapsed

        print(f"  Completed in {elapsed:.2f}s ({rps:.1f} req/s)")
        print(f"  200s: {successes_200}, 4xx: {successes_4xx}")
        print(f"  Connection errors: {connection_errors}")
        print(f"  Other errors: {other_errors}")

        return {
            "successes_200": successes_200,
            "successes_4xx": successes_4xx,
            "connection_errors": connection_errors,
            "other_errors": other_errors,
            "rps": rps,
        }


async def monitor_health_during_load(duration_s: float = 10.0, interval_s: float = 1.0):
    """Monitor health endpoint while load test runs."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        start = time.time()
        checks = 0
        failures = 0

        while time.time() - start < duration_s:
            try:
                resp = await client.get(
                    f"{TASK_APP_URL}/health",
                    headers=headers,
                    timeout=2.0,
                )
                checks += 1
                if resp.status_code != 200:
                    failures += 1
                    print(f"  [HEALTH] Degraded: status={resp.status_code}")
            except Exception as e:
                checks += 1
                failures += 1
                print(f"  [HEALTH] FAILED: {type(e).__name__}: {e}")

            await asyncio.sleep(interval_s)

        return {"checks": checks, "failures": failures}


async def main():
    print("=" * 70)
    print("TASK APP CRASH REPRODUCTION TEST")
    print("=" * 70)

    # Check initial health
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{TASK_APP_URL}/health",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5.0,
            )
            print(f"\n[INITIAL HEALTH] {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"\n[INITIAL HEALTH] FAILED: {e}")
            print("Task app not running. Start it with:")
            print("  python banking77_local_api.py --port 8001")
            sys.exit(1)

    # Test 1: Baseline health stress
    await stress_health(num_requests=100, concurrent=20)

    # Test 2: Low concurrency rollouts
    await stress_rollout_fast_fail(num_requests=50, concurrent=5)

    # Test 3: Medium concurrency rollouts
    await stress_rollout_fast_fail(num_requests=100, concurrent=10)

    # Test 4: High concurrency rollouts (likely to cause issues)
    await stress_rollout_fast_fail(num_requests=200, concurrent=20)

    # Test 5: Very high concurrency (likely to cause issues)
    await stress_rollout_fast_fail(num_requests=500, concurrent=50)

    # Final health check
    await asyncio.sleep(1)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{TASK_APP_URL}/health",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5.0,
            )
            print(f"\n[FINAL HEALTH] {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"\n[FINAL HEALTH] FAILED - TASK APP MAY HAVE CRASHED: {e}")


if __name__ == "__main__":
    asyncio.run(main())
