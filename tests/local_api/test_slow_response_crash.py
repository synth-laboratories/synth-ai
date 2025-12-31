#!/usr/bin/env python3
"""Test task app behavior with slow backend responses.

This simulates the real-world crash scenario:
1. Task app has limited connection pool (10 connections, 5 per host)
2. Backend interceptor responds slowly (30+ seconds per LLM call)
3. Multiple concurrent requests exhaust connection pool
4. New requests timeout waiting for connections
5. Eventually task app process crashes

We simulate this by running a slow mock server and pointing task app at it.
"""

import asyncio
import time
import signal
import sys
from multiprocessing import Process

import httpx
import uvicorn
from fastapi import FastAPI

MOCK_SERVER_PORT = 8002
TASK_APP_URL = "http://localhost:8001"
API_KEY = "sk_env_30c78a787bac223c716918181209f263"


def create_mock_slow_server(delay_seconds: float = 5.0):
    """Create a mock LLM server that responds slowly."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok", "delay": delay_seconds}

    @app.post("/v1/chat/completions")
    async def slow_completions():
        """Simulate slow LLM response."""
        await asyncio.sleep(delay_seconds)
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-slow-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock",
                                "type": "function",
                                "function": {
                                    "name": "banking77_classify",
                                    "arguments": '{"intent": "card_arrival"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }

    return app


def run_mock_server(delay: float, port: int):
    """Run mock server in a separate process."""
    app = create_mock_slow_server(delay)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


async def stress_with_slow_backend(
    num_requests: int,
    concurrent: int,
    delay_seconds: float,
) -> dict:
    """Stress test task app with slow mock backend.

    Args:
        num_requests: Total number of requests to send
        concurrent: Max concurrent requests
        delay_seconds: How long each mock LLM response takes

    Returns:
        Dict with test results
    """
    print(f"\n{'='*70}")
    print(f"SLOW BACKEND STRESS TEST")
    print(f"  Requests: {num_requests}, Concurrent: {concurrent}")
    print(f"  Mock LLM delay: {delay_seconds}s")
    print(f"{'='*70}")

    # Start mock slow server
    print("\n[1] Starting mock slow LLM server...")
    mock_process = Process(target=run_mock_server, args=(delay_seconds, MOCK_SERVER_PORT))
    mock_process.start()
    await asyncio.sleep(2)  # Wait for server to start

    # Verify mock server is running
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"http://localhost:{MOCK_SERVER_PORT}/health", timeout=2.0)
            print(f"  Mock server ready: {resp.json()}")
        except Exception as e:
            print(f"  Mock server failed to start: {e}")
            mock_process.terminate()
            return {"error": "Mock server failed to start"}

    # Verify task app is running
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{TASK_APP_URL}/health",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5.0,
            )
            print(f"  Task app ready: {resp.status_code}")
    except Exception as e:
        print(f"  Task app not ready: {e}")
        mock_process.terminate()
        return {"error": "Task app not running"}

    # Run stress test
    print(f"\n[2] Starting stress test...")
    results = {
        "successes": 0,
        "failures_4xx": 0,
        "failures_5xx": 0,
        "connection_errors": 0,
        "timeouts": 0,
        "other_errors": 0,
        "durations": [],
    }

    semaphore = asyncio.Semaphore(concurrent)

    async def make_request(i: int) -> None:
        async with semaphore:
            payload = {
                "run_id": f"slow-test-{i}",
                "mode": "rl",
                "env": {
                    "env_name": "banking77",
                    "seed": i % 100,
                    "config": {"split": "train"},
                },
                "policy": {
                    "config": {
                        "model": "mock-slow-model",
                        "provider": "openai",
                        "inference_url": f"http://localhost:{MOCK_SERVER_PORT}/v1",
                    }
                },
            }
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "X-API-Key": API_KEY,
                "Content-Type": "application/json",
            }

            start = time.time()
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{TASK_APP_URL}/rollout",
                        json=payload,
                        headers=headers,
                        timeout=60.0,  # Allow time for slow response
                    )
                    elapsed = time.time() - start
                    results["durations"].append(elapsed)

                    if resp.status_code == 200:
                        results["successes"] += 1
                        if i % 10 == 0:
                            print(f"  [OK] Request {i}: {elapsed:.1f}s")
                    elif 400 <= resp.status_code < 500:
                        results["failures_4xx"] += 1
                        print(f"  [4xx] Request {i}: {resp.status_code} in {elapsed:.1f}s")
                    else:
                        results["failures_5xx"] += 1
                        print(f"  [5xx] Request {i}: {resp.status_code} in {elapsed:.1f}s")
            except httpx.ConnectError as e:
                elapsed = time.time() - start
                results["connection_errors"] += 1
                print(f"  [CONNECT] Request {i}: {e} after {elapsed:.1f}s")
            except httpx.TimeoutException as e:
                elapsed = time.time() - start
                results["timeouts"] += 1
                print(f"  [TIMEOUT] Request {i}: {e} after {elapsed:.1f}s")
            except httpx.RemoteProtocolError as e:
                elapsed = time.time() - start
                results["connection_errors"] += 1
                print(f"  [PROTOCOL] Request {i}: {e} after {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.time() - start
                results["other_errors"] += 1
                print(f"  [ERROR] Request {i}: {type(e).__name__}: {e} after {elapsed:.1f}s")

    start_time = time.time()
    tasks = [make_request(i) for i in range(num_requests)]
    await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time

    # Check final health
    print(f"\n[3] Checking task app health...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{TASK_APP_URL}/health",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5.0,
            )
            print(f"  Final health: {resp.status_code}")
    except Exception as e:
        print(f"  TASK APP CRASHED: {e}")
        results["task_app_crashed"] = True

    # Stop mock server
    mock_process.terminate()
    mock_process.join(timeout=2)
    if mock_process.is_alive():
        mock_process.kill()

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successes: {results['successes']}")
    print(f"4xx failures: {results['failures_4xx']}")
    print(f"5xx failures: {results['failures_5xx']}")
    print(f"Connection errors: {results['connection_errors']}")
    print(f"Timeouts: {results['timeouts']}")
    print(f"Other errors: {results['other_errors']}")

    if results["durations"]:
        avg_duration = sum(results["durations"]) / len(results["durations"])
        max_duration = max(results["durations"])
        min_duration = min(results["durations"])
        print(f"Duration: avg={avg_duration:.1f}s, min={min_duration:.1f}s, max={max_duration:.1f}s")

    return results


async def main():
    # Test 1: Low concurrency with short delays (should work)
    results1 = await stress_with_slow_backend(
        num_requests=10,
        concurrent=2,
        delay_seconds=1.0,
    )

    # Test 2: Medium concurrency with medium delays (may show issues)
    results2 = await stress_with_slow_backend(
        num_requests=20,
        concurrent=5,
        delay_seconds=3.0,
    )

    # Test 3: High concurrency with longer delays (likely to cause issues)
    # This simulates GEPA with 50 candidates being evaluated
    results3 = await stress_with_slow_backend(
        num_requests=30,
        concurrent=10,
        delay_seconds=5.0,
    )

    # Test 4: Extreme - simulate production crash scenario
    # High concurrency + long delays
    results4 = await stress_with_slow_backend(
        num_requests=50,
        concurrent=20,
        delay_seconds=10.0,
    )

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Test 1 (low load): {results1.get('successes', 0)} successes, {results1.get('connection_errors', 0)} connection errors")
    print(f"Test 2 (medium load): {results2.get('successes', 0)} successes, {results2.get('connection_errors', 0)} connection errors")
    print(f"Test 3 (high load): {results3.get('successes', 0)} successes, {results3.get('connection_errors', 0)} connection errors")
    print(f"Test 4 (extreme): {results4.get('successes', 0)} successes, {results4.get('connection_errors', 0)} connection errors")


if __name__ == "__main__":
    asyncio.run(main())
