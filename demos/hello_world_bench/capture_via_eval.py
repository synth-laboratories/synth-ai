#!/usr/bin/env python3
"""Run an eval job and capture the trace."""

import asyncio
import os
import time
from pathlib import Path

import httpx
from localapi_hello_world_bench import app
from synth_ai.core.env import mint_demo_api_key
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.tunnels import PortConflictBehavior, acquire_port


async def wait_for_health(host: str, port: int, api_key: str, timeout: float = 30.0) -> None:
    url = f"http://{host}:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, headers=headers, timeout=5.0)
            if r.status_code in (200, 400):
                return
        except:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Health check failed: {url}")


async def main():
    backend = "http://localhost:8000"

    # Check backend
    r = httpx.get(f"{backend}/health", timeout=10)
    if r.status_code != 200:
        print(f"Backend not healthy: {r.status_code}")
        return

    # Get API key
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        api_key = mint_demo_api_key(backend_url=backend)
    print(f"API key: {api_key[:20]}...")

    # Get env key
    env_key = ensure_localapi_auth(backend_base=backend, synth_api_key=api_key)
    print(f"Env key: {env_key[:20]}...")

    # Start task app
    port = acquire_port(8030, on_conflict=PortConflictBehavior.FIND_NEW)
    run_server_background(app, port)
    await wait_for_health("localhost", port, env_key)
    task_url = f"http://localhost:{port}"
    print(f"Task app ready: {task_url}")

    # Submit eval job
    print("\nSubmitting eval job...")
    job_payload = {
        "task_app_url": task_url,
        "task_app_api_key": env_key,
        "app_id": "hello_world_bench",
        "env_name": "hello_world_bench",
        "seeds": [0],
        "policy": {
            "model": "gpt-5-nano",
            "provider": "openai",
        },
        "max_concurrent": 1,
        "timeout": 120.0,
    }

    headers = {"Authorization": f"Bearer {api_key}"}
    r = httpx.post(f"{backend}/api/eval/jobs", json=job_payload, headers=headers, timeout=30)
    if r.status_code >= 400:
        print(f"Failed to create job: {r.status_code}")
        print(f"Response: {r.text}")
        try:
            print(f"JSON: {r.json()}")
        except:
            pass
        return

    job_id = r.json()["job_id"]
    print(f"Job ID: {job_id}")

    # Poll until complete
    print("\nPolling job status...")
    for _ in range(60):
        r = httpx.get(f"{backend}/api/eval/jobs/{job_id}", headers=headers, timeout=10)
        if r.status_code != 200:
            print(f"Failed to get job status: {r.status_code}")
            return

        status = r.json()["status"]
        print(f"Status: {status}")

        if status == "completed":
            break
        if status == "failed":
            print(f"Job failed: {r.json().get('error')}")
            return

        await asyncio.sleep(2)
    else:
        print("Job timed out")
        return

    # Get results to find correlation_id
    print("\nFetching results...")
    r = httpx.get(f"{backend}/api/eval/jobs/{job_id}/results", headers=headers, timeout=10)
    if r.status_code != 200:
        print(f"Failed to get results: {r.status_code}")
        return

    results = r.json()
    rows = results.get("rows", [])
    if not rows:
        print("No results rows")
        return

    # Get correlation_id from first result
    first_row = rows[0]
    correlation_id = first_row.get("trace_correlation_id")
    if not correlation_id:
        print("No trace_correlation_id in results")
        print(f"Row keys: {list(first_row.keys())}")
        return

    print(f"\nCorrelation ID: {correlation_id}")

    # Fetch trace
    await asyncio.sleep(2)
    trace_url = f"{backend}/api/interceptor/v1/trace/by-correlation/{correlation_id}"
    print(f"\nFetching trace: {trace_url}")

    r = httpx.get(trace_url, headers=headers, timeout=10)
    if r.status_code != 200:
        print(f"Failed to fetch trace: {r.status_code} {r.text}")
        return

    data = r.json()
    matches = data.get("matches", [])
    print(f"Matches: {len(matches)}")

    if not matches:
        print("No trace matches!")
        return

    # Extract request_messages
    match = matches[0]
    metadata = match.get("metadata", {})
    conversation = metadata.get("conversation", {})
    request_messages = conversation.get("request_messages", [])

    print(f"\n{'=' * 60}")
    print("REQUEST MESSAGES")
    print(f"{'=' * 60}\n")

    for i, msg in enumerate(request_messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        print(f"[{i}] ROLE: {role}")
        print(f"    LENGTH: {len(content)} chars")
        print("    CONTENT:")
        print("-" * 40)
        print(content)
        print("-" * 40)
        print()

    # Save to file
    import json

    out_file = Path(__file__).parent / "captured_messages.json"
    with open(out_file, "w") as f:
        json.dump({"correlation_id": correlation_id, "messages": request_messages}, f, indent=2)
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
