#!/usr/bin/env python3
"""
Create Harbor deployments for EngineBench evaluations.

This script creates deployments via the Harbor API:
1. Packages the Dockerfile and run_rollout.py script
2. Uploads to the Harbor API
3. Triggers a build
4. Waits for the deployment to be ready

Usage:
    export SYNTH_API_KEY=sk_live_...
    uv run python demos/harbor/create_deployment.py --name engine-bench-v1

    # Create multiple deployments (one per seed variant)
    uv run python demos/harbor/create_deployment.py --name engine-bench-seed --count 5
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Optional

import httpx


def get_api_client(base_url: str, api_key: str) -> httpx.AsyncClient:
    """Create an async HTTP client with auth."""
    return httpx.AsyncClient(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=300.0,
    )


def create_build_context() -> str:
    """Create a base64-encoded tar.gz of the build context.

    Includes:
    - Dockerfile
    - run_rollout.py
    """
    demo_dir = Path(__file__).parent

    # Create tar.gz in memory
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        # Add Dockerfile
        dockerfile_path = demo_dir / "Dockerfile"
        tar.add(dockerfile_path, arcname="Dockerfile")

        # Add run_rollout.py
        runner_path = demo_dir / "run_rollout.py"
        tar.add(runner_path, arcname="run_rollout.py")

    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


async def create_deployment(
    client: httpx.AsyncClient,
    name: str,
    description: Optional[str] = None,
    agent_type: str = "opencode",
) -> dict:
    """Create a new Harbor deployment.

    Args:
        client: HTTP client
        name: Deployment name (org-unique)
        description: Optional description
        agent_type: Agent type (opencode, codex)

    Returns:
        Deployment response
    """
    print(f"Creating deployment: {name}")

    # Create build context
    print("  Packaging build context...")
    context_tar = create_build_context()
    print(f"  Build context size: {len(context_tar)} bytes (base64)")

    # Create deployment request
    # NOTE: Don't set SYNTH_API_KEY here - the executor injects it at runtime
    # This ensures the API key matches the requesting org and rotates automatically
    request_body = {
        "name": name,
        "description": description or f"EngineBench deployment with {agent_type} agent",
        "context_tar_base64": context_tar,
        "entrypoint": "run_rollout --input /tmp/rollout.json --output /tmp/result.json",
        "entrypoint_mode": "file",
        "limits": {
            "timeout_s": 600,  # 10 minutes for complex card implementations
            "cpu_cores": 4,
            "memory_mb": 8192,
            "disk_mb": 20480,
        },
        "env_vars": {},  # No LLM keys - executor injects SYNTH_API_KEY at runtime
        "metadata": {
            "agent_type": agent_type,
            "benchmark": "engine-bench",
            "version": "1.0",
        },
    }

    print("  Sending create request...")
    response = await client.post("/api/harbor/deployments", json=request_body)

    if response.status_code == 409:
        print(f"  Deployment '{name}' already exists")
        # Get existing deployment
        list_response = await client.get(f"/api/harbor/deployments?name={name}")
        if list_response.status_code == 200:
            deployments = list_response.json().get("deployments", [])
            for d in deployments:
                if d["name"] == name:
                    return d
        raise RuntimeError(f"Deployment exists but couldn't retrieve it")

    if response.status_code != 200:
        raise RuntimeError(f"Failed to create deployment: {response.status_code} - {response.text}")

    deployment = response.json()
    print(f"  Created deployment: {deployment['id']}")
    return deployment


async def wait_for_deployment_ready(
    client: httpx.AsyncClient,
    deployment_name: str,
    timeout_s: int = 600,
    poll_interval_s: int = 10,
) -> dict:
    """Wait for a deployment to be ready.

    Args:
        client: HTTP client
        deployment_name: Deployment name (routes use name, not ID)
        timeout_s: Maximum wait time
        poll_interval_s: Polling interval

    Returns:
        Deployment status response
    """
    print(f"Waiting for deployment '{deployment_name}' to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout_s:
        response = await client.get(f"/api/harbor/deployments/{deployment_name}/status")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get status: {response.status_code} {response.text}")

        status = response.json()
        deployment_status = status["status"]

        print(f"  Status: {deployment_status}")

        if deployment_status == "ready":
            print(f"  Deployment ready! Snapshot: {status.get('snapshot_id')}")
            return status
        elif deployment_status == "failed":
            # Get build logs
            builds = status.get("builds", [])
            if builds:
                latest_build = builds[0]
                error = latest_build.get("error_message", "Unknown error")
                raise RuntimeError(f"Build failed: {error}")
            raise RuntimeError("Build failed with unknown error")
        elif deployment_status in ("pending", "building"):
            await asyncio.sleep(poll_interval_s)
        else:
            raise RuntimeError(f"Unexpected status: {deployment_status}")

    raise TimeoutError(f"Deployment not ready after {timeout_s}s")


async def main():
    parser = argparse.ArgumentParser(description="Create Harbor EngineBench deployments")
    parser.add_argument(
        "--name",
        type=str,
        default="engine-bench",
        help="Base deployment name (default: engine-bench)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of deployments to create (default: 1)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="opencode",
        choices=["opencode", "codex", "claude_code"],
        help="Agent type (default: opencode)",
    )
    parser.add_argument(
        "--backend-url",
        type=str,
        default="https://api-dev.usesynth.ai",
        help="Backend API URL",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for deployments to be ready",
    )
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY")
    if not api_key:
        print("ERROR: SYNTH_API_KEY not set")
        sys.exit(1)

    print("=" * 60)
    print("CREATING HARBOR DEPLOYMENTS FOR ENGINE-BENCH")
    print("=" * 60)
    print(f"Backend: {args.backend_url}")
    print(f"Count: {args.count}")
    print(f"Agent: {args.agent}")
    print()

    async with get_api_client(args.backend_url, api_key) as client:
        deployments = []

        for i in range(args.count):
            name = args.name if args.count == 1 else f"{args.name}-{i}"

            try:
                deployment = await create_deployment(
                    client=client,
                    name=name,
                    agent_type=args.agent,
                )
                deployments.append(deployment)

                if args.wait:
                    await wait_for_deployment_ready(client, deployment["name"])

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Created {len(deployments)} deployment(s):")
        for d in deployments:
            print(f"  - {d['name']}: {d['id']} (status: {d['status']})")

        # Output deployment IDs for scripting
        print()
        print("Deployment IDs (for run_harbor_eval.py):")
        print(",".join(d["id"] for d in deployments))


if __name__ == "__main__":
    asyncio.run(main())
