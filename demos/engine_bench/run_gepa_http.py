#!/usr/bin/env python3
"""Run GEPA via HTTP API."""

import asyncio
import json
import os
from pathlib import Path

import httpx


async def run_gepa():
    config_path = Path(__file__).parent / "enginebench_gepa_quick.toml"

    # Get API key from environment
    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("⚠ Warning: SYNTH_API_KEY not set in environment")
        return None

    request = {
        "algorithm": "gepa",
        "config_path": str(config_path),
        "auto_start": True
    }

    print("=" * 80)
    print("Starting GEPA Unified Optimization via HTTP API")
    print("=" * 80)
    print(f"Config: {config_path}")
    print("Backend: http://localhost:8000")
    print(f"API Key: {api_key[:20]}...")
    print("=" * 80)

    async with httpx.AsyncClient(timeout=600.0) as client:
        print("\nSending request to /api/prompt-learning/online/jobs endpoint...")
        response = await client.post(
            "http://localhost:8000/api/prompt-learning/online/jobs",
            json=request,
            headers={"X-API-Key": api_key}
        )

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        if response.status_code == 200:
            print("\n✓ GEPA optimization started successfully!")
            print(f"Job ID: {result.get('job_id')}")
        else:
            print(f"\n✗ Failed: {result.get('error')}")

        return result

if __name__ == "__main__":
    asyncio.run(run_gepa())
