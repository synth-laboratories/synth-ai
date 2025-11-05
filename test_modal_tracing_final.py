#!/usr/bin/env python3
"""
Final verification test for Modal tracing.
This will make a rollout request and capture the exact Modal container logs.
"""
import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv("/Users/joshpurtell/Documents/GitHub/synth-ai/.env")

MODAL_URL = "https://synth-laboratories--crafter-blogpost-fastapi-app-dev.modal.run"
API_KEY = os.getenv("ENVIRONMENT_API_KEY")

async def test_modal_rollout():
    print("=" * 80)
    print("MODAL TRACING VERIFICATION TEST")
    print("=" * 80)
    print(f"\nModal URL: {MODAL_URL}")
    print(f"API Key: {API_KEY[:15]}..." if API_KEY else "API Key: MISSING!")
    
    # Minimal rollout request (will fail but that's ok - we just want to see logs)
    payload = {
        "run_id": "test-modal-verification",
        "env": {"env_name": "crafter", "config": {}, "seed": 42},
        "policy": {
            "policy_name": "crafter-react",
            "config": {
                "inference_url": "http://fake-url",
                "model": "test-model",
                "trace_correlation_id": "test-trace-id"
            }
        },
        "ops": ["agent", "env"],  # Just 1 step
        "record": {"trajectories": True, "logprobs": False, "value": False},
        "on_done": "reset",
        "mode": "rl"
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print("\n" + "=" * 80)
    print("SENDING TEST ROLLOUT REQUEST")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{MODAL_URL}/rollout",
                json=payload,
                headers=headers
            )
            print(f"\nResponse status: {resp.status_code}")
            print(f"Response body preview: {str(resp.text)[:500]}")
        except Exception as e:
            print(f"\nRequest error (expected): {e}")
    
    print("\n" + "=" * 80)
    print("CHECK MODAL LOGS FOR:")
    print("=" * 80)
    print("✅ [TRACING_V3_CONFIG_LOADED] Python=3.11 MODAL_IS_REMOTE=1")
    print("✅ [TRACE_CONFIG] Modal detection: True")
    print("✅ [TRACE_CONFIG] Using Modal SQLite: file:/tmp/synth_traces.db")
    print("✅ [task:tracing] enabled (db=file:/tmp/synth_traces.db)")
    print("\n❌ Should NOT see: RuntimeError: Tracing backend not reachable")
    print("❌ Should NOT see: [task:tracing] enabled (db=libsql://127.0.0.1:8080)")

if __name__ == "__main__":
    asyncio.run(test_modal_rollout())



