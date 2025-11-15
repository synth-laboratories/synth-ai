#!/usr/bin/env python3
"""Test script to replicate the GEPA pattern validation issue locally."""

import asyncio
import sys
import os
from pathlib import Path

# Add monorepo backend to path
monorepo_backend = Path(__file__).parent.parent.parent.parent.parent / "monorepo" / "backend"
sys.path.insert(0, str(monorepo_backend))

from app.routes.prompt_learning.core.validation import fetch_baseline_messages
from synth_ai.tracing_v3.client import TraceClient


async def test_fetch_baseline_messages():
    """Test fetching baseline messages from Iris task app."""
    
    task_app_url = "http://127.0.0.1:8115"
    task_app_api_key = os.getenv("ENVIRONMENT_API_KEY", "sk_env_30c78a787bac223c716918181209f263")
    
    policy_config = {
        "model": "openai/gpt-oss-20b",
        "provider": "groq",
        "temperature": 0.0,
        "max_completion_tokens": 128,
        "inference_mode": "synth_hosted",
        "inference_url": "http://localhost:8000/api/inference/v1",  # Use local backend interceptor
    }
    
    # Create trace client
    trace_client = TraceClient(
        base_url="http://127.0.0.1:8081",  # Local sqld
    )
    
    print("=" * 80)
    print("Testing fetch_baseline_messages with Iris task app")
    print("=" * 80)
    print(f"Task app URL: {task_app_url}")
    print(f"Env name: iris")
    print(f"Seed: 0")
    print()
    
    try:
        messages, error = await fetch_baseline_messages(
            task_app_url=task_app_url,
            task_app_api_key=task_app_api_key,
            policy_config=policy_config,
            env_name="iris",
            env_config=None,
            seed=0,
            timeout=120.0,
            trace_client=trace_client,
            trace_lease_ttl=30.0,
        )
        
        if error:
            print(f"❌ Error: {error}")
            return False
        
        if not messages:
            print("❌ No messages returned")
            return False
        
        print(f"✅ Success! Got {len(messages)} messages:")
        for i, msg in enumerate(messages):
            print(f"  {i+1}. [{msg.get('role')}]: {msg.get('content', '')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_fetch_baseline_messages())
    sys.exit(0 if success else 1)

