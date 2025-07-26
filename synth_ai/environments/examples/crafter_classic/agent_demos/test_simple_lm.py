#!/usr/bin/env python3
"""
Simple test of LM tracing with a basic Crafter episode.
"""

import asyncio
import os
from httpx import AsyncClient

# Set up environment
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["SYNTH_LOGGING"] = "false"
os.environ["SYNTH_TRACING_MODE"] = "v2"

from synth_ai.lm.core.main_v2 import LM
from synth_ai.tracing_v2.session_tracer import SessionTracer


async def test_simple():
    """Test basic LM functionality with Crafter."""
    print("Testing LM with Crafter...")
    
    # Check if service is running
    async with AsyncClient(base_url="http://localhost:8901") as client:
        health = await client.get("/health")
        print(f"Service health: {health.json()}")
        
        # Create instance
        create_resp = await client.post("/create_instance", json={
            "env_name": "CrafterClassic",
            "config": {"difficulty": "easy"}
        })
        print(f"Create response: {create_resp.status_code}")
        print(f"Create data: {create_resp.json()}")
        
        if create_resp.status_code == 200:
            instance_id = create_resp.json()["instance_id"]
            print(f"Instance ID: {instance_id}")
            
            # Reset
            reset_resp = await client.post(f"/reset/{instance_id}", json={"seed": 42})
            print(f"Reset response: {reset_resp.status_code}")
            
            # Clean up
            destroy_resp = await client.delete(f"/destroy/{instance_id}")
            print(f"Destroy response: {destroy_resp.status_code}")
            
    # Test LM
    tracer = SessionTracer()
    lm = LM(
        model_name="gpt-3.5-turbo",
        formatting_model_name="gpt-3.5-turbo",
        temperature=0,
        synth_logging=False,
        session_tracer=tracer,
        system_id="test_simple",
        enable_v2_tracing=True
    )
    
    tracer.start_session("test_session")
    
    response = await lm.respond_async(
        system_message="You are a game assistant.",
        user_message="I'm in a game world. What should I do first?",
        turn_number=0
    )
    
    print(f"\nLM Response: {response.raw_response}")
    
    trace_path = tracer.end_session()
    print(f"Trace saved to: {trace_path}")


if __name__ == "__main__":
    asyncio.run(test_simple())