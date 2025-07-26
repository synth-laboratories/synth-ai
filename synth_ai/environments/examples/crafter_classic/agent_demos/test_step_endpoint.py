#!/usr/bin/env python3
"""Test the step endpoint directly."""

import asyncio
import httpx
import json

async def test_step():
    async with httpx.AsyncClient(base_url="http://localhost:8901") as client:
        # Initialize
        init_resp = await client.post("/env/CrafterClassic/initialize", json={
            "config": {"difficulty": "easy", "seed": 42}
        })
        print(f"Init status: {init_resp.status_code}")
        init_data = init_resp.json()
        print(f"Init keys: {list(init_data.keys())}")
        
        env_id = init_data["env_id"]
        print(f"Env ID: {env_id}")
        
        # Try step
        step_resp = await client.post(f"/step/{env_id}", json={
            "env_id": env_id,
            "action": {"tool_calls": [{"tool": "move_up", "args": {}}]}
        })
        print(f"\nStep status: {step_resp.status_code}")
        
        if step_resp.status_code == 200:
            step_data = step_resp.json()
            print(f"Step keys: {list(step_data.keys())}")
            print(f"Step data (first 200 chars): {str(step_data)[:200]}...")
        else:
            print(f"Step error: {step_resp.text}")
        
        # Cleanup
        term_resp = await client.post("/env/CrafterClassic/terminate", json={
            "env_id": env_id
        })
        print(f"\nTerminate status: {term_resp.status_code}")

if __name__ == "__main__":
    asyncio.run(test_step())