#!/usr/bin/env python3
"""
Simple test script for the GRPO Synth Envs Hosted Service.

Run this after starting the service with:
    python main.py
"""

import asyncio
import json

import httpx


async def test_service():
    """Test basic service functionality."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        # Test 1: Service info
        print("1. Testing /info endpoint...")
        response = await client.get(f"{base_url}/info")
        assert response.status_code == 200
        info = response.json()
        print(f"   Service info: {json.dumps(info, indent=2)}")

        # Test 2: Health check
        print("\n2. Testing /health endpoint...")
        response = await client.get(f"{base_url}/health")
        assert response.status_code == 200
        print(f"   Health: {response.json()}")

        # Test 3: Create environment
        print("\n3. Creating environment...")
        response = await client.post(
            f"{base_url}/env/create",
            json={
                "env_name": "crafter",
                "config": {},
                "seed": 42,
                "rl_run_id": "test-run-001",
            },
        )
        if response.status_code != 200:
            print(f"   Error: {response.status_code} - {response.text}")
            return
        env_data = response.json()
        env_id = env_data["env_id"]
        print(f"   Created env: {env_id}")
        print(f"   Initial observation keys: {list(env_data['observation'].keys())}")

        # Test 4: Create policy
        print("\n4. Creating policy...")
        response = await client.post(
            f"{base_url}/policy/create",
            json={
                "policy_name": "crafter-react",
                "config": {
                    "inference_url": "http://localhost:8001",
                    "model": "test-model",
                },
                "rl_run_id": "test-run-001",
                "bound_env_id": env_id,
            },
        )
        if response.status_code != 200:
            print(f"   Error: {response.status_code} - {response.text}")
            return
        policy_data = response.json()
        policy_id = policy_data["policy_id"]
        print(f"   Created policy: {policy_id}")

        # Test 5: Environment step with dummy tool calls
        print("\n5. Testing environment step...")
        response = await client.post(
            f"{base_url}/env/step",
            json={
                "env_id": env_id,
                "tool_calls": [{"tool": "interact", "args": {"action": "move_left"}}],
            },
        )
        if response.status_code != 200:
            print(f"   Error: {response.status_code} - {response.text}")
        else:
            step_data = response.json()
            print(f"   Step result - done: {step_data['done']}, reward: {step_data.get('reward')}")

        # Test 6: Environment snapshot
        print("\n6. Creating environment snapshot...")
        response = await client.post(f"{base_url}/env/snapshot", json={"env_id": env_id})
        if response.status_code != 200:
            print(f"   Error: {response.status_code} - {response.text}")
        else:
            snapshot_data = response.json()
            print(f"   Snapshot ID: {snapshot_data['snapshot_id']}")
            print(f"   Size: {snapshot_data['size']} bytes")

        # Test 7: Policy snapshot
        print("\n7. Creating policy snapshot...")
        response = await client.post(f"{base_url}/policy/snapshot", json={"policy_id": policy_id})
        if response.status_code != 200:
            print(f"   Error: {response.status_code} - {response.text}")
        else:
            snapshot_data = response.json()
            print(f"   Snapshot ID: {snapshot_data['snapshot_id']}")
            print(f"   Size: {snapshot_data['size']} bytes")

        # Test 8: Run status
        print("\n8. Testing run status...")
        response = await client.get(f"{base_url}/run/status/test-run-001")
        if response.status_code != 200:
            print(f"   Error: {response.status_code} - {response.text}")
        else:
            status_data = response.json()
            print(f"   Run status: {status_data['status']}")

        # Test 9: Terminate environment
        print("\n9. Terminating environment...")
        response = await client.post(f"{base_url}/env/terminate", json={"env_id": env_id})
        if response.status_code != 200:
            print(f"   Error: {response.status_code} - {response.text}")
        else:
            print(f"   Environment terminated: {response.json()['ok']}")

        # Test 10: Terminate policy
        print("\n10. Terminating policy...")
        response = await client.post(f"{base_url}/policy/terminate", json={"policy_id": policy_id})
        if response.status_code != 200:
            print(f"   Error: {response.status_code} - {response.text}")
        else:
            print(f"   Policy terminated: {response.json()['ok']}")

        print("\n✅ All basic tests completed!")


if __name__ == "__main__":
    asyncio.run(test_service())
