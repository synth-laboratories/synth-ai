#!/usr/bin/env python3
"""
Diagnostic script to test Crafter service performance directly.
This will help identify if the slowdown is in the service or the client.
"""

import asyncio
import time
import httpx
import uuid
from typing import List, Dict, Any

async def test_service_performance():
    """Test the Crafter service performance directly."""
    
    service_url = "http://localhost:8901"
    
    print("🔍 Crafter Service Performance Diagnostic")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        start_time = time.time()
        try:
            response = await client.get(f"{service_url}/health")
            end_time = time.time()
            print(f"   ✅ Health check: {response.status_code} in {(end_time - start_time)*1000:.1f}ms")
            print(f"   📄 Response size: {len(response.content)} bytes")
        except Exception as e:
            print(f"   ❌ Health check failed: {e}")
            return
    
    # Test 2: Environment creation
    print("\n2️⃣ Testing environment creation...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        start_time = time.time()
        try:
            # Create proper task instance format
            task_instance = {
                "id": str(uuid.uuid4()),
                "metadata": {
                    "difficulty": "easy",
                    "seed": 42,
                    "num_trees_radius": 5,
                    "num_cows_radius": 2,
                    "num_hostiles_radius": 0
                },
                "impetus": {
                    "instructions": "Survive and unlock achievements in easy environment."
                },
                "intent": {
                    "rubric": {},
                    "gold_trajectories": None,
                    "gold_state_diff": {}
                },
                "is_reproducible": True,
                "initial_engine_snapshot": None,
                "config": {"world_config": "easy"}
            }
            
            response = await client.post(
                f"{service_url}/env/CrafterClassic/initialize",
                json={"task_instance": task_instance}
            )
            end_time = time.time()
            print(f"   ✅ Environment creation: {response.status_code} in {(end_time - start_time)*1000:.1f}ms")
            
            if response.status_code != 200:
                print(f"   ❌ Error response: {response.text[:200]}")
            
            if response.status_code == 200:
                data = response.json()
                obs_size = len(str(data.get("observation", {})))
                print(f"   📄 Observation size: {obs_size} characters")
                print(f"   🆔 Environment ID: {data.get('env_id', 'N/A')}")
                
                env_id = data.get("env_id")
                
                # Test 3: Step execution
                if env_id:
                    print(f"\n3️⃣ Testing step execution (env_id: {env_id})...")
                    step_start = time.time()
                    step_response = await client.post(
                        f"{service_url}/env/CrafterClassic/step",
                        json={
                            "env_id": env_id,
                            "request_id": str(uuid.uuid4()),
                            "action": {
                                "tool_calls": [{"tool": "interact", "args": {"action": 5}}]
                            }
                        }
                    )
                    
                    if step_response.status_code != 200:
                        print(f"   ❌ Step error: {step_response.text[:200]}")
                    step_end = time.time()
                    print(f"   ✅ Step execution: {step_response.status_code} in {(step_end - step_start)*1000:.1f}ms")
                    
                    if step_response.status_code == 200:
                        step_data = step_response.json()
                        step_obs_size = len(str(step_data.get("observation", {})))
                        print(f"   📄 Step observation size: {step_obs_size} characters")
                        
                        # Test 4: Multiple rapid steps
                        print(f"\n4️⃣ Testing multiple rapid steps...")
                        step_times = []
                        for i in range(5):
                            step_start = time.time()
                            rapid_response = await client.post(
                                f"{service_url}/env/CrafterClassic/step",
                                json={
                                    "env_id": env_id,
                                    "request_id": str(uuid.uuid4()),
                                    "action": {
                                        "tool_calls": [{"tool": "interact", "args": {"action": 5}}]
                                    }
                                }
                            )
                            step_end = time.time()
                            step_time = (step_end - step_start) * 1000
                            step_times.append(step_time)
                            print(f"   Step {i+1}: {rapid_response.status_code} in {step_time:.1f}ms")
                        
                        avg_step_time = sum(step_times) / len(step_times)
                        print(f"   📊 Average step time: {avg_step_time:.1f}ms")
                        print(f"   📊 Min step time: {min(step_times):.1f}ms")
                        print(f"   📊 Max step time: {max(step_times):.1f}ms")
                        
                        # Test 5: Close environment
                        print(f"\n5️⃣ Testing environment cleanup...")
                        cleanup_start = time.time()
                        cleanup_response = await client.post(f"{service_url}/env/{env_id}/close")
                        cleanup_end = time.time()
                        print(f"   ✅ Cleanup: {cleanup_response.status_code} in {(cleanup_end - cleanup_start)*1000:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ Environment test failed: {e}")
    
    # Test 6: Concurrent requests
    print(f"\n6️⃣ Testing concurrent requests...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        async def create_and_step():
            try:
                # Create proper task instance format
                task_instance = {
                    "id": str(uuid.uuid4()),
                    "metadata": {
                        "difficulty": "easy",
                        "seed": 42,
                        "num_trees_radius": 5,
                        "num_cows_radius": 2,
                        "num_hostiles_radius": 0
                    },
                    "impetus": {
                        "instructions": "Survive and unlock achievements in easy environment."
                    },
                    "intent": {
                        "rubric": {},
                        "gold_trajectories": None,
                        "gold_state_diff": {}
                    },
                    "is_reproducible": True,
                    "initial_engine_snapshot": None,
                    "config": {"world_config": "easy"}
                }
                
                # Create environment
                create_response = await client.post(
                    f"{service_url}/env/CrafterClassic/initialize",
                    json={"task_instance": task_instance}
                )
                if create_response.status_code == 200:
                    env_id = create_response.json()["env_id"]
                    # Take one step
                    step_response = await client.post(
                        f"{service_url}/env/CrafterClassic/step",
                        json={
                            "env_id": env_id,
                            "request_id": str(uuid.uuid4()),
                            "action": {
                                "tool_calls": [{"tool": "interact", "args": {"action": 5}}]
                            }
                        }
                    )
                    # Cleanup
                    await client.post(f"{service_url}/env/{env_id}/close")
                    return step_response.status_code == 200
                return False
            except:
                return False
        
        # Run 3 concurrent requests
        start_time = time.time()
        results = await asyncio.gather(*[create_and_step() for _ in range(3)])
        end_time = time.time()
        
        success_count = sum(results)
        print(f"   ✅ Concurrent test: {success_count}/3 successful in {(end_time - start_time)*1000:.1f}ms")
        print(f"   📊 Average per request: {(end_time - start_time)*1000/3:.1f}ms")

if __name__ == "__main__":
    asyncio.run(test_service_performance()) 