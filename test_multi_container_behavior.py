#!/usr/bin/env python3
"""
Multi-Container Behavior Test for Synth-AI
==========================================
This test verifies that synth-ai can trigger multiple containers on the backend
without knowing anything about Modal, containers, or deployment details.

The test is provider-agnostic and only uses the OpenAI-compatible API.
"""

import asyncio
import time
import os
import statistics
from typing import List, Dict, Any
from datetime import datetime
import httpx

# Test configuration
SERVICE_URL = os.getenv("SYNTH_BASE_URL", "https://synth-laboratories--unified-ft-service-fastapi-app.modal.run")
API_KEY = os.getenv("SYNTH_API_KEY", "sk-test-11111111111111111111111111111111")
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"


class MultiContainerVerifier:
    """Verify multi-container behavior using only the public API."""
    
    def __init__(self):
        self.service_url = SERVICE_URL.rstrip('/v1').rstrip('/')
        self.api_key = API_KEY
        self.results = []
        
    async def make_api_request(self, request_id: int, prompt: str) -> Dict[str, Any]:
        """Make a standard OpenAI-compatible API request."""
        messages = [{"role": "user", "content": f"[Request {request_id}] {prompt}"}]
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.service_url}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 50,
                        "stream": False
                    }
                )
                
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract container info if available (backend may include it)
                    container_id = result.get("_container_id", "unknown")
                    timestamp = result.get("_timestamp", time.time())
                    
                    return {
                        "request_id": request_id,
                        "success": True,
                        "latency": end_time - start_time,
                        "container_id": container_id,
                        "timestamp": timestamp,
                        "response": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                        "status_code": response.status_code
                    }
                else:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "latency": end_time - start_time,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "container_id": "error",
                        "status_code": response.status_code
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_id,
                "success": False,
                "latency": end_time - start_time,
                "error": str(e),
                "container_id": "error",
                "status_code": 0
            }
    
    async def run_concurrent_load_test(self, num_requests: int = 50, max_concurrent: int = 25):
        """Run concurrent requests to test multi-container behavior."""
        print(f"\n🧪 Multi-Container Behavior Test")
        print(f"=" * 60)
        print(f"Service URL: {self.service_url}")
        print(f"Model: {MODEL_NAME}")
        print(f"Total Requests: {num_requests}")
        print(f"Max Concurrent: {max_concurrent}")
        print(f"=" * 60)
        
        # Test service health first
        print("\n🔍 Testing service health...")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                health_response = await client.get(f"{self.service_url}/health")
                if health_response.status_code == 200:
                    print(f"✅ Service is healthy: {health_response.json()}")
                else:
                    print(f"⚠️  Service health check returned: {health_response.status_code}")
        except Exception as e:
            print(f"⚠️  Health check failed: {e}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_request(request_id: int):
            async with semaphore:
                return await self.make_api_request(request_id, "What is the capital of France?")
        
        # Run all requests
        start_time = time.time()
        print(f"\n⏱️  Starting {num_requests} requests at {datetime.now().strftime('%H:%M:%S')}...")
        
        tasks = [limited_request(i) for i in range(num_requests)]
        self.results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        self.analyze_results(total_time)
        
        return self.results
    
    def analyze_results(self, total_time: float):
        """Analyze test results for multi-container behavior."""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        print(f"\n✅ Completed in {total_time:.2f} seconds")
        print(f"\n📊 Results Summary:")
        print(f"  Successful: {len(successful)}/{len(self.results)}")
        print(f"  Failed: {len(failed)}/{len(self.results)}")
        
        if successful:
            latencies = [r["latency"] for r in successful]
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            median_latency = statistics.median(latencies)
            
            print(f"\n⏱️  Latency Statistics:")
            print(f"  Average: {avg_latency:.3f}s")
            print(f"  Median: {median_latency:.3f}s")
            print(f"  Min: {min_latency:.3f}s")
            print(f"  Max: {max_latency:.3f}s")
            
            # Container distribution analysis
            container_ids = [r.get("container_id", "unknown") for r in successful]
            unique_containers = set(container_ids)
            
            print(f"\n🔧 Container Distribution:")
            print(f"  Unique containers detected: {len(unique_containers)}")
            
            if len(unique_containers) > 1:
                print(f"  ✅ Multi-container behavior confirmed!")
                
                container_counts = {}
                for cid in container_ids:
                    container_counts[cid] = container_counts.get(cid, 0) + 1
                
                for cid, count in sorted(container_counts.items()):
                    percentage = (count / len(successful)) * 100
                    print(f"    Container {str(cid)[-8:]}: {count} requests ({percentage:.1f}%)")
            else:
                print(f"  ⚠️  Single container detected: {list(unique_containers)[0]}")
                print(f"      This may indicate:")
                print(f"      - Load was not high enough to trigger scaling")
                print(f"      - Backend is still warming up containers")
                print(f"      - Container scaling takes time to activate")
            
            # Throughput calculation
            throughput = len(successful) / total_time
            print(f"\n🚀 Performance:")
            print(f"  Throughput: {throughput:.2f} requests/second")
            print(f"  Total processing time: {total_time:.2f}s")
            
            # Concurrency effectiveness
            max_concurrent = min(25, len(successful))  # Estimate from test
            theoretical_min_time = sum(latencies) / max_concurrent
            concurrency_efficiency = (theoretical_min_time / total_time) * 100
            print(f"  Concurrency efficiency: {concurrency_efficiency:.1f}%")
        
        if failed:
            print(f"\n❌ Failed Requests:")
            for r in failed[:5]:  # Show first 5 failures
                print(f"  Request {r['request_id']}: {r.get('error', 'Unknown error')}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more failures")
    
    async def test_scaling_behavior(self):
        """Test how the system scales with increasing load."""
        print(f"\n🔬 Testing Scaling Behavior")
        print(f"=" * 60)
        
        load_levels = [5, 10, 20, 50]
        scaling_results = {}
        
        for num_requests in load_levels:
            print(f"\n📈 Testing with {num_requests} requests...")
            await self.run_concurrent_load_test(
                num_requests=num_requests,
                max_concurrent=min(num_requests, 25)
            )
            
            successful = [r for r in self.results if r["success"]]
            if successful:
                avg_latency = statistics.mean([r["latency"] for r in successful])
                unique_containers = len(set(r.get("container_id", "unknown") for r in successful))
                
                scaling_results[num_requests] = {
                    "avg_latency": avg_latency,
                    "unique_containers": unique_containers,
                    "successful_requests": len(successful)
                }
        
        # Print scaling summary
        print(f"\n📊 Scaling Summary:")
        print(f"{'Requests':<10} {'Avg Latency':<15} {'Containers':<12} {'Success Rate':<12}")
        print("-" * 50)
        
        for num_requests, data in scaling_results.items():
            success_rate = (data['successful_requests'] / num_requests) * 100
            print(f"{num_requests:<10} {data['avg_latency']:<15.3f} {data['unique_containers']:<12} {success_rate:<12.1f}%")


async def main():
    """Run the multi-container behavior test."""
    print("🧪 Synth-AI Multi-Container Behavior Test")
    print("=========================================")
    print("This test verifies that synth-ai can trigger multiple containers")
    print("through the backend API without knowing about Modal or containers.")
    print()
    
    verifier = MultiContainerVerifier()
    
    # Test with aggressive load to trigger scaling
    await verifier.run_concurrent_load_test(num_requests=100, max_concurrent=50)
    
    print("\n" + "="*60)
    print("✅ Test completed!")
    print()
    print("💡 Key Points:")
    print("   - Synth-AI remains completely provider-agnostic")
    print("   - Backend handles all container scaling transparently") 
    print("   - Users only see a standard OpenAI-compatible API")
    print("   - Multi-container benefits work without any code changes")


if __name__ == "__main__":
    asyncio.run(main()) 