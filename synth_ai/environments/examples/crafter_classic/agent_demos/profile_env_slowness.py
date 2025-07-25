#!/usr/bin/env python3
"""Deep profiling of environment service slowness."""

import asyncio
import time
import httpx
import json
import cProfile
import pstats
import io
from typing import Dict, List, Any, Tuple
import statistics
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnvironmentProfiler:
    """Profile different layers of the environment service stack."""
    
    def __init__(self, service_url: str = "http://localhost:8901"):
        self.service_url = service_url
        self.timings: Dict[str, List[float]] = {
            # Network layer
            "http_request_total": [],
            "http_request_only": [],
            "json_serialization": [],
            "json_deserialization": [],
            
            # Service layer
            "service_processing": [],
            "env_initialization": [],
            "env_step": [],
            
            # Data transfer
            "request_size_bytes": [],
            "response_size_bytes": [],
            
            # Connection
            "connection_setup": [],
            "dns_lookup": [],
        }
        
    async def profile_single_step(self, client: httpx.AsyncClient, env_id: str) -> Dict[str, Any]:
        """Profile a single environment step with detailed timing."""
        
        # Prepare request payload
        payload = {
            "env_id": env_id,
            "action": {
                "tool_calls": [{
                    "tool": "interact",
                    "args": {"action": 0}
                }]
            }
        }
        
        # Time JSON serialization
        json_start = time.time()
        json_data = json.dumps(payload)
        json_time = time.time() - json_start
        self.timings["json_serialization"].append(json_time)
        self.timings["request_size_bytes"].append(len(json_data))
        
        # Time the full HTTP request
        total_start = time.time()
        
        # Make request with detailed timing
        response = await client.post(
            f"{self.service_url}/env/CrafterClassic/step",
            content=json_data,
            headers={"Content-Type": "application/json"},
            timeout=30.0
        )
        
        total_time = time.time() - total_start
        self.timings["http_request_total"].append(total_time)
        
        # Time JSON deserialization
        response_text = response.text
        self.timings["response_size_bytes"].append(len(response_text))
        
        deser_start = time.time()
        response_data = json.loads(response_text)
        deser_time = time.time() - deser_start
        self.timings["json_deserialization"].append(deser_time)
        
        # Calculate network-only time (excluding serialization)
        network_only = total_time - json_time - deser_time
        self.timings["http_request_only"].append(network_only)
        
        return {
            "total_time": total_time,
            "network_time": network_only,
            "json_serialize": json_time,
            "json_deserialize": deser_time,
            "request_size": len(json_data),
            "response_size": len(response_text),
            "response_data": response_data
        }
        
    async def profile_with_connection_reuse(self):
        """Test performance with connection reuse vs new connections."""
        logger.info("\n" + "="*60)
        logger.info("TESTING CONNECTION REUSE IMPACT")
        logger.info("="*60)
        
        # Test 1: Reusing connection
        logger.info("\n1. With connection reuse:")
        async with httpx.AsyncClient() as client:
            # Initialize environment
            init_response = await client.post(
                f"{self.service_url}/env/CrafterClassic/initialize",
                json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}}
            )
            env_id = init_response.json()["env_id"]
            
            # Run 10 steps with same connection
            reuse_times = []
            for i in range(10):
                result = await self.profile_single_step(client, env_id)
                reuse_times.append(result["total_time"])
                logger.debug(f"  Step {i+1}: {result['total_time']:.3f}s")
                
            # Cleanup
            await client.post(f"{self.service_url}/env/CrafterClassic/terminate", json={"env_id": env_id})
            
        logger.info(f"  Mean time with reuse: {statistics.mean(reuse_times):.3f}s")
        
        # Test 2: New connection each time
        logger.info("\n2. With new connection each time:")
        
        # Initialize environment first
        async with httpx.AsyncClient() as client:
            init_response = await client.post(
                f"{self.service_url}/env/CrafterClassic/initialize",
                json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}}
            )
            env_id = init_response.json()["env_id"]
        
        new_conn_times = []
        for i in range(10):
            # New client each time
            async with httpx.AsyncClient() as client:
                result = await self.profile_single_step(client, env_id)
                new_conn_times.append(result["total_time"])
                logger.debug(f"  Step {i+1}: {result['total_time']:.3f}s")
                
        # Cleanup
        async with httpx.AsyncClient() as client:
            await client.post(f"{self.service_url}/env/CrafterClassic/terminate", json={"env_id": env_id})
            
        logger.info(f"  Mean time with new connections: {statistics.mean(new_conn_times):.3f}s")
        logger.info(f"  Overhead from new connections: {statistics.mean(new_conn_times) - statistics.mean(reuse_times):.3f}s")
        
    async def profile_payload_size_impact(self):
        """Test if large payloads are causing slowness."""
        logger.info("\n" + "="*60)
        logger.info("TESTING PAYLOAD SIZE IMPACT")
        logger.info("="*60)
        
        async with httpx.AsyncClient() as client:
            # Initialize environment
            init_response = await client.post(
                f"{self.service_url}/env/CrafterClassic/initialize",
                json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}}
            )
            env_data = init_response.json()
            env_id = env_data["env_id"]
            
            # Check observation size
            obs_json = json.dumps(env_data["observation"])
            logger.info(f"\nObservation size: {len(obs_json)} bytes ({len(obs_json)/1024:.1f} KB)")
            
            # Analyze observation structure
            obs = env_data["observation"]
            field_sizes = {}
            for key, value in obs.items():
                if isinstance(value, (list, dict)):
                    field_sizes[key] = len(json.dumps(value))
                else:
                    field_sizes[key] = len(str(value))
                    
            # Sort by size
            sorted_fields = sorted(field_sizes.items(), key=lambda x: x[1], reverse=True)
            logger.info("\nLargest observation fields:")
            for field, size in sorted_fields[:5]:
                logger.info(f"  {field}: {size} bytes ({size/1024:.1f} KB)")
                
            # Test step timing
            step_times = []
            for i in range(5):
                result = await self.profile_single_step(client, env_id)
                step_times.append(result)
                
            # Analyze
            logger.info("\nTiming breakdown (average of 5 steps):")
            logger.info(f"  Total time: {statistics.mean([r['total_time'] for r in step_times]):.3f}s")
            logger.info(f"  Network only: {statistics.mean([r['network_time'] for r in step_times]):.3f}s")
            logger.info(f"  JSON serialize: {statistics.mean([r['json_serialize'] for r in step_times]):.6f}s")
            logger.info(f"  JSON deserialize: {statistics.mean([r['json_deserialize'] for r in step_times]):.6f}s")
            logger.info(f"  Response size: {statistics.mean([r['response_size'] for r in step_times]):.0f} bytes")
            
            # Cleanup
            await client.post(f"{self.service_url}/env/CrafterClassic/terminate", json={"env_id": env_id})
            
    async def test_concurrent_environments(self):
        """Test if multiple environments interfere with each other."""
        logger.info("\n" + "="*60)
        logger.info("TESTING CONCURRENT ENVIRONMENT INTERFERENCE")
        logger.info("="*60)
        
        async with httpx.AsyncClient() as client:
            # Create 5 environments
            env_ids = []
            for i in range(5):
                init_response = await client.post(
                    f"{self.service_url}/env/CrafterClassic/initialize",
                    json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}}
                )
                env_ids.append(init_response.json()["env_id"])
                
            logger.info(f"Created {len(env_ids)} environments")
            
            # Test 1: Sequential steps across different environments
            logger.info("\n1. Sequential steps across environments:")
            seq_times = []
            for i in range(10):
                env_id = env_ids[i % len(env_ids)]
                result = await self.profile_single_step(client, env_id)
                seq_times.append(result["total_time"])
                
            logger.info(f"  Mean time: {statistics.mean(seq_times):.3f}s")
            
            # Test 2: Concurrent steps
            logger.info("\n2. Concurrent steps:")
            
            async def concurrent_step(env_id: str) -> float:
                result = await self.profile_single_step(client, env_id)
                return result["total_time"]
                
            # Run 5 concurrent steps
            start = time.time()
            concurrent_results = await asyncio.gather(*[
                concurrent_step(env_id) for env_id in env_ids
            ])
            concurrent_time = time.time() - start
            
            logger.info(f"  Total time for 5 concurrent steps: {concurrent_time:.3f}s")
            logger.info(f"  Mean individual step time: {statistics.mean(concurrent_results):.3f}s")
            
            # Cleanup
            for env_id in env_ids:
                await client.post(f"{self.service_url}/env/CrafterClassic/terminate", json={"env_id": env_id})
                
    async def profile_service_internals(self):
        """Try to understand what the service is doing internally."""
        logger.info("\n" + "="*60)
        logger.info("ANALYZING SERVICE BEHAVIOR")
        logger.info("="*60)
        
        async with httpx.AsyncClient() as client:
            # Test with minimal config
            logger.info("\n1. Testing with minimal world size:")
            
            # Small world
            small_response = await client.post(
                f"{self.service_url}/env/CrafterClassic/initialize",
                json={"initial_state": {}, "config": {"area": [32, 32], "length": 10}}
            )
            small_env_id = small_response.json()["env_id"]
            
            # Time steps
            small_times = []
            for i in range(5):
                result = await self.profile_single_step(client, small_env_id)
                small_times.append(result["total_time"])
                
            logger.info(f"  32x32 world mean step time: {statistics.mean(small_times):.3f}s")
            
            # Normal world
            logger.info("\n2. Testing with normal world size:")
            normal_response = await client.post(
                f"{self.service_url}/env/CrafterClassic/initialize",
                json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}}
            )
            normal_env_id = normal_response.json()["env_id"]
            
            normal_times = []
            for i in range(5):
                result = await self.profile_single_step(client, normal_env_id)
                normal_times.append(result["total_time"])
                
            logger.info(f"  64x64 world mean step time: {statistics.mean(normal_times):.3f}s")
            
            # Cleanup
            await client.post(f"{self.service_url}/env/CrafterClassic/terminate", json={"env_id": small_env_id})
            await client.post(f"{self.service_url}/env/CrafterClassic/terminate", json={"env_id": normal_env_id})
            
    def print_summary(self):
        """Print timing summary."""
        logger.info("\n" + "="*60)
        logger.info("TIMING SUMMARY")
        logger.info("="*60)
        
        for category, times in self.timings.items():
            if times and category not in ["request_size_bytes", "response_size_bytes"]:
                logger.info(f"\n{category}:")
                logger.info(f"  Samples: {len(times)}")
                logger.info(f"  Mean: {statistics.mean(times):.3f}s")
                logger.info(f"  Median: {statistics.median(times):.3f}s")
                logger.info(f"  Min: {min(times):.3f}s")
                logger.info(f"  Max: {max(times):.3f}s")
                
        # Print size statistics
        if self.timings["response_size_bytes"]:
            logger.info(f"\nResponse sizes:")
            logger.info(f"  Mean: {statistics.mean(self.timings['response_size_bytes'])/1024:.1f} KB")
            logger.info(f"  Max: {max(self.timings['response_size_bytes'])/1024:.1f} KB")


async def trace_single_request():
    """Trace a single request in detail to see where time goes."""
    logger.info("\n" + "="*60)
    logger.info("TRACING SINGLE REQUEST IN DETAIL")
    logger.info("="*60)
    
    # Use httpx events to trace request lifecycle
    async def log_request_start(request):
        logger.info(f"  [REQUEST START] {request.method} {request.url}")
        
    async def log_request_end(request):
        logger.info(f"  [REQUEST END] {request.method} {request.url}")
        
    async def log_response_start(response):
        logger.info(f"  [RESPONSE START] Status: {response.status_code}")
        
    async def log_response_end(response):
        logger.info(f"  [RESPONSE END] Status: {response.status_code}")
    
    event_hooks = {
        "request": [log_request_start, log_request_end],
        "response": [log_response_start, log_response_end]
    }
    
    async with httpx.AsyncClient(event_hooks=event_hooks) as client:
        # Initialize
        logger.info("\nInitializing environment...")
        init_start = time.time()
        
        init_response = await client.post(
            "http://localhost:8901/env/CrafterClassic/initialize",
            json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}},
            timeout=30.0
        )
        
        init_time = time.time() - init_start
        env_id = init_response.json()["env_id"]
        logger.info(f"  Initialization took: {init_time:.3f}s")
        
        # Single step with detailed timing
        logger.info("\nExecuting single step...")
        
        payload = {
            "env_id": env_id,
            "action": {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]}
        }
        
        # Time each phase
        phases = {}
        
        # Phase 1: Serialize
        phase_start = time.time()
        json_data = json.dumps(payload)
        phases["serialize"] = time.time() - phase_start
        
        # Phase 2: Send request and wait for response
        phase_start = time.time()
        response = await client.post(
            "http://localhost:8901/env/CrafterClassic/step",
            content=json_data,
            headers={"Content-Type": "application/json"},
            timeout=30.0
        )
        phases["network"] = time.time() - phase_start
        
        # Phase 3: Read response
        phase_start = time.time()
        response_text = response.text
        phases["read_response"] = time.time() - phase_start
        
        # Phase 4: Parse JSON
        phase_start = time.time()
        response_data = json.loads(response_text)
        phases["deserialize"] = time.time() - phase_start
        
        # Print breakdown
        total_time = sum(phases.values())
        logger.info(f"\n  Total step time: {total_time:.3f}s")
        logger.info("  Breakdown:")
        for phase, duration in phases.items():
            percentage = (duration / total_time) * 100
            logger.info(f"    {phase}: {duration:.3f}s ({percentage:.1f}%)")
            
        # Cleanup
        await client.post(
            "http://localhost:8901/env/CrafterClassic/terminate",
            json={"env_id": env_id}
        )


async def main():
    """Run all profiling tests."""
    profiler = EnvironmentProfiler()
    
    # First trace a single request
    await trace_single_request()
    
    # Run profiling tests
    await profiler.profile_with_connection_reuse()
    await profiler.profile_payload_size_impact()
    await profiler.test_concurrent_environments()
    await profiler.profile_service_internals()
    
    # Print summary
    profiler.print_summary()
    
    # Final analysis
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS & RECOMMENDATIONS")
    logger.info("="*60)
    
    if profiler.timings["http_request_only"]:
        mean_network = statistics.mean(profiler.timings["http_request_only"])
        if mean_network > 1.0:
            logger.info("\n⚠️  Network latency is high (>1s). Possible causes:")
            logger.info("   - Service is overloaded")
            logger.info("   - Python GIL blocking with concurrent requests")
            logger.info("   - Inefficient service implementation")
            
    if profiler.timings["response_size_bytes"]:
        mean_size = statistics.mean(profiler.timings["response_size_bytes"])
        if mean_size > 50000:  # 50KB
            logger.info("\n⚠️  Large response payloads. Consider:")
            logger.info("   - Compressing responses")
            logger.info("   - Removing unnecessary fields from observations")
            logger.info("   - Using binary protocols instead of JSON")


if __name__ == "__main__":
    asyncio.run(main())