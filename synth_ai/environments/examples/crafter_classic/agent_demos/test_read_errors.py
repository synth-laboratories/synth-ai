#!/usr/bin/env python3
"""Test script to reproduce ReadError issues with Crafter environment service."""

import asyncio
import httpx
import time
import logging
from typing import Dict, List, Any, Optional
import json
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable httpx info logs to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ReadErrorTester:
    """Test for ReadError issues with the environment service."""
    
    def __init__(self, service_url: str = "http://localhost:8901"):
        self.service_url = service_url
        self.errors: List[Dict[str, Any]] = []
        self.success_count = 0
        self.failure_count = 0
        
    async def test_single_request(self, client: httpx.AsyncClient, test_name: str) -> Optional[Dict[str, Any]]:
        """Test a single request and capture any errors."""
        try:
            logger.info(f"Starting test: {test_name}")
            
            # 1. Initialize environment
            init_payload = {
                "initial_state": {},
                "config": {"area": [64, 64], "length": 100}
            }
            
            init_start = time.time()
            init_response = await client.post(
                f"{self.service_url}/env/CrafterClassic/initialize",
                json=init_payload,
                timeout=30.0
            )
            init_time = time.time() - init_start
            
            if init_response.status_code != 200:
                raise Exception(f"Initialize failed with status {init_response.status_code}: {init_response.text}")
                
            init_data = init_response.json()
            env_id = init_data["env_id"]
            logger.info(f"✅ Environment initialized in {init_time:.3f}s (env_id: {env_id})")
            
            # 2. Perform 5 steps
            for step in range(5):
                step_payload = {
                    "env_id": env_id,
                    "action": {
                        "tool_calls": [{
                            "tool": "interact",
                            "args": {"action": step % 17}
                        }]
                    }
                }
                
                step_start = time.time()
                try:
                    step_response = await client.post(
                        f"{self.service_url}/env/CrafterClassic/step",
                        json=step_payload,
                        timeout=30.0
                    )
                    step_time = time.time() - step_start
                    
                    if step_response.status_code != 200:
                        raise Exception(f"Step failed with status {step_response.status_code}: {step_response.text}")
                        
                    logger.debug(f"  Step {step+1} completed in {step_time:.3f}s")
                    
                except httpx.ReadError as e:
                    # This is what we're looking for!
                    error_info = {
                        "test_name": test_name,
                        "step": step,
                        "error_type": "ReadError",
                        "error_class": type(e).__name__,
                        "error_msg": str(e),
                        "traceback": traceback.format_exc(),
                        "time": time.time() - step_start
                    }
                    self.errors.append(error_info)
                    logger.error(f"❌ ReadError at step {step}: {e}")
                    raise
                    
                except httpx.TimeoutException as e:
                    error_info = {
                        "test_name": test_name,
                        "step": step,
                        "error_type": "Timeout",
                        "error_class": type(e).__name__,
                        "error_msg": str(e),
                        "time": time.time() - step_start
                    }
                    self.errors.append(error_info)
                    logger.error(f"❌ Timeout at step {step}: {e}")
                    raise
                    
            # 3. Terminate
            term_response = await client.post(
                f"{self.service_url}/env/CrafterClassic/terminate",
                json={"env_id": env_id},
                timeout=30.0
            )
            
            logger.info(f"✅ Test '{test_name}' completed successfully")
            self.success_count += 1
            return {"status": "success", "test": test_name}
            
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' failed: {type(e).__name__}: {e}")
            self.failure_count += 1
            if not any(err["test_name"] == test_name for err in self.errors):
                self.errors.append({
                    "test_name": test_name,
                    "error_type": "General",
                    "error_class": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": traceback.format_exc()
                })
            return {"status": "failed", "test": test_name, "error": str(e)}
            
    async def test_concurrent_requests(self, num_concurrent: int = 5):
        """Test multiple concurrent requests to stress the service."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {num_concurrent} concurrent requests")
        logger.info(f"{'='*60}")
        
        async with httpx.AsyncClient() as client:
            tasks = []
            for i in range(num_concurrent):
                task = self.test_single_request(client, f"concurrent_{i+1}")
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        return results
        
    async def test_rapid_sequential(self, num_requests: int = 10):
        """Test rapid sequential requests without delays."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {num_requests} rapid sequential requests")
        logger.info(f"{'='*60}")
        
        async with httpx.AsyncClient() as client:
            results = []
            for i in range(num_requests):
                result = await self.test_single_request(client, f"sequential_{i+1}")
                results.append(result)
                # No delay between requests
                
        return results
        
    async def test_large_payload(self):
        """Test with larger observation/state to see if size causes issues."""
        logger.info(f"\n{'='*60}")
        logger.info("Testing with large payload")
        logger.info(f"{'='*60}")
        
        async with httpx.AsyncClient() as client:
            # Initialize with a larger world
            init_payload = {
                "initial_state": {},
                "config": {"area": [128, 128], "length": 1000}  # Larger world
            }
            
            try:
                response = await client.post(
                    f"{self.service_url}/env/CrafterClassic/initialize",
                    json=init_payload,
                    timeout=60.0  # Longer timeout for larger world
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Check response size
                    response_size = len(response.text)
                    logger.info(f"✅ Large world initialized, response size: {response_size} bytes")
                    
                    # Try a step
                    env_id = data["env_id"]
                    step_response = await client.post(
                        f"{self.service_url}/env/CrafterClassic/step",
                        json={
                            "env_id": env_id,
                            "action": {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]}
                        },
                        timeout=30.0
                    )
                    step_size = len(step_response.text)
                    logger.info(f"✅ Step response size: {step_size} bytes")
                    
                    # Cleanup
                    await client.post(
                        f"{self.service_url}/env/CrafterClassic/terminate",
                        json={"env_id": env_id}
                    )
                    
                    return {"status": "success", "init_size": response_size, "step_size": step_size}
                    
            except Exception as e:
                logger.error(f"❌ Large payload test failed: {e}")
                self.errors.append({
                    "test_name": "large_payload",
                    "error_type": type(e).__name__,
                    "error_msg": str(e)
                })
                return {"status": "failed", "error": str(e)}
                
    async def test_connection_limits(self):
        """Test if connection pool limits cause issues."""
        logger.info(f"\n{'='*60}")
        logger.info("Testing connection pool limits")
        logger.info(f"{'='*60}")
        
        # Create client with specific connection limits
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(limits=limits) as client:
            # Fire off 20 requests to exceed connection limit
            tasks = []
            for i in range(20):
                task = self.test_single_request(client, f"connection_test_{i+1}")
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        connection_errors = [r for r in results if isinstance(r, Exception)]
        logger.info(f"Connection test results: {len(connection_errors)}/{len(results)} failed")
        
        return results
        
    def print_summary(self):
        """Print a summary of all errors found."""
        print(f"\n{'='*80}")
        print("🔍 ERROR SUMMARY")
        print(f"{'='*80}")
        print(f"Total tests: {self.success_count + self.failure_count}")
        print(f"Successful: {self.success_count}")
        print(f"Failed: {self.failure_count}")
        
        if self.errors:
            print(f"\n📋 Error Details ({len(self.errors)} errors):")
            print("-"*80)
            
            # Group errors by type
            error_types = {}
            for error in self.errors:
                error_type = error.get("error_type", "Unknown")
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error)
                
            for error_type, errors in error_types.items():
                print(f"\n{error_type} Errors ({len(errors)} occurrences):")
                for i, error in enumerate(errors[:3]):  # Show first 3 of each type
                    print(f"  {i+1}. Test: {error.get('test_name', 'N/A')}")
                    print(f"     Error: {error.get('error_msg', 'N/A')[:200]}")
                    if "step" in error:
                        print(f"     Step: {error['step']}")
                if len(errors) > 3:
                    print(f"  ... and {len(errors) - 3} more")
                    
        else:
            print("\n✅ No errors detected!")
            
        print(f"\n{'='*80}\n")


async def check_service_status():
    """Quick check of service status and capabilities."""
    logger.info("Checking service status...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Check health endpoint
            response = await client.get("http://localhost:8901/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Service is healthy: {data}")
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
                
            # Check if service supports keep-alive
            logger.info(f"Connection headers: {response.headers.get('connection', 'Not specified')}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Cannot connect to service: {e}")
            return False


async def main():
    """Run all tests to reproduce ReadError."""
    # First check if service is running
    if not await check_service_status():
        logger.error("Service is not available. Please ensure it's running on port 8901")
        return
        
    tester = ReadErrorTester()
    
    # Run different test scenarios
    logger.info("\n🧪 Starting ReadError reproduction tests...\n")
    
    # Test 1: Concurrent requests (most likely to cause issues)
    await tester.test_concurrent_requests(num_concurrent=10)
    
    # Test 2: Rapid sequential requests
    await tester.test_rapid_sequential(num_requests=5)
    
    # Test 3: Large payload test
    await tester.test_large_payload()
    
    # Test 4: Connection pool limits
    await tester.test_connection_limits()
    
    # Print summary
    tester.print_summary()
    
    # Save detailed error log
    if tester.errors:
        with open("read_error_details.json", "w") as f:
            json.dump(tester.errors, f, indent=2)
        logger.info("💾 Detailed error log saved to read_error_details.json")


if __name__ == "__main__":
    asyncio.run(main())