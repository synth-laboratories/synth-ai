#!/usr/bin/env python3
"""Test if the retry mechanism is causing the slowness."""

import asyncio
import httpx
import time
import random
import logging
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Copy the retry configuration from the OpenAI test
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 10.0  # seconds
HTTP_TIMEOUT = 30.0  # seconds


async def retry_http_request_with_logging(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> Any:
    """Same retry logic but with detailed logging."""
    last_exception = None
    total_start = time.time()
    
    for attempt in range(MAX_RETRIES):
        attempt_start = time.time()
        
        try:
            # Calculate delay with exponential backoff and jitter
            if attempt > 0:
                delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
                jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
                total_delay = delay + jitter
                logger.info(f"  Retry {attempt}: Waiting {total_delay:.3f}s (delay: {delay}s + jitter: {jitter:.3f}s)")
                await asyncio.sleep(total_delay)
            
            # Make the request with timeout
            logger.info(f"  Attempt {attempt + 1}/{MAX_RETRIES}: {method} {url}")
            request_start = time.time()
            response = await client.request(method, url, timeout=HTTP_TIMEOUT, **kwargs)
            request_time = time.time() - request_start
            logger.info(f"  Response: {response.status_code} in {request_time:.3f}s")
            
            # Check if response is successful
            if response.status_code < 500:  # Don't retry client errors (4xx)
                total_time = time.time() - total_start
                logger.info(f"  ✅ Success after {attempt + 1} attempts, total time: {total_time:.3f}s")
                return response
            
            # For server errors (5xx), continue retrying
            last_exception = Exception(f"HTTP {response.status_code}: {response.text[:100]}")
            logger.warning(f"  Server error: {response.status_code}")
            
        except Exception as e:
            attempt_time = time.time() - attempt_start
            last_exception = e
            logger.error(f"  Exception: {type(e).__name__}: {str(e)[:100]} (after {attempt_time:.3f}s)")
    
    # All retries failed
    total_time = time.time() - total_start
    logger.error(f"  ❌ Failed after {MAX_RETRIES} attempts, total time: {total_time:.3f}s")
    raise last_exception


async def test_normal_request():
    """Test a normal request that should succeed immediately."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Normal request (should succeed immediately)")
    logger.info("="*60)
    
    async with httpx.AsyncClient(base_url="http://localhost:8901") as client:
        # Initialize environment
        response = await client.post(
            "/env/CrafterClassic/initialize",
            json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}}
        )
        env_id = response.json()["env_id"]
        
        # Test step with retry
        start = time.time()
        response = await retry_http_request_with_logging(
            client, "POST", "/env/CrafterClassic/step",
            json={
                "env_id": env_id,
                "action": {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]}
            }
        )
        total_time = time.time() - start
        
        logger.info(f"\nTotal time for step: {total_time:.3f}s")
        
        # Cleanup
        await client.post("/env/CrafterClassic/terminate", json={"env_id": env_id})


async def test_with_artificial_failures():
    """Test with a proxy that simulates failures."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Simulating intermittent failures")
    logger.info("="*60)
    
    # Create a simple proxy that fails sometimes
    failure_count = 0
    
    class FailingTransport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            nonlocal failure_count
            
            # Fail the first 2 requests to trigger retries
            if "step" in str(request.url) and failure_count < 2:
                failure_count += 1
                logger.info(f"  🔥 Simulating failure {failure_count}")
                raise httpx.ConnectError("Simulated connection error")
                
            # Otherwise forward the request normally
            return await super().handle_async_request(request)
    
    transport = FailingTransport()
    async with httpx.AsyncClient(base_url="http://localhost:8901", transport=transport) as client:
        # Initialize environment
        response = await client.post(
            "/env/CrafterClassic/initialize",
            json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}}
        )
        env_id = response.json()["env_id"]
        
        # Test step with retry (should fail twice then succeed)
        start = time.time()
        try:
            response = await retry_http_request_with_logging(
                client, "POST", "/env/CrafterClassic/step",
                json={
                    "env_id": env_id,
                    "action": {"tool_calls": [{"tool": "interact", "args": {"action": 0}}]}
                }
            )
            total_time = time.time() - start
            logger.info(f"\nTotal time with retries: {total_time:.3f}s")
        except Exception as e:
            total_time = time.time() - start
            logger.error(f"\nFailed after {total_time:.3f}s: {e}")
            
        # Cleanup
        try:
            await client.post("/env/CrafterClassic/terminate", json={"env_id": env_id})
        except:
            pass


async def test_multiple_actions_timing():
    """Test how multiple actions affect timing."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Multiple actions in sequence")
    logger.info("="*60)
    
    async with httpx.AsyncClient(base_url="http://localhost:8901") as client:
        # Initialize environment
        response = await client.post(
            "/env/CrafterClassic/initialize",
            json={"initial_state": {}, "config": {"area": [64, 64], "length": 100}}
        )
        env_id = response.json()["env_id"]
        
        # Simulate the OpenAI test's action loop
        action_ints = [0, 1, 2, 3, 4]  # 5 actions
        
        logger.info(f"\nExecuting {len(action_ints)} actions...")
        env_start_time = time.time()
        
        for i, action_int in enumerate(action_ints):
            logger.info(f"\nAction {i+1}/{len(action_ints)}:")
            step_start = time.time()
            
            response = await retry_http_request_with_logging(
                client, "POST", "/env/CrafterClassic/step",
                json={
                    "env_id": env_id,
                    "action": {"tool_calls": [{"tool": "interact", "args": {"action": action_int}}]}
                }
            )
            
            step_time = time.time() - step_start
            logger.info(f"Action {i+1} took: {step_time:.3f}s")
            
        env_end_time = time.time()
        total_time = env_end_time - env_start_time
        
        logger.info(f"\nTotal time for {len(action_ints)} actions: {total_time:.3f}s")
        logger.info(f"Average per action: {total_time/len(action_ints):.3f}s")
        
        # Cleanup
        await client.post("/env/CrafterClassic/terminate", json={"env_id": env_id})


async def main():
    """Run all tests."""
    # Test 1: Normal requests
    await test_normal_request()
    
    # Test 2: With simulated failures
    await test_with_artificial_failures()
    
    # Test 3: Multiple actions
    await test_multiple_actions_timing()
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    logger.info("\nIf Test 1 is fast (~15-30ms) but the OpenAI test shows 5-7s, then:")
    logger.info("1. The service is experiencing intermittent failures causing retries")
    logger.info("2. The timing includes multiple actions (check how many actions per step)")
    logger.info("3. There's a network/proxy issue between the test and service")
    logger.info("\nCheck the service logs for 5xx errors or exceptions!")


if __name__ == "__main__":
    asyncio.run(main())