#!/usr/bin/env python3
"""
Quick test to verify improved retry logic for ReadErrors
"""

import asyncio
import httpx
from test_crafter_react_agent_openai import retry_http_request, MAX_RETRIES, READ_ERROR_RETRIES

async def test_retry_logic():
    """Test the retry logic with simulated ReadErrors"""
    
    print(f"Testing retry configuration:")
    print(f"  MAX_RETRIES: {MAX_RETRIES}")
    print(f"  READ_ERROR_RETRIES: {READ_ERROR_RETRIES}")
    print()
    
    # Create a mock client that always fails with ReadError
    class MockClient:
        def __init__(self):
            self.call_count = 0
            
        async def request(self, method, url, **kwargs):
            self.call_count += 1
            if self.call_count <= 3:
                raise httpx.ReadError("Simulated ReadError")
            # Succeed on 4th attempt
            return httpx.Response(200, json={"status": "ok"})
    
    client = MockClient()
    
    try:
        print("Testing ReadError retry (should succeed on 4th attempt)...")
        response = await retry_http_request(client, "GET", "/test")
        print(f"✅ Success after {client.call_count} attempts")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test with persistent ReadError
    class AlwaysFailClient:
        async def request(self, method, url, **kwargs):
            raise httpx.ReadError("Persistent ReadError")
    
    client2 = AlwaysFailClient()
    
    try:
        print("\nTesting persistent ReadError (should fail after 5 attempts)...")
        response = await retry_http_request(client2, "GET", "/test")
        print("❌ Should not reach here")
    except httpx.ReadError as e:
        print(f"✅ Correctly failed with ReadError after max retries")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(test_retry_logic())