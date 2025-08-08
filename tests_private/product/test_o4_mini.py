#!/usr/bin/env python3
"""
Test o4-mini model via OpenAI Responses API.
"""

import asyncio
import os
from synth_ai.lm import LM


async def test_o4_mini():
    """Test o4-mini with the Responses API."""
    print("Testing o4-mini via OpenAI Responses API")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return
    
    # Initialize with o4-mini
    print("\n1. Initializing LM with o4-mini...")
    lm = LM(model="o4-mini", vendor="openai")
    
    # Test 1: Basic response
    print("\n2. Testing basic response...")
    try:
        r1 = await lm.respond_async(
            system_message="You are a helpful math tutor.",
            user_message="What is 15 * 23? Please explain your calculation step by step."
        )
        
        print(f"✓ Response received: {r1.raw_response[:200]}...")
        print(f"  - Response ID: {r1.response_id}")
        print(f"  - API Type: {r1.api_type}")
        print(f"  - Has reasoning: {r1.reasoning is not None}")
        if r1.reasoning:
            print(f"  - Reasoning length: {len(r1.reasoning)} chars")
            print(f"  - Reasoning preview: {r1.reasoning[:100]}...")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Thread continuation
    print("\n3. Testing thread continuation...")
    try:
        r2 = await lm.respond_async(
            system_message="You are a helpful math tutor.",
            user_message="Now divide the result by 7."
        )
        
        print(f"✓ Continuation received: {r2.raw_response[:200]}...")
        print(f"  - Response ID: {r2.response_id}")
        print(f"  - Thread maintained: {r2.response_id != r1.response_id}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(test_o4_mini())