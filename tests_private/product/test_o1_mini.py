#!/usr/bin/env python3
"""
Test o1-mini model via OpenAI API with Responses API support.
"""

import asyncio
import os
from synth_ai.lm import LM


async def test_o1_mini():
    """Test o1-mini with the new Responses API implementation."""
    print("Testing o1-mini via OpenAI API")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return
    
    # Initialize with o1-mini
    print("\n1. Initializing LM with o1-mini...")
    lm = LM(model="o1-mini", vendor="openai")
    
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
    
    # Test 2: Thread continuation (auto-store)
    print("\n3. Testing thread continuation with auto-store...")
    try:
        r2 = await lm.respond_async(
            system_message="You are a helpful math tutor.",
            user_message="Now divide the result by 7."
        )
        
        print(f"✓ Continuation received: {r2.raw_response[:200]}...")
        print(f"  - Response ID: {r2.response_id}")
        print(f"  - Previous ID used: {lm._last_response_id}")
        print(f"  - Thread maintained: {r2.response_id != r1.response_id}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Manual thread management
    print("\n4. Testing manual thread management...")
    lm_manual = LM(model="o1-mini", vendor="openai", auto_store_responses=False)
    
    try:
        r3 = await lm_manual.respond_async(
            system_message="You are a creative writer.",
            user_message="Write the first line of a mystery novel."
        )
        print(f"✓ Initial response: {r3.raw_response[:100]}...")
        print(f"  - Response ID: {r3.response_id}")
        
        # Continue with manual ID
        r4 = await lm_manual.respond_async(
            system_message="You are a creative writer.",
            user_message="Continue with the second line.",
            previous_response_id=r3.response_id
        )
        print(f"✓ Manual continuation: {r4.raw_response[:100]}...")
        print(f"  - New Response ID: {r4.response_id}")
        print(f"  - Used previous ID: {r3.response_id}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(test_o1_mini())