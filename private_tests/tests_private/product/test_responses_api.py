#!/usr/bin/env python3
"""
Test script for OpenAI Responses API implementation in Synth AI.

This script demonstrates:
1. Basic Responses API usage with thread management
2. Auto-chaining of responses
3. Manual thread management with branching
4. Accessing reasoning traces (for o1 models)
"""

import asyncio
import os
from synth_ai.lm import LM


async def test_auto_chaining():
    """Test automatic response chaining."""
    print("\n=== Testing Auto-Chaining ===")
    
    # Initialize with a model that supports Responses API
    lm = LM(model="o4-mini", auto_store_responses=True)
    
    # First message - no previous response
    r1 = await lm.respond_async(
        system_message="You are a helpful assistant.",
        user_message="Hello! Can you count to 3?"
    )
    print(f"Response 1: {r1.raw_response[:100]}...")
    print(f"Response ID: {r1.response_id}")
    print(f"API Type: {r1.api_type}")
    
    # Second message - should auto-use r1's response_id
    r2 = await lm.respond_async(
        system_message="You are a helpful assistant.",
        user_message="Now count to 5."
    )
    print(f"\nResponse 2: {r2.raw_response[:100]}...")
    print(f"Response ID: {r2.response_id}")
    print(f"Chained from: {lm._last_response_id}")


async def test_manual_threading():
    """Test manual thread management with branching."""
    print("\n=== Testing Manual Threading ===")
    
    # Initialize with manual control
    lm = LM(model="o4-mini", auto_store_responses=False)
    
    # Initial message
    r1 = await lm.respond_async(
        system_message="You are a creative storyteller.",
        user_message="Start a story about a dragon."
    )
    print(f"Initial story: {r1.raw_response[:100]}...")
    print(f"Response ID: {r1.response_id}")
    
    # Branch A - continue with adventure
    r2a = await lm.respond_async(
        system_message="You are a creative storyteller.",
        user_message="Continue with an adventure.",
        previous_response_id=r1.response_id
    )
    print(f"\nBranch A: {r2a.raw_response[:100]}...")
    
    # Branch B - continue with romance
    r2b = await lm.respond_async(
        system_message="You are a creative storyteller.",
        user_message="Continue with a romance twist.",
        previous_response_id=r1.response_id
    )
    print(f"\nBranch B: {r2b.raw_response[:100]}...")


async def test_reasoning_model():
    """Test with a reasoning model (o1) if available."""
    print("\n=== Testing Reasoning Model ===")
    
    try:
        lm = LM(model="o4-mini")
        
        r = await lm.respond_async(
            system_message="You are a helpful assistant.",
            user_message="What is 23 * 47? Show your work."
        )
        
        print(f"Answer: {r.raw_response}")
        if r.reasoning:
            print(f"Reasoning trace available: {len(r.reasoning)} characters")
            print(f"Reasoning preview: {r.reasoning[:200]}...")
        else:
            print("No reasoning trace available")
            
    except Exception as e:
        print(f"Reasoning model test skipped: {e}")


async def test_fallback_to_chat():
    """Test fallback to chat completions when Responses API not available."""
    print("\n=== Testing Fallback to Chat Completions ===")
    
    # Use a model that doesn't support Responses API
    lm = LM(model="gpt-3.5-turbo")
    
    r = await lm.respond_async(
        system_message="You are a helpful assistant.",
        user_message="What's 2+2?"
    )
    
    print(f"Response: {r.raw_response}")
    print(f"API Type: {r.api_type}")
    print(f"Response ID: {r.response_id}")  # Should be None for chat completions


async def main():
    """Run all tests."""
    print("OpenAI Responses API Test Suite - o4-mini Only")
    print("==============================================")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Some tests may fail.")
    
    try:
        await test_auto_chaining()
    except Exception as e:
        print(f"Auto-chaining test failed: {e}")
    
    try:
        await test_manual_threading()
    except Exception as e:
        print(f"Manual threading test failed: {e}")
    
    try:
        await test_reasoning_model()
    except Exception as e:
        print(f"Reasoning model test failed: {e}")
    
    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    asyncio.run(main())