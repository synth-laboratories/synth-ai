#!/usr/bin/env python3
"""
Basic test script to verify LM class with v2 tracing works correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Disable v1 logging
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["SYNTH_LOGGING"] = "false"
os.environ["SYNTH_TRACING_MODE"] = "v2"  # Use v2 only mode

from synth_ai.lm.core.main_v2 import LM
from synth_ai.tracing_v2.session_tracer import SessionTracer
from pydantic import BaseModel


class WeatherInfo(BaseModel):
    """Example structured output model."""
    location: str
    temperature: int
    condition: str


async def test_basic_lm_v2():
    """Test basic LM functionality with v2 tracing."""
    print("🧪 Testing LM with V2 Tracing\n")
    
    # Create session tracer
    tracer = SessionTracer()
    
    # Test 1: Basic text response
    print("Test 1: Basic text response")
    print("-" * 40)
    
    lm = LM(
        model_name="gpt-3.5-turbo",
        formatting_model_name="gpt-3.5-turbo", 
        temperature=0.7,
        synth_logging=False,  # Disable v1
        session_tracer=tracer,
        system_id="test_lm_basic",
        enable_v2_tracing=True
    )
    
    async with tracer.start_session("test_session_1"):
        response = await lm.respond_async(
            system_message="You are a helpful assistant. Be concise.",
            user_message="What is 2+2?",
            turn_number=0
        )
        
        print(f"Response: {response.raw_response}")
        print(f"Has structured output: {response.structured_output is not None}")
        print()
    
    # Test 2: Structured output
    print("\nTest 2: Structured output")
    print("-" * 40)
    
    async with tracer.start_session("test_session_2"):
        response = await lm.respond_async(
            system_message="Extract weather information from the text.",
            user_message="It's 72 degrees and sunny in San Francisco today.",
            response_model=WeatherInfo,
            turn_number=0
        )
        
        print(f"Raw response: {response.raw_response[:50]}...")
        print(f"Structured output: {response.structured_output}")
        if response.structured_output:
            print(f"  Location: {response.structured_output.location}")
            print(f"  Temperature: {response.structured_output.temperature}")
            print(f"  Condition: {response.structured_output.condition}")
        print()
    
    # Test 3: Multi-turn conversation
    print("\nTest 3: Multi-turn conversation")
    print("-" * 40)
    
    async with tracer.start_session("test_session_3"):
        # Turn 0
        response1 = await lm.respond_async(
            system_message="You are a math tutor. Be encouraging.",
            user_message="Can you help me understand fractions?",
            turn_number=0
        )
        print(f"Turn 0: {response1.raw_response[:100]}...")
        
        # Turn 1 - Follow up
        messages = [
            {"role": "system", "content": "You are a math tutor. Be encouraging."},
            {"role": "user", "content": "Can you help me understand fractions?"},
            {"role": "assistant", "content": response1.raw_response},
            {"role": "user", "content": "What is 1/2 + 1/4?"}
        ]
        
        response2 = await lm.respond_async(
            messages=messages,
            turn_number=1
        )
        print(f"Turn 1: {response2.raw_response[:100]}...")
    
    # Save traces
    print("\n📊 Saving traces...")
    trace_dir = Path("./test_traces_v2")
    trace_dir.mkdir(exist_ok=True)
    
    for i, session_id in enumerate(["test_session_1", "test_session_2", "test_session_3"]):
        trace_file = trace_dir / f"test_{i+1}.json"
        tracer.save_session(session_id, str(trace_file))
        print(f"  Saved {session_id} to {trace_file}")
    
    print("\n✅ All tests completed successfully!")


async def test_error_handling():
    """Test error handling with v2 tracing."""
    print("\n🧪 Testing Error Handling\n")
    
    tracer = SessionTracer()
    
    # Use a model that might have rate limits
    lm = LM(
        model_name="gpt-4",  # More likely to hit rate limits
        formatting_model_name="gpt-3.5-turbo",
        temperature=0,
        synth_logging=False,
        session_tracer=tracer,
        system_id="test_lm_errors",
        enable_v2_tracing=True
    )
    
    async with tracer.start_session("test_errors"):
        try:
            # This might fail with rate limits or other errors
            response = await lm.respond_async(
                system_message="You are a helpful assistant.",
                user_message="Generate a very long story about space exploration. Make it at least 1000 words.",
                turn_number=0
            )
            print(f"Success: Got {len(response.raw_response)} chars")
        except Exception as e:
            print(f"Expected error caught: {type(e).__name__}: {str(e)[:100]}")
            # V2 tracing should have captured the error


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_basic_lm_v2())
    
    # Optionally test error handling
    # asyncio.run(test_error_handling())