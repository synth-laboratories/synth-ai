#!/usr/bin/env python3
"""
Simple test of Synth backend using synth_ai package.
"""

import os
from synth_ai.lm import LM

# Configuration
MODEL_ID = "gpt-4o-mini"  # Use OpenAI model for testing
API_KEY = os.getenv("OPENAI_API_KEY", "")

def test_basic():
    """Test basic LM functionality."""
    print("Testing basic LM functionality...")
    
    # Initialize LM
    lm = LM(
        model=MODEL_ID,
        api_key=API_KEY,
        temperature=0.7,
        max_tokens=100
    )
    
    # Test with proper message format
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    print(f"Model: {MODEL_ID}")
    print(f"Sending messages: {messages}")
    
    try:
        # Use respond with messages parameter
        response = lm.respond(messages=messages)
        print(f"Response: {response}")
        print("✓ Test passed!")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_user_message():
    """Test with user_message parameter."""
    print("\nTesting with user_message parameter...")
    
    lm = LM(
        model=MODEL_ID,
        api_key=API_KEY,
        temperature=0.5
    )
    
    try:
        # Use respond with user_message
        response = lm.respond(
            user_message="What is the capital of France?",
            system_message="You are a helpful geography assistant."
        )
        print(f"Response: {response}")
        print("✓ Test passed!")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_synth_model():
    """Test with Synth model."""
    print("\nTesting with Synth model...")
    
    synth_model = "Qwen/Qwen2.5-0.5B-Instruct"
    synth_api_key = os.getenv("SYNTH_API_KEY", "")
    
    lm = LM(
        model=synth_model,
        api_key=synth_api_key,
        temperature=0.7,
        max_tokens=50
    )
    
    try:
        response = lm.respond(
            user_message="Hello, how are you?",
            system_message="You are a friendly assistant."
        )
        print(f"Model: {synth_model}")
        print(f"Response: {response}")
        print("✓ Synth model test passed!")
        return True
    except Exception as e:
        print(f"✗ Synth model test failed: {e}")
        print("Note: This might be due to API key or model availability")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Synth AI Package Tests")
    print("="*60)
    
    # Run tests
    results = []
    results.append(test_basic())
    results.append(test_user_message())
    results.append(test_synth_model())
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")