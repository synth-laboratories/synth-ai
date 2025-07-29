#!/usr/bin/env python3
"""
Test the Synth LM implementation.
Run this to verify the installation works correctly.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_config():
    """Test configuration loading."""
    print("=== Testing Configuration ===")
    
    try:
        from synth_ai.lm.config import SynthConfig
        config = SynthConfig.from_env()
        print(f"✅ Config loaded: {config}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False


async def test_warmup():
    """Test model warmup."""
    print("\n=== Testing Warmup ===")
    
    try:
        from synth_ai.lm import warmup_synth_model
        
        model = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Warming up {model}...")
        
        # Test with verbose=False to reduce output
        success = await warmup_synth_model(model, verbose=False, max_attempts=2)
        
        if success:
            print("✅ Warmup successful")
            
            # Test cached warmup
            success2 = await warmup_synth_model(model, verbose=False)
            print("✅ Cached warmup works")
            return True
        else:
            print("❌ Warmup failed")
            return False
            
    except Exception as e:
        print(f"❌ Warmup error: {e}")
        return False


async def test_basic_request():
    """Test basic chat completion."""
    print("\n=== Testing Basic Request ===")
    
    try:
        from synth_ai.lm import create_provider
        
        provider = create_provider("synth")
        
        response = await provider.create_chat_completion(
            model="Qwen/Qwen2.5-0.5B-Instruct",  # Use smaller model for faster test
            messages=[
                {"role": "user", "content": "Say 'test successful' if you're working."}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        content = response["choices"][0]["message"]["content"]
        print(f"✅ Response received: {content}")
        
        await provider.close()
        return True
        
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False


async def test_unified_client():
    """Test unified client."""
    print("\n=== Testing Unified Client ===")
    
    try:
        from synth_ai.lm import UnifiedLMClient
        
        async with UnifiedLMClient(default_provider="synth") as client:
            response = await client.create_chat_completion(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                messages=[{"role": "user", "content": "Count: 1, 2, 3"}],
                temperature=0.0,
                max_tokens=10
            )
            
            content = response["choices"][0]["message"]["content"]
            print(f"✅ Unified client works: {content}")
            return True
            
    except Exception as e:
        print(f"❌ Unified client error: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing Synth LM Implementation\n")
    
    # Check environment
    if not os.getenv("SYNTH_BASE_URL") and not os.getenv("MODAL_BASE_URL"):
        print("❌ Environment not configured!")
        print("\nPlease create a .env file with:")
        print("SYNTH_BASE_URL=<your-synth-url>")
        print("SYNTH_API_KEY=<your-api-key>")
        return
    
    # Run tests
    results = []
    
    # Test 1: Configuration
    results.append(await test_config())
    
    # Test 2: Warmup (skip if config failed)
    if results[0]:
        results.append(await test_warmup())
    
    # Test 3: Basic request (skip if warmup failed)
    if len(results) > 1 and results[1]:
        results.append(await test_basic_request())
    
    # Test 4: Unified client
    if len(results) > 2 and results[2]:
        results.append(await test_unified_client())
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary:")
    print(f"  Configuration: {'✅' if results[0] else '❌'}")
    if len(results) > 1:
        print(f"  Warmup: {'✅' if results[1] else '❌'}")
    if len(results) > 2:
        print(f"  Basic Request: {'✅' if results[2] else '❌'}")
    if len(results) > 3:
        print(f"  Unified Client: {'✅' if results[3] else '❌'}")
    
    if all(results):
        print("\n✅ All tests passed! The Synth LM implementation is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())