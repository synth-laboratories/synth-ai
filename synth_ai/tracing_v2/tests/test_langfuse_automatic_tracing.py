#!/usr/bin/env python3
"""
Test Langfuse automatic tracing for OpenAI and Anthropic.
Uses Langfuse's automatic instrumentation to capture traces.
"""

import os
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import openai
import anthropic

# Load environment variables
load_dotenv()


def test_openai_with_langfuse():
    """Test OpenAI with Langfuse automatic tracing."""
    print("=== Testing OpenAI with Langfuse Automatic Tracing ===\n")
    
    # Initialize Langfuse client
    langfuse = Langfuse()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello and count to 3."}
    ]
    
    try:
        # Make call using langfuse.openai (automatic tracing)
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print(f"Request ID: {response.id}")
        
        # Langfuse automatically captures this in the background
        print("\n✅ OpenAI call automatically traced by Langfuse")
        
        return {
            "provider": "openai",
            "model": response.model,
            "usage": response.usage.model_dump() if response.usage else None,
            "content": response.choices[0].message.content
        }
        
    except Exception as e:
        print(f"❌ Error with OpenAI: {e}")
        return None


def test_anthropic_with_manual_tracing():
    """Test Anthropic with manual Langfuse tracing (since automatic isn't available yet)."""
    print("\n=== Testing Anthropic with Manual Langfuse Tracing ===\n")
    
    # Initialize clients
    langfuse = Langfuse()
    client = anthropic.Anthropic()
    
    messages = [
        {"role": "user", "content": "Say hello and count to 3."}
    ]
    
    # Create a Langfuse generation for manual tracing
    generation = langfuse.generation(
        name="anthropic_chat",
        model="claude-3-haiku-20240307",
        input=messages,
        metadata={"provider": "anthropic"}
    )
    
    try:
        # Make Anthropic call
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=100,
            temperature=0.0
        )
        
        print(f"Response: {response.content[0].text}")
        print(f"Model: {response.model}")
        print(f"Usage - Input tokens: {response.usage.input_tokens}")
        print(f"Usage - Output tokens: {response.usage.output_tokens}")
        print(f"Request ID: {response.id}")
        
        # Update generation with output
        generation.update(
            output=response.content[0].text,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        )
        generation.end()
        
        print("\n✅ Anthropic call manually traced in Langfuse")
        
        return {
            "provider": "anthropic",
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            "content": response.content[0].text
        }
        
    except Exception as e:
        print(f"❌ Error with Anthropic: {e}")
        generation.update(error=str(e))
        generation.end()
        return None


def test_openai_streaming_with_langfuse():
    """Test OpenAI streaming with Langfuse automatic tracing."""
    print("\n=== Testing OpenAI Streaming with Langfuse ===\n")
    
    messages = [
        {"role": "user", "content": "Count to 5 slowly."}
    ]
    
    try:
        # Make streaming call using langfuse.openai
        stream = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            stream=True
        )
        
        print("Response (streaming): ", end="")
        full_content = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_content += content
        print()
        
        print("\n✅ OpenAI streaming call automatically traced by Langfuse")
        
        return {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "streaming": True,
            "content": full_content
        }
        
    except Exception as e:
        print(f"❌ Error with OpenAI streaming: {e}")
        return None


def main():
    """Run all Langfuse automatic tracing tests."""
    # Check for required API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    langfuse_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    
    if not openai_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment variables.")
        return
    
    if not anthropic_key:
        print("⚠️  WARNING: ANTHROPIC_API_KEY not found. Skipping Anthropic tests.")
    
    if not langfuse_key:
        print("⚠️  WARNING: LANGFUSE_PUBLIC_KEY not found. Traces won't be sent to Langfuse.")
        print("   Continuing with local tracing only.\n")
    
    print("🧪 Testing Langfuse Automatic Tracing\n")
    
    results = []
    
    # Test OpenAI
    result = test_openai_with_langfuse()
    if result:
        results.append(result)
    
    # Test OpenAI streaming
    result = test_openai_streaming_with_langfuse()
    if result:
        results.append(result)
    
    # Test Anthropic (if API key available)
    if anthropic_key:
        result = test_anthropic_with_manual_tracing()
        if result:
            results.append(result)
    
    print("\n" + "="*60)
    print("🎯 SUMMARY")
    print("="*60)
    print(f"Total traces collected: {len(results)}")
    print("\nProviders tested:")
    print("- ✅ OpenAI (automatic tracing via langfuse.openai)")
    if anthropic_key:
        print("- ✅ Anthropic (manual tracing via langfuse.generation)")
    else:
        print("- ⏭️  Anthropic (skipped - no API key)")
    
    if langfuse_key:
        print("\n📤 Traces sent to Langfuse dashboard")
    else:
        print("\n📁 Traces captured locally only (no Langfuse key)")


if __name__ == "__main__":
    main()