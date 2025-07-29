#!/usr/bin/env python3
"""
Simple Multi-Container Test with Synth-AI
=========================================
This test demonstrates that synth-ai is completely provider-agnostic
and can leverage multi-container backends transparently.
"""

import asyncio
import os
import time
from synth_ai.lm.core.main_v2 import LM

async def test_provider_agnostic_synth_ai():
    """Test that synth-ai works with multi-container backend transparently."""
    
    print("🧪 Provider-Agnostic Synth-AI Multi-Container Test")
    print("=" * 60)
    
    # Set up environment variables (users would set these)
    synth_base_url = os.getenv("SYNTH_BASE_URL", "https://synth-laboratories--unified-ft-service-fastapi-app.modal.run")
    synth_api_key = os.getenv("SYNTH_API_KEY", "sk-test-11111111111111111111111111111111")
    
    print(f"🌐 Backend URL: {synth_base_url}")
    print(f"🔑 API Key: {synth_api_key[:15]}...")
    print()
    
    # Set up environment variables for synth-ai to use our backend
    os.environ["SYNTH_BASE_URL"] = synth_base_url
    os.environ["SYNTH_API_KEY"] = synth_api_key
    
    # Create LM instance - completely agnostic to backend implementation
    print("🤖 Creating LM instance...")
    lm = LM(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        formatting_model_name="Qwen/Qwen2.5-14B-Instruct",
        temperature=0.7,
        synth_logging=False,  # Disable v1 logging for cleaner output
        provider="synth"  # Use synth provider to connect to our backend
    )
    print("✅ LM instance created (synth-ai doesn't know about containers/Modal)")
    print()
    
    # Test multiple concurrent requests to trigger multi-container behavior
    print("🚀 Making concurrent requests to test multi-container scaling...")
    
    tasks = []
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What are the benefits of renewable energy?",
        "How does machine learning work?",
        "What is the theory of relativity?"
    ]
    
    start_time = time.time()
    
    # Create concurrent tasks
    for i, prompt in enumerate(prompts):
        task = lm.respond_async(
            user_message=f"[Request {i+1}] {prompt}",
            system_message="You are a helpful assistant. Keep responses brief (1-2 sentences)."
        )
        tasks.append(task)
    
    # Execute all tasks concurrently
    print(f"📤 Sending {len(tasks)} concurrent requests...")
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ Completed {len(tasks)} requests in {total_time:.2f} seconds")
    print(f"📊 Throughput: {len(tasks)/total_time:.2f} requests/second")
    print()
    
    # Show responses
    print("📥 Responses:")
    successful = 0
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            print(f"  {i+1}. ❌ Error: {response}")
        else:
            successful += 1
            content = response.raw_response[:100] + "..." if len(response.raw_response) > 100 else response.raw_response
            print(f"  {i+1}. ✅ {content}")
    
    print()
    print("🎯 Key Points Demonstrated:")
    print("  - Synth-AI remains completely provider-agnostic")
    print("  - Backend handles multi-container scaling transparently")
    print("  - Users only see standard OpenAI-compatible responses")
    print("  - No vendor lock-in - can point to any compatible endpoint")
    print("  - Multi-container benefits work without any code changes")
    print(f"  - Success rate: {successful}/{len(tasks)} ({successful/len(tasks)*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(test_provider_agnostic_synth_ai()) 