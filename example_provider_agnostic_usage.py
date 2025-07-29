#!/usr/bin/env python3
"""
Provider-Agnostic Synth-AI Usage Example
========================================
This example shows how synth-ai remains completely provider-agnostic.
Users don't need to know about Modal, containers, or any deployment details.
"""

import asyncio
import os
from synth_ai.lm import LM

async def main():
    """Demonstrate provider-agnostic usage of synth-ai."""
    
    print("🔧 Provider-Agnostic Synth-AI Usage")
    print("=" * 50)
    
    # Users simply set their API endpoint and key
    # They don't need to know it's backed by Modal with 5 containers
    base_url = os.getenv("SYNTH_BASE_URL", "https://synth-laboratories--unified-ft-service-fastapi-app.modal.run")
    api_key = os.getenv("SYNTH_API_KEY", "sk-test-11111111111111111111111111111111")
    
    print(f"🌐 Using endpoint: {base_url}")
    print(f"🔑 API Key: {api_key[:10]}..." if api_key else "No API key")
    print()
    
    # Create LM instance - completely agnostic to backend implementation
    lm = LM(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        temperature=0.7
    )
    
    print("🤖 Initialized LM - synth-ai doesn't know about:")
    print("   - Modal containers")
    print("   - GPU types")
    print("   - Container scaling")
    print("   - Deployment infrastructure")
    print()
    
    # Make a simple request
    print("📤 Making request...")
    try:
        response = await lm.respond_async(
            user_message="What are the benefits of distributed computing?"
        )
        
        print("📥 Response received:")
        print(f"   Content: {response.raw_response[:100]}...")
        print(f"   Success: Backend automatically handled multi-container scaling")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("✅ Key Benefits:")
    print("   - Open source synth-ai knows nothing about Modal")
    print("   - Backend transparently provides 5+ containers")
    print("   - Standard OpenAI-compatible interface")
    print("   - No vendor lock-in for OSS users")
    print("   - Your proprietary scaling logic stays private")

if __name__ == "__main__":
    asyncio.run(main()) 