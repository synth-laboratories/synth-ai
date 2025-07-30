#!/usr/bin/env python3
"""
Script to replicate the issue with Qwen 7B model inference
"""

import httpx
import asyncio
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000/api/v1/learning"
API_KEY = "test-api-key"  # Replace with your actual API key
MODEL = "Qwen/Qwen2.5-7B-Instruct"

async def test_base_model_inference():
    """Test base model inference through the local learning service."""
    
    print(f"🔍 Testing Base Model Inference")
    print(f"Time: {datetime.now()}")
    print(f"Endpoint: {BASE_URL}/v1/chat/completions")
    print(f"Model: {MODEL}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Hello, can you help me play Crafter?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        print("\n📤 Request:")
        print(f"Headers: {json.dumps(headers, indent=2)}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            print("\n⏳ Sending request...")
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            print(f"\n📥 Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n✅ Success!")
                print(f"Response: {json.dumps(data, indent=2)}")
                
                # Extract the assistant's message
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    print(f"\n🤖 Assistant says: {content}")
                    
                    # Check if it's the "not implemented" message
                    if "Base model inference not implemented" in content:
                        print("\n⚠️  WARNING: The service returned a 'not implemented' message!")
                        print("This means the service is responding but base models aren't supported yet.")
            else:
                print(f"\n❌ Error Response:")
                print(f"Body: {response.text}")
                
        except httpx.ConnectError as e:
            print(f"\n❌ Connection Error: Could not connect to {BASE_URL}")
            print(f"Error: {e}")
            print("\nPossible issues:")
            print("1. The backend service is not running on port 8000")
            print("2. The learning service is not properly configured")
            print("3. The proxy route /api/v1/learning is not set up")
            
        except httpx.TimeoutException as e:
            print(f"\n❌ Timeout Error: Request timed out after 30 seconds")
            print(f"Error: {e}")
            
        except Exception as e:
            print(f"\n❌ Unexpected Error: {type(e).__name__}")
            print(f"Error: {e}")


async def test_crafter_integration():
    """Test the full Crafter + LLM integration."""
    
    print("\n\n🎮 Testing Crafter Integration")
    print("=" * 60)
    
    # First check if Crafter service is running
    crafter_url = "http://localhost:8901"
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            print(f"\n1️⃣ Checking Crafter service at {crafter_url}...")
            response = await client.get(f"{crafter_url}/health")
            if response.status_code == 200:
                print("✅ Crafter service is running!")
            else:
                print(f"⚠️  Crafter service returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Crafter service is not running on port 8901")
            print(f"   Error: {e}")
            print("\n   To start Crafter service:")
            print("   uv run python -m uvicorn synth_ai.environments.service.app:app --host 0.0.0.0 --port 8901")
    
    # Test creating an environment
    if True:  # You can set to False to skip environment creation
        print(f"\n2️⃣ Testing environment creation...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{crafter_url}/create_env",
                    json={
                        "instance_id": "test_instance",
                        "render_mode": "rgb_array",
                        "difficulty": "easy",
                        "seed": 42
                    }
                )
                if response.status_code == 200:
                    print("✅ Environment created successfully!")
                else:
                    print(f"❌ Failed to create environment: {response.text}")
            except Exception as e:
                print(f"❌ Error creating environment: {e}")


def main():
    """Run all tests."""
    print("🚀 Replicating Qwen 7B Model Inference Issue")
    print("=" * 60)
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test base model inference
        loop.run_until_complete(test_base_model_inference())
        
        # Test Crafter integration
        loop.run_until_complete(test_crafter_integration())
        
    finally:
        loop.close()
    
    print("\n\n📋 Summary:")
    print("=" * 60)
    print("This script tested:")
    print("1. Base model inference through the learning service proxy")
    print("2. Crafter environment service availability")
    print("\nTo fix the issues:")
    print("1. Ensure the backend is running: cd backend && uvicorn app.main:app --reload")
    print("2. Ensure the learning service is configured and running")
    print("3. Start Crafter service: uv run python -m uvicorn synth_ai.environments.service.app:app --host 0.0.0.0 --port 8901")


if __name__ == "__main__":
    main()