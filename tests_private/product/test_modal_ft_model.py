#!/usr/bin/env python3
"""Test Modal fine-tuned model with a simple chat completion request."""

import asyncio
import httpx
import os
import json

# Modal/Synth API configuration
MODAL_BASE_URL = os.getenv('MODAL_BASE_URL', os.getenv('SYNTH_BASE_URL', 'https://synth-laboratories--unified-ft-service-fastapi-app.modal.run'))
MODAL_API_KEY = os.getenv('MODAL_API_KEY', os.getenv('SYNTH_API_KEY', 'sk-test-11111111111111111111111111111111'))

# Fine-tuned model ID
FT_MODEL = "ft:qwen2-5-7b-instruct:org-test123:20250730165348:modal-1753894173"


async def test_chat_completion():
    """Test the fine-tuned model with a simple chat completion."""
    print(f"🤖 Testing fine-tuned model: {FT_MODEL}")
    print(f"📡 API Base URL: {MODAL_BASE_URL}")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Prepare the request
        messages = [
            {"role": "system", "content": "You are CrafterAgent playing Crafter survival environment."},
            {"role": "user", "content": "I see a tree, stone, and table in my vicinity. What actions should I take to progress in Crafter?"}
        ]
        
        payload = {
            "model": FT_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        headers = {
            "Authorization": f"Bearer {MODAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        print("📤 Sending request...")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print()
        
        try:
            # Try the standard v1/chat/completions endpoint
            response = await client.post(
                f"{MODAL_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            print(f"📥 Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print()
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success! Response:")
                print(json.dumps(result, indent=2))
                
                # Extract the assistant's message
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    print(f"\n🤖 Assistant: {content}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
                # Try to parse error details
                try:
                    error_data = response.json()
                    if 'detail' in error_data:
                        print(f"\nError details: {error_data['detail']}")
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ Request failed: {type(e).__name__}: {e}")
            
    print("\n" + "=" * 60)
    print("💡 Troubleshooting tips:")
    print("1. Check if the model ID is correct")
    print("2. Verify the API key and base URL")
    print("3. Try using just the base model name: Qwen/Qwen2.5-7B-Instruct")
    print("4. Check Modal documentation for fine-tuned model usage")


async def test_with_base_model():
    """Test with base model for comparison."""
    print("\n🔄 Testing with base model for comparison...")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        headers = {
            "Authorization": f"Bearer {MODAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            response = await client.post(
                f"{MODAL_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                print("✅ Base model works!")
            else:
                print(f"❌ Base model also failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Base model request failed: {e}")


async def check_available_models():
    """Check what models are available."""
    print("\n🔍 Checking available models...")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {MODAL_API_KEY}",
        }
        
        try:
            # Try to list models
            response = await client.get(
                f"{MODAL_BASE_URL}/v1/models",
                headers=headers
            )
            
            if response.status_code == 200:
                models = response.json()
                print("📋 Available models:")
                print(json.dumps(models, indent=2))
            else:
                print(f"❌ Could not list models: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Failed to list models: {e}")


async def main():
    """Run all tests."""
    # Test fine-tuned model
    await test_chat_completion()
    
    # Test base model
    await test_with_base_model()
    
    # Check available models
    await check_available_models()


if __name__ == "__main__":
    asyncio.run(main())