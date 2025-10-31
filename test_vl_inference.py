#!/usr/bin/env python3
"""Test script to verify Qwen3-VL inference endpoint is working."""
import asyncio
import base64
import json
import os

import httpx

# Try to load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Create a simple test image (small red square)
def create_test_image():
    # Create a minimal PNG - 1x1 red pixel
    # PNG signature + minimal IHDR + minimal IDAT + IEND
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(png_data).decode('utf-8')

async def test_inference():
    """Test the prod inference endpoint with a vision request."""
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        print("ERROR: SYNTH_API_KEY not set")
        return
    
    inference_url = "https://synth-laboratories-dev--learning-v2-service-fastapi-app.modal.run/chat/completions"
    
    # Create test image data URL
    image_base64 = create_test_image()
    image_data_url = f"data:image/png;base64,{image_base64}"
    
    payload = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond with tool calls when requested."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image? Use the execute_sequence tool if you see anything."},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_sequence",
                    "description": "Execute a sequence of actions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "button": {"type": "string", "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B"]},
                                        "frames": {"type": "integer", "minimum": 1, "maximum": 120}
                                    },
                                    "required": ["button", "frames"]
                                },
                                "minItems": 1,
                                "maxItems": 20
                            }
                        },
                        "required": ["actions"]
                    }
                }
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "execute_sequence"}},
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,  # Reduced to avoid token budget issues
        "thinking_budget": 0  # Required by backend
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-GPU-Preference": "H100"  # Prefer H100 GPU (default for Qwen3-VL-8B)
    }
    
    print(f"Testing inference endpoint: {inference_url}")
    print(f"API Key present: {bool(api_key)}")
    print(f"Model: {payload['model']}")
    print(f"Request payload keys: {list(payload.keys())}")
    
    # Retry logic to handle cold starts and timeouts
    max_retries = 10
    retry_delay = 5  # seconds
    use_gpu_header = True  # Track whether to use GPU preference header
    
    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*60}")
        print(f"Attempt {attempt}/{max_retries}")
        print(f"{'='*60}")
        
        # Remove GPU preference header if we've had device errors
        current_headers = {k: v for k, v in headers.items() if k != "X-GPU-Preference"} if not use_gpu_header else headers
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
            try:
                print("Making request... (this may take a while on cold start)")
                resp = await client.post(inference_url, json=payload, headers=current_headers)
                print(f"\n✓ Response status: {resp.status_code}")
                
                if resp.status_code != 200:
                    print(f"Error response: {resp.text[:500]}")
                    if resp.status_code == 400:
                        # Device errors might be temporary - retry anyway
                        if "Device string must not be empty" in resp.text and attempt < max_retries:
                            print("Device selection error (may be temporary). Retrying...")
                            await asyncio.sleep(retry_delay)
                            continue
                        # Other 400 errors - don't retry
                        return
                    # Other errors - retry
                    if attempt < max_retries:
                        print(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        continue
                    return
                
                data = resp.json()
                print("\n✓ Response received!")
                print(f"Response keys: {list(data.keys())}")
                
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    print(f"\n✓ Message keys: {list(message.keys())}")
                    content = message.get('content', '')
                    if content:
                        print(f"Content: {content[:200]}")
                    
                    tool_calls = message.get("tool_calls", [])
                    print(f"\n✓ Tool calls: {len(tool_calls)}")
                    if tool_calls:
                        print("\n🎉 SUCCESS! Tool calls received:")
                        for i, tc in enumerate(tool_calls):
                            print(f"  Tool call {i+1}:")
                            print(f"    ID: {tc.get('id')}")
                            print(f"    Type: {tc.get('type')}")
                            func = tc.get("function", {})
                            print(f"    Function: {func.get('name')}")
                            args = func.get('arguments', '')
                            print(f"    Arguments: {args[:200]}")
                    else:
                        print("⚠️  No tool calls in response")
                        print(f"  Full message: {json.dumps(message, indent=2)[:500]}")
                else:
                    print("\n⚠️  No choices in response")
                    print(f"Full response: {json.dumps(data, indent=2)[:500]}")
                
                # Success - exit retry loop
                print(f"\n{'='*60}")
                print("✅ Request completed successfully!")
                print(f"{'='*60}")
                return
                
            except httpx.ReadTimeout:
                print(f"\n⚠️  Timeout (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    print("\n❌ Max retries reached. Last attempt timed out.")
                    return
                    
            except Exception as e:
                print(f"\n⚠️  Exception occurred: {type(e).__name__}: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    import traceback
                    traceback.print_exc()
                    return

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_inference())

