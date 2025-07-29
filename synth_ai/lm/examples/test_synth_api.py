"""
Example usage of the Synth LM API.
Demonstrates various ways to use the unified interface.
"""

import asyncio
import logging
from synth_ai.lm import (
    create_provider,
    UnifiedLMClient,
    warmup_synth_model,
    create_chat_completion_async
)

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)


async def example_basic_usage():
    """Basic usage with automatic configuration from environment."""
    print("\n=== Basic Usage Example ===")
    
    # Create a Synth provider (config loaded from .env automatically)
    provider = create_provider("synth")
    
    # Warm up the model
    model = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Warming up {model}...")
    success = await provider.warmup(model)
    
    if not success:
        print("Failed to warm up model!")
        return
    
    # Use the model
    response = await provider.create_chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"Response: {response['choices'][0]['message']['content']}")
    
    # Clean up
    await provider.close()


async def example_unified_client():
    """Using the unified client for easy provider switching."""
    print("\n=== Unified Client Example ===")
    
    async with UnifiedLMClient(default_provider="synth") as client:
        # Warm up model
        model = "Qwen/Qwen2.5-7B-Instruct"
        await client.warmup(model)
        
        # Use with default provider (Synth)
        response = await client.create_chat_completion(
            model=model,
            messages=[
                {"role": "user", "content": "Count from 1 to 5."}
            ],
            temperature=0.0
        )
        
        print(f"Synth response: {response['choices'][0]['message']['content']}")
        
        # You can also override the provider per request
        # response = await client.create_chat_completion(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": "Count from 1 to 5."}],
        #     provider="openai"  # Use OpenAI for this request
        # )


async def example_direct_client():
    """Using the Synth client directly for more control."""
    print("\n=== Direct Client Example ===")
    
    from synth_ai.lm import AsyncSynthClient, SynthConfig
    
    # You can create a custom config if needed
    # config = SynthConfig(
    #     base_url="https://your-custom-url.com",
    #     api_key="your-api-key",
    #     timeout=60.0
    # )
    
    # Or just use the default from environment
    async with AsyncSynthClient() as client:
        response = await client.chat_completions_create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "user", "content": "Write a haiku about programming."}
            ],
            temperature=0.9,
            max_tokens=50
        )
        
        print(f"Haiku: {response['choices'][0]['message']['content']}")


async def example_one_shot():
    """One-shot request without managing clients."""
    print("\n=== One-Shot Example ===")
    
    # First warm up the model
    await warmup_synth_model("Qwen/Qwen2.5-7B-Instruct")
    
    # Then make a single request
    response = await create_chat_completion_async(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
            {"role": "user", "content": "How's the weather today?"}
        ],
        temperature=0.8
    )
    
    print(f"Pirate says: {response['choices'][0]['message']['content']}")


async def example_tool_calling():
    """Example with tool/function calling."""
    print("\n=== Tool Calling Example ===")
    
    provider = create_provider("synth")
    
    # Define a simple tool
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }]
    
    response = await provider.create_chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"}
        ],
        tools=tools,
        tool_choice="auto"
    )
    
    # Check if the model wants to call a tool
    message = response['choices'][0]['message']
    if 'tool_calls' in message and message['tool_calls']:
        tool_call = message['tool_calls'][0]
        print(f"Model wants to call: {tool_call['function']['name']}")
        print(f"With arguments: {tool_call['function']['arguments']}")
    else:
        print(f"Model response: {message['content']}")
    
    await provider.close()


async def main():
    """Run all examples."""
    try:
        await example_basic_usage()
        await example_unified_client()
        await example_direct_client()
        await example_one_shot()
        await example_tool_calling()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have set up your .env file with:")
        print("SYNTH_BASE_URL=<your-synth-url>")
        print("SYNTH_API_KEY=<your-api-key>")


if __name__ == "__main__":
    asyncio.run(main())