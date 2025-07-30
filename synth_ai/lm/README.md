# Synth AI Language Model Interface

A unified interface for interacting with multiple LLM providers (OpenAI, Synth) with minimal code changes.

## Features

- 🔄 **Unified Interface**: Switch between providers with just a configuration change
- 🔥 **Smart Warmup**: Automatic model warmup with caching for Synth backend
- 🔒 **Secure Configuration**: All sensitive data loaded from environment variables
- ⚡ **Async & Sync**: Both async and sync clients available
- 🛡️ **Retry Logic**: Built-in exponential backoff for rate limits
- 🎯 **OpenAI Compatible**: Drop-in replacement for OpenAI API

## Installation

```bash
pip install python-dotenv httpx
```

## Configuration

Create a `.env` file in your project root:

```bash
# Synth Configuration
SYNTH_BASE_URL=https://your-synth-url.modal.run
SYNTH_API_KEY=your-synth-api-key

# Optional: Modal compatibility
MODAL_BASE_URL=https://your-modal-url.modal.run
MODAL_API_KEY=your-modal-api-key

# OpenAI Configuration (if using)
OPENAI_API_KEY=your-openai-api-key

# Optional settings
SYNTH_TIMEOUT=120
SYNTH_MAX_RETRIES=3
```

**Important**: Add `.env` to your `.gitignore` to keep credentials secure!

## Quick Start

### Basic Usage

```python
import asyncio
from synth_ai.lm import create_provider

async def main():
    # Create provider (config from environment)
    provider = create_provider("synth")
    
    # Warm up model (cached for 10 minutes)
    await provider.warmup("Qwen/Qwen2.5-7B-Instruct")
    
    # Make request
    response = await provider.create_chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print(response["choices"][0]["message"]["content"])
    
    # Clean up
    await provider.close()

asyncio.run(main())
```

### One-Shot Request

```python
from synth_ai.lm import create_chat_completion_async

response = await create_chat_completion_async(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Provider Switching

```python
from synth_ai.lm import UnifiedLMClient

async with UnifiedLMClient() as client:
    # Use Synth
    response = await client.create_chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}],
        provider="synth"
    )
    
    # Use OpenAI
    response = await client.create_chat_completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}],
        provider="openai"
    )
```

## API Reference

### Providers

#### `create_provider(provider_type: str, **config) -> UnifiedLMProvider`

Create a provider instance.

- `provider_type`: "openai" or "synth"
- `**config`: Optional configuration overrides

### Warmup

#### `warmup_synth_model(model_name: str, config: Optional[SynthConfig] = None, max_attempts: int = 30, force: bool = False) -> bool`

Warm up a Synth model with intelligent caching.

- `model_name`: Name of the model to warm up
- `config`: Optional config (loads from env if not provided)
- `max_attempts`: Maximum polling attempts
- `force`: Force warmup even if cached

### Clients

#### AsyncSynthClient

Async client with OpenAI-compatible interface:

```python
async with AsyncSynthClient() as client:
    response = await client.chat_completions_create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[...],
        temperature=0.7,
        max_tokens=1000,
        tools=[...],  # Optional tool definitions
        tool_choice="auto"
    )
```

#### SyncSynthClient

Synchronous version for non-async code:

```python
with SyncSynthClient() as client:
    response = client.chat_completions_create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[...]
    )
```

## Advanced Usage

### Custom Configuration

```python
from synth_ai.lm import SynthConfig, SynthProvider

# Create custom config
config = SynthConfig(
    base_url="https://custom-url.com",
    api_key="custom-key",
    timeout=60.0,
    max_retries=5
)

# Use with provider
provider = SynthProvider(config)
```

### Tool/Function Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = await provider.create_chat_completion(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Check for tool calls
if response["choices"][0]["message"].get("tool_calls"):
    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    print(f"Function: {tool_call['function']['name']}")
    print(f"Arguments: {tool_call['function']['arguments']}")
```

## Migration Guide

### From OpenAI

```python
# Before (OpenAI)
from openai import AsyncOpenAI
client = AsyncOpenAI()
response = await client.chat.completions.create(...)

# After (Synth)
from synth_ai.lm import create_provider
provider = create_provider("synth")
await provider.warmup(model)  # Added step
response = await provider.create_chat_completion(...)
```

### From Direct Modal/Synth Calls

```python
# Before (Direct HTTP)
response = httpx.post(
    "https://synth-url/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={...}
)

# After (Synth Client)
from synth_ai.lm import create_chat_completion_async
response = await create_chat_completion_async(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[...]
)
```

## Best Practices

1. **Always use environment variables** for sensitive configuration
2. **Warm up models** before first use to avoid timeouts
3. **Use context managers** (`async with`) for proper cleanup
4. **Handle rate limits** - the client has built-in retry logic
5. **Cache warmup status** - models stay warm for ~10 minutes

## Troubleshooting

### "SYNTH_BASE_URL not found in environment"

Make sure your `.env` file exists and contains:
```
SYNTH_BASE_URL=https://your-synth-url.modal.run
SYNTH_API_KEY=your-api-key
```

### Timeout errors

1. Increase timeout in config: `SYNTH_TIMEOUT=180`
2. Ensure model is warmed up before use
3. Check if the model size requires longer loading time

### Rate limit errors

The client automatically retries with exponential backoff. You can adjust:
```
SYNTH_MAX_RETRIES=5
```

## Examples

See the `examples/` directory for complete examples:
- `test_synth_api.py` - Comprehensive usage examples
- More examples coming soon...

## Support

For issues or questions, please check the main Synth AI documentation or create an issue in the repository.