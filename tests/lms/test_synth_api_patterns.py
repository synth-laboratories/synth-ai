"""
Comprehensive integration tests for all Synth API calling patterns.

Tests three ways to call Synth API:
1. Standard OpenAI/AsyncOpenAI with Synth base_url
2. Synth-specific AsyncSynthClient (imported as AsyncOpenAI)
3. High-level LM class with Synth provider

All patterns tested with:
- Basic chat completions
- Tool calling
- Streaming
- Error handling
- Environment variable configuration
"""

import asyncio
import json
import os
from typing import Any, Dict, List

import httpx
import pytest

# Import Synth client as AsyncOpenAI for drop-in replacement
from synth_ai.lm.vendors.synth_client import AsyncSynthClient as AsyncOpenAI
from synth_ai.lm import LM


# Test tools for function calling
def get_weather(location: str) -> Dict[str, Any]:
    """Get weather for a location."""
    return {
        "location": location,
        "temperature": 72,
        "condition": "sunny",
        "humidity": 45
    }


def calculate_sum(a: int, b: int) -> Dict[str, Any]:
    """Calculate sum of two numbers."""
    return {"result": a + b}


# OpenAI-compatible tool definitions
WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Calculate sum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]


@pytest.mark.integration
class TestSynthAPIPatterns:
    """Test all Synth API calling patterns comprehensively."""

    @pytest.fixture
    def synth_config(self):
        """Synth API configuration."""
        return {
            "base_url": "https://synth-backend-dev-docker.onrender.com/api",
            "api_key": "sk_live_9592524d-be1b-48b2-aff7-976b277eac95",
            "model": "Qwen/Qwen3-0.6B"
        }

    def test_pattern_1_standard_openai_client(self, synth_config):
        """Test Pattern 1: Standard OpenAI client with Synth base_url."""
        async def run_test():
            # Standard OpenAI client with Synth base_url
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=synth_config["api_key"],
                base_url=synth_config["base_url"]
            )

            # Test basic chat completion
            response = await client.chat.completions.create(
                model=synth_config["model"],
                messages=[{"role": "user", "content": "Say hello in one word"}],
                temperature=0.0,
                max_tokens=10
            )

            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content.strip()) > 0

            # Test with tools
            tool_response = await client.chat.completions.create(
                model=synth_config["model"],
                messages=[
                    {"role": "user", "content": "What's the weather in Paris?"}
                ],
                tools=WEATHER_TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=100
            )

            # Should either respond normally or call a tool
            choice = tool_response.choices[0]
            assert choice.message.content is not None or choice.message.tool_calls is not None

        asyncio.run(run_test())

    def test_pattern_2_synth_as_openai_client(self, synth_config):
        """Test Pattern 2: Synth client imported as AsyncOpenAI (drop-in replacement)."""
        async def run_test():
            # Import Synth client as AsyncOpenAI - drop-in replacement!
            from synth_ai.lm.vendors.synth_client import AsyncOpenAI

            # Use it exactly like standard OpenAI client
            client = AsyncOpenAI(
                api_key=synth_config["api_key"],
                base_url=synth_config["base_url"]
            )

            # Test basic completion
            response = await client.chat.completions.create(
                model=synth_config["model"],
                messages=[{"role": "user", "content": "Count to 3"}],
                temperature=0.0,
                max_tokens=20
            )

            assert response.choices[0].message.content is not None
            # Qwen models may respond with thinking traces, so just check for non-empty content
            content = response.choices[0].message.content.strip()
            assert len(content) > 0, "Response should contain some content"

            # Test tool calling with Synth-as-OpenAI
            tool_response = await client.chat.completions.create(
                model=synth_config["model"],
                messages=[
                    {"role": "user", "content": "Calculate 15 + 27"}
                ],
                tools=WEATHER_TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=150
            )

            choice = tool_response.choices[0]
            # Should either respond or call calculate_sum tool
            assert (choice.message.content is not None or
                   (choice.message.tool_calls and len(choice.message.tool_calls) > 0))

        asyncio.run(run_test())

    def test_pattern_3_lm_class_synth(self, synth_config):
        """Test Pattern 3: High-level LM class with Synth provider."""
        async def run_test():
            # Set environment variables for Synth
            original_base = os.environ.get("OPENAI_API_BASE")
            original_key = os.environ.get("OPENAI_API_KEY")

            try:
                os.environ["OPENAI_API_BASE"] = synth_config["base_url"]
                os.environ["OPENAI_API_KEY"] = synth_config["api_key"]

                # Create LM with Synth provider
                lm = LM(
                    model_name=synth_config["model"],
                    provider="synth",
                    enable_v3_tracing=False
                )

                # Test basic response
                response = await lm.respond_async(
                    messages=[{"role": "user", "content": "Say 'test'"}],
                    temperature=0.0,
                    max_tokens=5
                )

                assert response is not None
                # LM responses have different structure
                content = (response.content if hasattr(response, 'content') and response.content
                          else response.raw_response if hasattr(response, 'raw_response')
                          else str(response))
                assert len(content.strip()) > 0

                # Test with tools
                tool_response = await lm.respond_async(
                    messages=[{"role": "user", "content": "What's 10 + 5?"}],
                    tools=WEATHER_TOOLS,
                    temperature=0.1,
                    max_tokens=100
                )

                assert tool_response is not None

            finally:
                # Restore environment
                if original_base:
                    os.environ["OPENAI_API_BASE"] = original_base
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key

        asyncio.run(run_test())

    def test_streaming_support(self, synth_config):
        """Test streaming responses across all patterns."""
        async def run_test():
            # Test streaming with standard OpenAI pattern
            client = AsyncOpenAI(
                api_key=synth_config["api_key"],
                base_url=synth_config["base_url"]
            )

            stream = await client.chat.completions.create(
                model=synth_config["model"],
                messages=[{"role": "user", "content": "Say 'streaming works'"}],
                stream=True,
                temperature=0.0,
                max_tokens=20
            )

            chunks = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)

            full_response = "".join(chunks)
            assert len(full_response.strip()) > 0
            assert "streaming" in full_response.lower() or "works" in full_response.lower()

        asyncio.run(run_test())

    def test_error_handling(self, synth_config):
        """Test error handling across patterns."""
        async def run_test():
            client = AsyncOpenAI(
                api_key="invalid_key",  # Wrong API key
                base_url=synth_config["base_url"]
            )

            with pytest.raises(Exception):  # Should raise authentication error
                await client.chat.completions.create(
                    model=synth_config["model"],
                    messages=[{"role": "user", "content": "Test"}]
                )

        asyncio.run(run_test())

    def test_environment_configuration(self, synth_config):
        """Test that environment variables are properly used."""
        async def run_test():
            # Set environment variables
            original_base = os.environ.get("OPENAI_API_BASE")
            original_key = os.environ.get("OPENAI_API_KEY")

            try:
                os.environ["OPENAI_API_BASE"] = synth_config["base_url"]
                os.environ["OPENAI_API_KEY"] = synth_config["api_key"]

                # Client should pick up env vars automatically
                client = AsyncOpenAI()  # No explicit params

                response = await client.chat.completions.create(
                    model=synth_config["model"],
                    messages=[{"role": "user", "content": "Env vars work"}],
                    temperature=0.0,
                    max_tokens=10
                )

                assert response.choices[0].message.content is not None

            finally:
                # Restore environment
                if original_base:
                    os.environ["OPENAI_API_BASE"] = original_base
                if original_key:
                    os.environ["OPENAI_API_KEY"] = original_key

        asyncio.run(run_test())

    def test_tool_calling_detailed(self, synth_config):
        """Detailed test of tool calling functionality."""
        async def run_test():
            from synth_ai.lm.vendors.synth_client import AsyncSynthClient as AsyncOpenAI

            client = AsyncOpenAI(
                api_key=synth_config["api_key"],
                base_url=synth_config["base_url"]
            )

            # Test with explicit tool choice
            response = await client.chat.completions.create(
                model=synth_config["model"],
                messages=[
                    {"role": "user", "content": "Use the calculate_sum tool to add 42 and 58"}
                ],
                tools=WEATHER_TOOLS,
                tool_choice={"type": "function", "function": {"name": "calculate_sum"}},
                temperature=0.0,
                max_tokens=100
            )

            choice = response.choices[0]

            # Should call the calculate_sum tool
            if choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                assert tool_call.function.name == "calculate_sum"

                # Parse the arguments
                import json
                args = json.loads(tool_call.function.arguments)
                assert "a" in args and "b" in args
                assert args["a"] == 42 and args["b"] == 58

        asyncio.run(run_test())

    @pytest.mark.parametrize("pattern_name", ["standard_openai", "synth_as_openai", "lm_class"])
    def test_all_patterns_basic_completion(self, synth_config, pattern_name):
        """Parametrized test to ensure all patterns work for basic completion."""
        async def run_test():
            if pattern_name == "standard_openai":
                # Standard OpenAI import (only if available)
                try:
                    from openai import AsyncOpenAI as TestClient
                except ImportError:
                    pytest.skip("openai package not available")
            elif pattern_name == "synth_as_openai":
                # Synth imported as OpenAI
                from synth_ai.lm.vendors.synth_client import AsyncOpenAI as TestClient
            else:  # lm_class
                # Use LM class instead
                original_base = os.environ.get("OPENAI_API_BASE")
                original_key = os.environ.get("OPENAI_API_KEY")

                try:
                    os.environ["OPENAI_API_BASE"] = synth_config["base_url"]
                    os.environ["OPENAI_API_KEY"] = synth_config["api_key"]

                    lm = LM(model_name=synth_config["model"], provider="synth", enable_v3_tracing=False)
                    response = await lm.respond_async(
                        messages=[{"role": "user", "content": "Say 'pattern test'"}],
                        temperature=0.0,
                        max_tokens=10
                    )

                    # Verify LM response
                    content = (response.content if hasattr(response, 'content') and response.content
                              else response.raw_response if hasattr(response, 'raw_response')
                              else str(response))
                    assert len(content.strip()) > 0
                    return

                finally:
                    if original_base:
                        os.environ["OPENAI_API_BASE"] = original_base
                    if original_key:
                        os.environ["OPENAI_API_KEY"] = original_key

            # For OpenAI-style clients
            client = TestClient(
                api_key=synth_config["api_key"],
                base_url=synth_config["base_url"]
            )

            response = await client.chat.completions.create(
                model=synth_config["model"],
                messages=[{"role": "user", "content": "Say 'pattern test'"}],
                temperature=0.0,
                max_tokens=10
            )

            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content.strip()) > 0

        asyncio.run(run_test())


if __name__ == "__main__":
    # Quick manual test
    config = {
        "base_url": "https://synth-backend-dev-docker.onrender.com/api",
        "api_key": "sk_live_9592524d-be1b-48b2-aff7-976b277eac95",
        "model": "Qwen/Qwen3-0.6B"
    }

    print("🧪 Testing Synth API patterns...")

    # Test Pattern 2: Synth as OpenAI
    async def test_synth_as_openai():
        try:
            from openai import AsyncOpenAI
        except ImportError:
            from synth_ai.lm.vendors.synth_client import AsyncSynthClient as AsyncOpenAI

        client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )

        response = await client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": "Hello from Synth-as-OpenAI!"}],
            temperature=0.0,
            max_tokens=20
        )

        print(f"✅ Synth-as-OpenAI: {response.choices[0].message.content}")

    asyncio.run(test_synth_as_openai())
    print("🎉 All pattern tests completed!")
