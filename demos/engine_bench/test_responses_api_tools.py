#!/usr/bin/env python3
"""
Standalone test script to verify OpenAI Responses API tool handling with streaming.

This tests whether tools are being sent correctly and whether the model responds with tool calls.
"""

import asyncio
import json
import os

import httpx

# OpenAI API key from environment
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable required")

# Test tools - simple file read/write tools like OpenCode uses
TEST_TOOLS = [
    {
        "type": "function",
        "name": "read",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string", "description": "The path to the file to read"}
            },
            "required": ["filePath"],
        },
    },
    {
        "type": "function",
        "name": "write",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string", "description": "The path to the file to write"},
                "content": {"type": "string", "description": "The content to write to the file"},
            },
            "required": ["filePath", "content"],
        },
    },
    {
        "type": "function",
        "name": "edit",
        "description": "Edit a file by replacing old content with new content",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string", "description": "The path to the file to edit"},
                "old": {"type": "string", "description": "The old content to replace"},
                "new": {"type": "string", "description": "The new content to insert"},
            },
            "required": ["filePath", "old", "new"],
        },
    },
    {
        "type": "function",
        "name": "bash",
        "description": "Run a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run"}},
            "required": ["command"],
        },
    },
]

# Test prompt that should trigger tool use
TEST_PROMPT = """You are a coding assistant with access to file tools.

The user wants you to create a simple hello world Python file.

Use the write tool to create a file called hello.py with the content:
print("Hello, World!")

You MUST use the write tool to accomplish this task."""


async def test_responses_api_streaming():
    """Test the Responses API with streaming and tools."""
    print("=" * 80)
    print("TEST: OpenAI Responses API with Streaming + Tools")
    print("=" * 80)

    endpoint = "https://api.openai.com/v1/responses"
    model = "gpt-4o-mini"  # Use a model we know works

    payload = {
        "model": model,
        "input": [{"role": "user", "content": TEST_PROMPT}],
        "tools": TEST_TOOLS,
        "tool_choice": "auto",
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    print("\n📤 REQUEST:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Model: {model}")
    print(f"  Tools: {len(TEST_TOOLS)}")
    print(f"  Tool names: {[t['name'] for t in TEST_TOOLS]}")
    print("  Stream: True")
    print("  tool_choice: auto")

    print("\n📋 PAYLOAD (tools section):")
    print(json.dumps(payload["tools"], indent=2)[:500] + "...")

    print("\n⏳ Sending request...")

    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client,
        client.stream("POST", endpoint, json=payload, headers=headers) as response,
    ):
        print("\n📥 RESPONSE:")
        print(f"  Status: {response.status_code}")
        print(f"  Headers: {dict(response.headers)}")

        if response.status_code != 200:
            body = await response.aread()
            print(f"  ERROR Body: {body.decode()}")
            return

        print("\n🔄 STREAMING CHUNKS:")
        chunk_count = 0
        tool_calls_found = []

        async for line in response.aiter_lines():
            if not line.strip():
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    print("\n  [DONE] received")
                    break
                try:
                    event = json.loads(data)
                    chunk_count += 1

                    # Check for different event types
                    event_type = event.get("type", "unknown")

                    if (
                        chunk_count <= 5
                        or "function_call" in str(event)
                        or "tool" in str(event).lower()
                    ):
                        print(f"\n  Chunk {chunk_count}: type={event_type}")
                        print(f"    {json.dumps(event, indent=2)[:300]}...")

                    # Look for tool calls in the response
                    if "output" in event:
                        for item in event.get("output", []):
                            if item.get("type") == "function_call":
                                tool_calls_found.append(item)
                                print(f"\n  🔧 TOOL CALL FOUND: {item.get('name')}")
                                print(f"    Arguments: {item.get('arguments', {})}")

                    # Also check for tool_calls in delta format
                    if "delta" in event:
                        delta = event["delta"]
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                print(f"\n  🔧 TOOL CALL (delta): {tc}")

                except json.JSONDecodeError as e:
                    print(f"  [JSON Error] {e}: {data[:100]}")
        print("\n" + "=" * 80)
        print("📊 SUMMARY:")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Tool calls found: {len(tool_calls_found)}")
        for tc in tool_calls_found:
            print(f"    - {tc.get('name')}: {tc.get('arguments', {})}")

        if not tool_calls_found:
            print("\n  ⚠️ WARNING: No tool calls in response!")
            print("  The model may have responded with text instead of using tools.")


async def test_responses_api_non_streaming():
    """Test the Responses API WITHOUT streaming for comparison."""
    print("\n" + "=" * 80)
    print("TEST: OpenAI Responses API WITHOUT Streaming (for comparison)")
    print("=" * 80)

    endpoint = "https://api.openai.com/v1/responses"
    model = "gpt-4o-mini"

    payload = {
        "model": model,
        "input": [{"role": "user", "content": TEST_PROMPT}],
        "tools": TEST_TOOLS,
        "tool_choice": "auto",
        "stream": False,  # No streaming
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    print("\n📤 REQUEST (non-streaming):")
    print(f"  Model: {model}")
    print(f"  Tools: {len(TEST_TOOLS)}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(endpoint, json=payload, headers=headers)

        print("\n📥 RESPONSE:")
        print(f"  Status: {response.status_code}")

        if response.status_code != 200:
            print(f"  ERROR: {response.text}")
            return

        data = response.json()
        print("\n📋 RESPONSE DATA:")
        print(json.dumps(data, indent=2)[:2000])

        # Look for tool calls
        tool_calls = []
        for item in data.get("output", []):
            if item.get("type") == "function_call":
                tool_calls.append(item)

        print("\n📊 SUMMARY:")
        print(f"  Tool calls found: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"    - {tc.get('name')}: {tc.get('arguments', {})}")


async def test_chat_completions_streaming():
    """Test Chat Completions API with streaming and tools for comparison."""
    print("\n" + "=" * 80)
    print("TEST: OpenAI Chat Completions API with Streaming + Tools")
    print("=" * 80)

    endpoint = "https://api.openai.com/v1/chat/completions"
    model = "gpt-4o-mini"

    # Chat Completions format for tools (different from Responses API!)
    chat_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in TEST_TOOLS
    ]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "tools": chat_tools,
        "tool_choice": "auto",
        "stream": True,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    print("\n📤 REQUEST:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Model: {model}")
    print(f"  Tools: {len(chat_tools)}")
    print("  Tool format: Chat Completions (nested under 'function')")

    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client,
        client.stream("POST", endpoint, json=payload, headers=headers) as response,
    ):
        print("\n📥 RESPONSE:")
        print(f"  Status: {response.status_code}")

        if response.status_code != 200:
            body = await response.aread()
            print(f"  ERROR Body: {body.decode()}")
            return

        print("\n🔄 STREAMING CHUNKS:")
        chunk_count = 0
        tool_calls_found = []
        current_tool_call = {}

        async for line in response.aiter_lines():
            if not line.strip():
                continue
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    print("\n  [DONE] received")
                    break
                try:
                    event = json.loads(data)
                    chunk_count += 1

                    delta = event.get("choices", [{}])[0].get("delta", {})

                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            func = tc.get("function")
                            if func and func.get("name"):
                                current_tool_call[idx] = {
                                    "name": func["name"],
                                    "arguments": "",
                                }
                                print(f"\n  🔧 TOOL CALL START: {func['name']}")
                            if func and func.get("arguments") and idx in current_tool_call:
                                current_tool_call[idx]["arguments"] += func["arguments"]

                    if chunk_count <= 3:
                        print(f"\n  Chunk {chunk_count}: {json.dumps(event, indent=2)[:200]}...")

                except json.JSONDecodeError as e:
                    print(f"  [JSON Error] {e}")

        tool_calls_found = list(current_tool_call.values())

        print("\n" + "=" * 80)
        print("📊 SUMMARY:")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Tool calls found: {len(tool_calls_found)}")
        for tc in tool_calls_found:
            print(f"    - {tc.get('name')}: {tc.get('arguments', '')[:100]}...")


async def main():
    print("🧪 OpenAI API Tool Test Suite")
    print("Testing whether tools work correctly with streaming\n")

    # Test 1: Responses API with streaming
    await test_responses_api_streaming()

    # Test 2: Responses API without streaming (for comparison)
    await test_responses_api_non_streaming()

    # Test 3: Chat Completions API with streaming (for comparison)
    await test_chat_completions_streaming()

    print("\n" + "=" * 80)
    print("✅ All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
