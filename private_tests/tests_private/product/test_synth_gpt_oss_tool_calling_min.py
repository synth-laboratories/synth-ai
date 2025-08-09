#!/usr/bin/env python3
"""
Minimal tool-calling test against Synth's GPT-OSS-20B endpoint.
Mirrors monorepo's test: passes tools, tool_choice="auto", and a simple prompt.

Env required:
- SYNTH_API_KEY
- One of: SYNTH_BASE_URL (should include /api) or SYNTH_API_URL (we will append /api)

Run:
  uv run python synth-ai/test_synth_gpt_oss_tool_calling_min.py
"""

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI

MODEL_ID = os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-20b")


def _resolve_base_url() -> str:
    base = os.getenv("SYNTH_BASE_URL")
    if not base:
        api = os.getenv("SYNTH_API_URL", "")
        if not api:
            raise RuntimeError("SYNTH_BASE_URL or SYNTH_API_URL must be set")
        base = api.rstrip("/")
        if not base.endswith("/api"):
            base = base + "/api"
    return base


def _build_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"],
                },
            },
        },
    ]


async def main() -> None:
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY must be set")

    base_url = _resolve_base_url()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    tools = _build_tools()
    messages = [
        {"role": "system", "content": "You have access to external function tools. Use them to complete the task."},
        {"role": "user", "content": "What's the weather in Berlin in Celsius?"},
    ]

    print(f"\n🔧 Testing Tool Calling (Synth)\n{'='*50}")
    print(f"Model: {MODEL_ID}")
    print(f"Base URL: {base_url}")

    resp = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=2000,
    )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    if tool_calls:
        print(f"✅ Tool calling successful: {len(tool_calls)} call(s)")
        for tc in tool_calls:
            # tc.function.name, tc.function.arguments
            print(f"   • {tc.function.name}: {tc.function.arguments}")
    else:
        print("⚠️ Model did not invoke any tools")
        print(f"Assistant content: {msg.content}")

    # Print finish_reason and usage when available
    try:
        fr = getattr(resp.choices[0], "finish_reason", None)
        usage = getattr(resp, "usage", None)
        if fr is not None:
            print(f"finish_reason: {fr}")
        if usage is not None:
            print(
                f"usage: prompt_tokens={getattr(usage,'prompt_tokens',None)}, "
                f"completion_tokens={getattr(usage,'completion_tokens',None)}, "
                f"total_tokens={getattr(usage,'total_tokens',None)}"
            )
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())