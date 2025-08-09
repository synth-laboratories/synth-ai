#!/usr/bin/env python3
"""
Minimal Qwen3 smoke test against Synth backend.

Requirements:
  - Environment:
      SYNTH_API_KEY
      SYNTH_BASE_URL (should include /api) or SYNTH_API_URL (script appends /api)
      QWEN_MODEL (optional; default: Qwen/Qwen3-8B)

Run:
  uv run python synth-ai/tests/qwen3_smoketest.py
"""

import asyncio
import os
from typing import Any

try:
    from openai import AsyncOpenAI
except Exception:
    raise SystemExit("openai package is required. Install with: pip install openai>=1.0.0")


MODEL_ID = os.getenv("QWEN_MODEL", "Qwen/Qwen3-14B")


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


def _weather_tools() -> list[dict[str, Any]]:
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
        }
    ]


def _weather_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Use tools when helpful; return function calls via OpenAI tool-calling."},
        {"role": "user", "content": "What's the weather in Berlin in Celsius?"},
    ]


async def main() -> None:
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY must be set")
    base_url = _resolve_base_url()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print("\n🔧 Qwen3 smoke test (Synth)")
    print("=" * 40)
    print(f"Model: {MODEL_ID}")
    print(f"Base URL: {base_url}")

    # Build request
    tools = _weather_tools()
    messages = _weather_messages()

    # Hard switch off thinking via chat_template_kwargs (Qwen3 only)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    print("\n🧪 tool_choice=required, enable_thinking=False")
    resp = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools,
        tool_choice="required",
        max_tokens=512,
        extra_body=extra_body,
        temperature=0.6,
        top_p=0.95,
    )
    choice = resp.choices[0]
    msg = choice.message
    tc = getattr(msg, "tool_calls", None)
    if tc:
        print(f"✅ tool_calls produced: {len(tc)}")
        for c in tc:
            print(f"   • {c.function.name}: {c.function.arguments}")
    else:
        print("⚠️ No tool_calls returned")
        print(f"Assistant content (truncated): {str(msg.content)[:200] if msg.content else 'None'}")
    usage = getattr(resp, "usage", None)
    if usage:
        print(
            f"usage: prompt_tokens={getattr(usage,'prompt_tokens',None)}, "
            f"completion_tokens={getattr(usage,'completion_tokens',None)}, "
            f"total_tokens={getattr(usage,'total_tokens',None)}"
        )


if __name__ == "__main__":
    asyncio.run(main())

