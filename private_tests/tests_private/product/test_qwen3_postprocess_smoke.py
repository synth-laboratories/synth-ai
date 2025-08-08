#!/usr/bin/env python3
"""
Qwen3 tool-calling postprocess smoke test against Synth backend.

Verifies that tool_calls are populated and assistant content is cleaned
after moving parsing/formatting into the Qwen3 family.

Env:
  - SYNTH_API_KEY (required)
  - SYNTH_BASE_URL (should include /api) or SYNTH_API_URL (we append /api)
  - QWEN_MODEL (optional; default: Qwen/Qwen3-14B-Instruct)

Run:
  uv run python synth-ai/test_qwen3_postprocess_smoke.py
"""

import os
import asyncio
from typing import Any

from openai import AsyncOpenAI


MODEL_ID = os.getenv("QWEN_MODEL", "Qwen/Qwen3-14B-Instruct")


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
                "name": "weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]


async def main() -> None:
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY must be set")
    base_url = _resolve_base_url()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": "You have access to external function tools. Use them when needed."},
        {"role": "user", "content": "What's the weather in San Francisco?"},
    ]

    print(f"\n🔧 Qwen3 postprocess smoke test\n{'='*40}")
    print(f"Model: {MODEL_ID}")
    print(f"Base URL: {base_url}")

    resp = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=_build_tools(),
        tool_choice="required",
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    msg = resp.choices[0].message
    print("tool_calls:", getattr(msg, "tool_calls", None))
    print("assistant:", (msg.content or "")[:200])

    fr = getattr(resp.choices[0], "finish_reason", None)
    usage = getattr(resp, "usage", None)
    print("finish_reason:", fr)
    if usage:
        print(
            f"usage: prompt_tokens={getattr(usage,'prompt_tokens',None)}, "
            f"completion_tokens={getattr(usage,'completion_tokens',None)}, "
            f"total_tokens={getattr(usage,'total_tokens',None)}"
        )


if __name__ == "__main__":
    asyncio.run(main())

