#!/usr/bin/env python3
"""A minimal **tool-calling** smoke-test for a GPT-OSS fine-tuned model.

The script sends a single prompt that should trigger a tool call ("What's the
weather in Berlin in Celsius?") and prints the assistant's response verbatim so
you can manually verify the JSON tool-call payload.

Environment variables expected:
  • GPT_OSS_FINE_TUNED_MODEL – the `ft:` identifier of your model.
  • SYNTH_API_KEY / SYNTH_API_URL  – same as usual.
"""

import asyncio
import json
import os
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTH_API_URL = os.getenv("SYNTH_API_URL", "http://localhost:8000")
SYNTH_API_KEY = os.getenv("SYNTH_API_KEY", "")
MODEL_ID = os.getenv("GPT_OSS_FINE_TUNED_MODEL", "")

if not MODEL_ID:
    raise RuntimeError("GPT_OSS_FINE_TUNED_MODEL env var is required")

HEADERS = {
    "Authorization": f"Bearer {SYNTH_API_KEY}",
    "Content-Type": "application/json",
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
]

PROMPT = "What's the weather in Berlin in Celsius?"

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n" + "=" * 60)
    print("GPT-OSS TOOL-CALLING TEST")
    print("=" * 60)
    print(f"Model ID: {MODEL_ID}")

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You have access to external function tools. "
                    "Invoke them when useful."
                ),
            },
            {"role": "user", "content": PROMPT},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.4,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            f"{SYNTH_API_URL}/api/v1/chat/completions", json=payload, headers=HEADERS
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Request failed: {resp.status_code} – {resp.text[:200]}")

    data = resp.json()
    assistant_message = data["choices"][0]["message"]  # type: ignore[index]

    print("\nRaw assistant message: ")
    print(json.dumps(assistant_message, indent=2))

    # If a tool call is present, show just the call for convenience.
    if "tool_calls" in assistant_message:
        print("\nExtracted tool call(s):")
        for call in assistant_message["tool_calls"]:
            print(json.dumps(call, indent=2))

    print("\n🎉 Tool-calling test completed")


if __name__ == "__main__":
    asyncio.run(main())
