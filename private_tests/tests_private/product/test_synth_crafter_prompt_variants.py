#!/usr/bin/env python3
"""
Prompt-variant probe for Crafter tool-calling on Synth backend.

- Iterates system prompt variants from lenient to very strict
- Tests both tool_choice=auto and tool_choice=required
- Prints whether tool_calls are produced, along with finish_reason and usage

Env:
  - SYNTH_API_KEY
  - SYNTH_BASE_URL (should include /api) or SYNTH_API_URL (we will append /api)
  - CRAFTER_MODEL (optional; default: openai/gpt-oss-20b)

Run:
  uv run python synth-ai/test_synth_crafter_prompt_variants.py
"""

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI

MODEL_ID = os.getenv("CRAFTER_MODEL", "openai/gpt-oss-20b")


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


def crafter_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "interact",
                "description": "Perform actions in the Crafter environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of actions to perform in sequence (e.g., ['move_right', 'move_right', 'do']). Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword, noop",
                        },
                        "reasoning": {"type": "string", "description": "Reasoning for these actions"},
                    },
                    "required": ["actions", "reasoning"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminate",
                "description": "End the episode when finished or no progress can be made.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string", "description": "Reason for termination"}},
                    "required": ["reason"],
                },
            },
        },
    ]


def crafter_user_prompt() -> str:
    return (
        "Episode 0 - Turn 1/30\n\n"
        "=== SEMANTIC MAP VIEW (7x7) ===\n"
        "grass grass grass grass cow grass grass\n"
        "grass grass grass grass grass grass grass\n"
        "grass grass grass grass grass grass grass\n"
        "grass grass grass you grass grass grass\n"
        "grass grass grass grass grass grass grass\n"
        "grass grass grass grass grass grass grass\n"
        "grass grass grass grass grass grass grass\n"
        "Visible items: cow\n\n"
        "=== STATUS ===\n"
        "Health: 10/10 | Food: 10/10 | Drink: 10/10 | Energy: 10/10\n"
        "Inventory: health: 9, food: 9, drink: 9, energy: 9\n"
        "Achievements: 0/22 unlocked\n"
        "Unlocked: none\n\n"
        "What do you see in the map? What actions should you take?\n\n"
        "REMINDER: You MUST provide 1-5 actions in your interact() tool call. Plan multiple steps ahead!\n"
    )


def system_variants() -> list[str]:
    base_actions = (
        "Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, "
        "place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, "
        "make_wood_sword, make_stone_sword, make_iron_sword, noop."
    )
    return [
        # 0) Baseline concise
        # (
        #     "You are CrafterAgent playing Crafter survival environment. "
        #     f"{base_actions} "
        #     "CRITICAL: Always respond with a single function tool call. Do not output natural language."
        # ),
        # # 1) Strict with fallback to terminate
        # (
        #     "You are CrafterAgent. "
        #     f"{base_actions} "
        #     "Respond with exactly one function tool call. If you cannot act, call terminate(reason=...). "
        #     "Never include any natural language."
        # ),
        # 2) Force format and minimal reasoning
        (
            "You are CrafterAgent. "
            f"{base_actions} "
            #"Please use the tools provided, focusing on interacting with the game environment"
            #""  # reasoning must be a short phrase (<=10 words). reasoning must be a short phrase (<=10 words).
            # "No other text."
        ),
        # 3) Over-explicit instruction to avoid prose
        # (
        #     "You are CrafterAgent. "
        #     f"{base_actions} "
        #     "Do not write analysis. Do not explain. Do not prefix. Only emit one function tool call."
        # ),
        # # 4) Minimalistic hard constraint
        # (
        #     "Only output a single function tool call (interact or terminate). "
        #     "Never output any natural-language content."
        # ),
    ]


async def _call(client: AsyncOpenAI, sys: str, tool_choice: str) -> None:
    tools = crafter_tools()
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": crafter_user_prompt()},
    ]
    resp = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=2048,
        temperature=1,
    )
    print(resp.choices[0])
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        print(f"✅ tool_calls produced: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"   • {tc.function.name}: {tc.function.arguments}")
    else:
        print("⚠️ No tool_calls returned")
        print(f"Assistant content (truncated): {str(msg.content)}")
    fr = getattr(resp.choices[0], "finish_reason", None)
    usage = getattr(resp, "usage", None)
    print(f"finish_reason: {fr}")
    if usage:
        print(
            f"usage: prompt_tokens={getattr(usage,'prompt_tokens',None)}, "
            f"completion_tokens={getattr(usage,'completion_tokens',None)}, "
            f"total_tokens={getattr(usage,'total_tokens',None)}"
        )


async def main() -> None:
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY must be set")
    base_url = _resolve_base_url()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"\n🔧 Crafter Prompt Variants (Synth)\n{'='*50}")
    print(f"Model: {MODEL_ID}")
    print(f"Base URL: {base_url}")

    for idx, sys in enumerate(system_variants()):
        #print(f"\n—— Variant {idx} (auto) ——")
        #await _call(client, sys, "auto")
        print(f"\n—— Variant {idx} (required) ——")
        await _call(client, sys, "required")


if __name__ == "__main__":
    asyncio.run(main())