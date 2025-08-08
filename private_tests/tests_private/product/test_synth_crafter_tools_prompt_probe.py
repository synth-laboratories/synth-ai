#!/usr/bin/env python3
"""
Probe tool-calling with Crafter agent's tools and prompt against Synth GPT-OSS-20B.
Runs two calls: tool_choice=auto and tool_choice=required, printing tool_calls if any.

Env:
  SYNTH_API_KEY
  SYNTH_BASE_URL (should include /api) or SYNTH_API_URL (we will append /api)

Run:
  uv run python synth-ai/test_synth_crafter_tools_prompt_probe.py
"""

import asyncio
import os
from typing import Any

from openai import AsyncOpenAI

MODEL_ID = os.getenv("CRAFTER_MODEL", os.getenv("GPT_OSS_MODEL", "openai/gpt-oss-20b"))


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
                        "reasoning": {
                            "type": "string",
                            "description": "Reasoning for these actions",
                        },
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
                    "properties": {
                        "reason": {"type": "string", "description": "Reason for termination"}
                    },
                    "required": ["reason"],
                },
            },
        },
    ]


def crafter_system_prompt() -> str:
    # Minimal but representative variant of the Crafter agent system message
    return (
        "You are CrafterAgent playing Crafter survival environment. Your goal is to unlock as many "
        "achievements as possible while staying alive.\n\n"
        "Available actions: move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, "
        "place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, "
        "make_wood_sword, make_stone_sword, make_iron_sword, noop\n\n"
        "CRITICAL INSTRUCTION: You MUST ALWAYS provide MULTIPLE actions (2-5) in EVERY interact() tool call!\n"
        "Always respond with a single function tool call in the OpenAI tools format (no natural-language only replies)."
    )


def crafter_user_prompt() -> str:
    # Matches the map/status shown in logs
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
        "REMINDER: You MUST provide 2-5 actions in your interact() tool call. Plan multiple steps ahead!\n"
        "Example: interact(actions=[\"move_right\", \"move_right\", \"do\"], reasoning=\"Move to tree and collect wood\")"
    )


async def probe(tool_choice: str) -> None:
    api_key = os.getenv("SYNTH_API_KEY")
    if not api_key:
        raise RuntimeError("SYNTH_API_KEY must be set")
    base_url = _resolve_base_url()

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    tools = crafter_tools()
    messages = [
        {"role": "system", "content": crafter_system_prompt()},
        {"role": "user", "content": crafter_user_prompt()},
    ]

    print(f"\n🧪 Crafter probe: tool_choice={tool_choice}")
    resp = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=2000,
    )
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        print(f"✅ tool_calls produced: {len(tool_calls)}")
        for tc in tool_calls:
            print(f"   • {tc.function.name}: {tc.function.arguments}")
    else:
        print("⚠️ No tool_calls returned")
        print(f"Assistant content (truncated): {str(msg.content)[:200]}")
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
    await probe("auto")
    await probe("required")


if __name__ == "__main__":
    asyncio.run(main())