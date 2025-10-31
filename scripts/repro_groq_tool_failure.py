#!/usr/bin/env python3
"""
Reproduce the Groq tool_call failure we are seeing in Crafter rollouts.

Usage:
    python scripts/repro_groq_tool_failure.py

Requires GROQ_API_KEY in the environment.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    sys.stderr.write("GROQ_API_KEY not set in environment\n")
    sys.exit(1)


SYSTEM_PROMPT = """You are playing Crafter, a survival game by Danijar Hafner. Your goal is to collect resources, craft tools, survive, and unlock achievements.

Core rules:
- The world contains trees (wood), stone, coal, iron, plants, cows, zombies, and water.
- Movement constraints: you cannot walk onto blocking tiles: tree, stone, water, lava, coal, iron. Navigate around obstacles.
- You start with empty hands and low health/hunger.
- Interact ('do') only when adjacent to a resource (tree, stone, cow, zombie, etc.).
- Movement is essential: you can and should move multiple steps in one turn to explore effectively.
- Achievements are unlocked by collecting resources, crafting tools, placing objects, fighting, and surviving longer.

Key strategies:
1. Begin by moving around to find trees. Use 'do' to collect wood when adjacent.
2. Craft a wood pickaxe as soon as you have enough wood ('make_wood_pickaxe').
3. Use the pickaxe to gather stone, then craft a stone pickaxe. Progress to iron tools as you find iron.
4. Build a table ('place_table') to unlock more crafting options (furnace, sword, etc.).
5. Manage hunger by collecting and eating plants or interacting with cows.
6. Fight zombies with a sword for achievements and resources.
7. Survive by balancing exploration, combat, and resource gathering.

8. Keep moving to discover new resources and stay alive. If you're in the middle of nowhere, take 5-8 consecutive move-related actions to explore and see what's outside your field of view. Don't delay exploration when it's the right move.

Achievements to aim for:
- Collecting resources (wood, stone, coal, iron, plants).
- Crafting tools (wood/stone/iron pickaxe, wood/stone/iron sword).
- Placing structures (table, furnace, plant).
- Combat (killing a cow or zombie).
- Survival milestones (staying alive over time).

Action policy:
- Always return a single tool call: interact_many({actions: [...]})
- Use 2–5 actions per call; prefer long movement sequences to explore.
- Mix in 'do' only when it makes sense (tree, stone, animal, enemy nearby).

Available actions: noop, move_up, move_down, move_left, move_right, do (interact), sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword
"""

USER_PROMPT = """=== CRAFTER GAME STATE ===
Step: 4/10000
Health: 9
Position: [36, 32]
Facing: [1, 0]
Inventory: food:9, drink:9, energy:9
Achievements: none

Local Map View (5x5):
grass grass grass grass grass
grass grass grass grass grass
grass cow player grass grass
grass grass grass grass grass
grass grass grass tree tree

Choose your next actions.


Previous tool calls (most recent first):
- interact_many: {"actions":["move_right","move_right","move_right","move_right"]}
"""

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "interact_many",
            "description": "Execute a short sequence of Crafter actions in order (1-8).",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "List of Crafter actions to execute sequentially.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "noop",
                                "move_left",
                                "move_right",
                                "move_up",
                                "move_down",
                                "do",
                                "sleep",
                                "place_stone",
                                "place_table",
                                "place_furnace",
                                "place_plant",
                                "make_wood_pickaxe",
                                "make_stone_pickaxe",
                                "make_iron_pickaxe",
                                "make_wood_sword",
                                "make_stone_sword",
                                "make_iron_sword",
                            ],
                        },
                        "minItems": 1,
                        "maxItems": 8,
                    }
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
        },
    }
]

payload: dict[str, Any] = {
    "model": "qwen/qwen3-32b",
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 8192,
    "tools": TOOLS_SCHEMA,
    "tool_choice": "required",
    "function_call": {"name": "interact_many"},
    "parallel_tool_calls": False,
    "stop_after_tool_calls": 1,
}

print("Request payload:\n", json.dumps(payload, indent=2)[:2000], "\n", file=sys.stderr)

with httpx.Client(timeout=60.0) as client:
    response = client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
    )

print("Status:", response.status_code)
print("Raw body:", response.text)
