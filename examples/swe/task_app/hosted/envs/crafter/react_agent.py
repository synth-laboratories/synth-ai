"""Crafter ReAct agent: system prompt and message assembly.

This agent encapsulates the Crafter-specific system prompt and helpers to
construct OpenAI-style message lists. Response parsing delegates to shared
utilities to keep a single parser.
"""

from __future__ import annotations

from typing import Any

from .shared import parse_actions


class CrafterReActAgent:
    """Lightweight ReAct-style agent surface for Crafter prompts."""

    @staticmethod
    def get_system_prompt() -> str:
        return (
            "You are playing Crafter, a survival game by Danijar Hafner. Your goal is to collect resources, "
            "craft tools, survive, and unlock achievements.\n\n"
            "Core rules:\n"
            "- The world contains trees (wood), stone, coal, iron, plants, cows, zombies, and water.\n"
            "- Movement constraints: you cannot walk onto blocking tiles: tree, stone, water, lava, coal, iron. Navigate around obstacles.\n"
            "- You start with empty hands and low health/hunger.\n"
            "- Interact ('do') only when adjacent to a resource (tree, stone, cow, zombie, etc.).\n"
            "- Movement is essential: you can and should move multiple steps in one turn to explore effectively.\n"
            "- Achievements are unlocked by collecting resources, crafting tools, placing objects, fighting, and surviving longer.\n\n"
            "Key strategies:\n"
            "1. Begin by moving around to find trees. Use 'do' to collect wood when adjacent.\n"
            "2. Craft a wood pickaxe as soon as you have enough wood ('make_wood_pickaxe').\n"
            "3. Use the pickaxe to gather stone, then craft a stone pickaxe. Progress to iron tools as you find iron.\n"
            "4. Build a table ('place_table') to unlock more crafting options (furnace, sword, etc.).\n"
            "5. Manage hunger by collecting and eating plants or interacting with cows.\n"
            "6. Fight zombies with a sword for achievements and resources.\n"
            "7. Survive by balancing exploration, combat, and resource gathering.\n\n"
            "8. Keep moving to discover new resources and stay alive. If you're in the middle of nowhere, take 5-8 consecutive move-related actions to explore and see what's outside your field of view. Don't delay exploration when it's the right move.\n\n"
            "Achievements to aim for:\n"
            "- Collecting resources (wood, stone, coal, iron, plants).\n"
            "- Crafting tools (wood/stone/iron pickaxe, wood/stone/iron sword).\n"
            "- Placing structures (table, furnace, plant).\n"
            "- Combat (killing a cow or zombie).\n"
            "- Survival milestones (staying alive over time).\n\n"
            "Action policy:\n"
            "- Always return a single tool call: interact_many({actions: [...]})\n"
            "- Use 2–5 actions per call; prefer long movement sequences to explore.\n"
            "- Mix in 'do' only when it makes sense (tree, stone, animal, enemy nearby).\n"
            "- Do not spam the same exact sequence twice in a row—explore in varied directions.\n\n"
            "Available actions: noop, move_up, move_down, move_left, move_right, do (interact), sleep, "
            "place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, "
            "make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword\n"
        )

    @staticmethod
    def get_system_prompt_with_tools() -> str:
        """System prompt for tool-based interaction (e.g., Qwen3 models)."""
        return (
            "You are playing Crafter, a survival game by Danijar Hafner. Your goal is to collect resources, "
            "craft tools, survive, and unlock achievements.\n\n"
            "Rules & world:\n"
            "- Explore by chaining multiple movement actions in one turn.\n"
            "- You cannot walk onto blocking tiles: tree, stone, water, lava, coal, iron. Plan routes around obstacles.\n"
            "- Use 'do' intentionally when standing next to resources (trees, stone, cows, zombies, etc.).\n"
            "- Achievements come from collecting, crafting, building, fighting, and surviving.\n\n"
            "Strategy path:\n"
            "1. Move around to find trees → 'do' to collect wood.\n"
            "2. Craft a wood pickaxe.\n"
            "3. Gather stone → craft stone pickaxe.\n"
            "4. Place a table → unlock furnace and swords.\n"
            "5. Fight enemies (cow/zombie) with swords for achievements.\n"
            "6. Keep moving to discover new resources and stay alive. If you're in the middle of nowhere, take 5-8 consecutive move-related actions to explore and see what's outside your field of view. Don't delay exploration when it's the right move.\n\n"
            "You must use the 'interact_many' tool to perform actions in the game. "
            "This tool accepts an array of 1–5 actions to execute sequentially. Prefer sequences like "
            "[move_up, move_up, move_left, do] instead of single steps.\n\n"
            "Available actions: noop, move_up, move_down, move_left, move_right, do (interact), sleep, "
            "place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, "
            "make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword\n\n"
            "Always call the interact_many tool with your chosen actions. Do not write plain text actions.\n"
        )

    @staticmethod
    def build_messages(
        observation: str,
        history: list[dict[str, Any]] | None = None,
        turn: int | None = None,
        image_parts: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Construct OpenAI-style messages list for vLLM generation."""
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": CrafterReActAgent.get_system_prompt()}
        ]
        if history:
            msgs.extend(history)
        user_content: Any
        if image_parts:
            user_content = [{"type": "text", "text": observation}] + list(image_parts)
        else:
            user_content = observation
        msgs.append({"role": "user", "content": user_content})
        return msgs

    @staticmethod
    def parse_actions_from_response(response_text: str) -> list[str]:
        return parse_actions(response_text)


__all__ = ["CrafterReActAgent"]
