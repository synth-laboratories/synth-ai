from __future__ import annotations

from typing import Any


ACTION_NAMES = [
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
    "make_wood_axe",
    "make_stone_axe",
    "make_iron_axe",
    "make_wood_wall",
    "make_stone_wall",
    "make_iron_wall",
    "make_wood_door",
    "make_stone_door",
    "make_iron_door",
]
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}


class Policy:
    def __init__(self) -> None:
        self._move_cycle = [
            ACTION_TO_ID["move_right"],
            ACTION_TO_ID["move_down"],
            ACTION_TO_ID["move_left"],
            ACTION_TO_ID["move_up"],
        ]

    def act(self, observation: dict[str, Any], info: dict[str, Any] | None = None) -> int:
        inventory = observation.get("inventory")
        if not isinstance(inventory, dict):
            inventory = {}
        achievements = observation.get("achievements_status")
        if not isinstance(achievements, dict):
            achievements = {}
        step = int(observation.get("step_count") or 0)
        health = float(observation.get("health") or inventory.get("health") or 0)

        if health and health < 3 and step % 4 == 0:
            return ACTION_TO_ID["sleep"]
        if int(inventory.get("wood") or 0) >= 1 and not achievements.get("place_table"):
            return ACTION_TO_ID["place_table"]
        if int(inventory.get("wood") or 0) >= 1 and not achievements.get("make_wood_pickaxe"):
            return ACTION_TO_ID["make_wood_pickaxe"]
        if step % 5 in {0, 1}:
            return ACTION_TO_ID["do"]
        return self._move_cycle[step % len(self._move_cycle)]
