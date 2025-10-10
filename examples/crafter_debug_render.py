#!/usr/bin/env python3
"""
Quick local Crafter observation inspector.

- Initializes a CrafterClassic env via local service (default http://localhost:8901)
- Fetches one observation
- Renders a 7x7 semantic view around the player with best-effort item names
- Prints status (health/food/energy), inventory, and achievements

Run:
  uv run python examples/crafter_debug_render.py --base-url http://localhost:8901 --seed 1
"""

import argparse
import contextlib
import math
import os
from typing import Any

import httpx


def try_import_crafter_mapping():
    try:
        import crafter  # type: ignore

        env = crafter.Env()
        try:
            max_id = (
                max(max(env._world._mat_ids.values()), max(env._sem_view._obj_ids.values())) + 1
            )
            id_to_item = ["void"] * max_id
            for name, ind in env._world._mat_ids.items():
                label = name.__name__ if hasattr(name, "__name__") else str(name)
                id_to_item[ind] = label.lower()
            for name, ind in env._sem_view._obj_ids.items():
                label = name.__name__ if hasattr(name, "__name__") else str(name)
                id_to_item[ind] = label.lower()
            return id_to_item
        finally:
            with contextlib.suppress(Exception):
                env.close()
    except Exception:
        return None


def format_semantic_map_view(obs: dict[str, Any], view_size: int = 7) -> str:
    sem = obs.get("semantic_map") or obs.get("sem_map") or obs.get("map")
    if sem is None:
        return "No semantic map available"

    # Normalize to 2D grid
    grid: list[list[int]]
    if isinstance(sem, list) and sem and isinstance(sem[0], list):
        grid = sem
    elif isinstance(sem, list):
        try:
            n = int(math.sqrt(len(sem)))
            if n * n != len(sem) or n == 0:
                return "Semantic map format not recognized"
            grid = [sem[i * n : (i + 1) * n] for i in range(n)]
        except Exception:
            return "Semantic map format not recognized"
    else:
        return "Semantic map format not recognized"

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return "Empty semantic map"

    # Resolve item mapping if available
    id_to_item = try_import_crafter_mapping()

    # Player position if provided; otherwise center
    ppos = obs.get("player_position") or [rows // 2, cols // 2]
    try:
        px = int(ppos[0])
        py = int(ppos[1])
    except Exception:
        px, py = rows // 2, cols // 2

    half = max(1, view_size // 2)
    lines: list[str] = []
    visible: set[str] = set()
    for dy in range(-half, half + 1):
        row_cells: list[str] = []
        for dx in range(-half, half + 1):
            x = px + dx
            y = py + dy
            if dx == 0 and dy == 0:
                row_cells.append("you")
            elif 0 <= x < rows and 0 <= y < cols:
                try:
                    val = int(grid[x][y])
                except Exception:
                    val = -1
                if id_to_item and 0 <= val < len(id_to_item):
                    name = id_to_item[val]
                else:
                    # Fallback: simple mapping for common ids (best-effort)
                    name = {
                        0: "grass",
                        1: "stone",
                        2: "stone",
                        3: "tree",
                        4: "coal",
                        5: "iron",
                        6: "water",
                        7: "zombie",
                        14: "wood",
                    }.get(val, str(val))
                row_cells.append(name)
                if name not in {"grass", "you", "0"}:
                    visible.add(name)
            else:
                row_cells.append("void")
        lines.append(" ".join(row_cells))

    legend = (
        f"Visible items: {', '.join(sorted(visible))}" if visible else "No notable items visible"
    )
    return "\n".join(lines) + "\n" + legend


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default=os.getenv("CRAFTER_BASE_URL", "http://localhost:8901")
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=30.0) as client:
        init = await client.post(
            f"{args.base_url}/env/CrafterClassic/initialize",
            json={"config": {"difficulty": "easy", "seed": args.seed}},
        )
        init.raise_for_status()
        data = init.json()
        env_id = data["env_id"]
        obs = data["observation"]

        print("=== INITIAL OBSERVATION ===")
        print(format_semantic_map_view(obs, view_size=7))
        inv = obs.get("inventory", {})
        ach = obs.get("achievements_status", {})
        print("\n=== STATUS ===")
        print(f"Health: {obs.get('health', 10)}/10")
        print(f"Hunger: {obs.get('food', 10)}/10")
        print(f"Energy: {obs.get('energy', 10)}/10")
        inv_items = (
            ", ".join([f"{k}: {v}" for k, v in inv.items() if v])
            if isinstance(inv, dict)
            else str(inv)
        )
        print(f"Inventory: {inv_items if inv_items else 'empty'}")
        if isinstance(ach, dict):
            unlocked = sum(1 for v in ach.values() if v)
            print(f"Achievements: {unlocked}/{len(ach)} unlocked")

        # Take one step right to get a new obs
        step = await client.post(
            f"{args.base_url}/env/CrafterClassic/step",
            json={
                "env_id": env_id,
                "action": {"tool_calls": [{"tool": "interact", "args": {"action": 2}}]},
            },
        )
        step.raise_for_status()
        sdata = step.json()
        sobs = sdata["observation"]
        print("\n=== NEXT OBSERVATION (after move_right) ===")
        print(format_semantic_map_view(sobs, view_size=7))

        # Cleanup
        with contextlib.suppress(Exception):
            await client.post(
                f"{args.base_url}/env/CrafterClassic/terminate", json={"env_id": env_id}
            )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
