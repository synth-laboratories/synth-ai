#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx


def to_grid(obj: Any) -> List[List[int]]:
    if isinstance(obj, list) and obj and isinstance(obj[0], list):
        return obj
    if isinstance(obj, list):
        # attempt square reshape
        import math

        n = int(math.sqrt(len(obj))) if obj else 0
        if n * n == len(obj) and n > 0:
            return [obj[i * n : (i + 1) * n] for i in range(n)]
    return []


def color_for(val: int) -> Tuple[int, int, int]:
    palette = {
        0: (64, 160, 64),  # grass
        1: (120, 120, 120),  # stone
        2: (120, 120, 120),  # stone
        3: (34, 139, 34),  # tree
        4: (40, 40, 40),  # coal
        5: (180, 180, 180),  # iron
        6: (52, 152, 219),  # water
        7: (200, 50, 50),  # hostile
        14: (205, 133, 63),  # wood
    }
    return palette.get(int(val), (180, 180, 180))


def save_semantic(grid: List[List[int]], out_path: Path, scale: int = 8) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    if not grid:
        return
    rows = len(grid)
    cols = len(grid[0])
    img = Image.new("RGB", (cols, rows))
    px = img.load()
    for y in range(rows):
        row = grid[y]
        for x in range(cols):
            px[x, y] = color_for(row[x])
    if scale > 1:
        img = img.resize((cols * scale, rows * scale), Image.NEAREST)
    img.save(out_path)


def save_rgb(rgb: Any, out_path: Path) -> None:
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return
    try:
        arr = None
        if isinstance(rgb, list):
            arr = np.array(rgb, dtype="uint8")
        elif isinstance(rgb, dict) and "data" in rgb and "shape" in rgb:
            # optional binary-like encoding not used here
            arr = np.array(rgb["data"], dtype="uint8").reshape(rgb["shape"])  # type: ignore
        if arr is None:
            return
        img = Image.fromarray(arr, mode="RGB")
        img.save(out_path)
    except Exception:
        return


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("CRAFTER_BASE_URL", "http://127.0.0.1:8901"))
    parser.add_argument(
        "--out-dir",
        default=os.getenv(
            "CRAFTER_DEBUG_OUT",
            "/Users/joshpurtell/Documents/GitHub/monorepo/backend/app/traces/crafter_debug",
        ),
    )
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sess_dir = out_root / stamp
    sess_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{args.base_url}/env/CrafterClassic/initialize",
            json={"config": {"difficulty": "easy", "seed": 1}},
        )
        r.raise_for_status()
        payload = r.json()
        env_id = payload["env_id"]
        obs = payload["observation"]

        # Save initial observation images
        sem_full = to_grid(obs.get("semantic_map")) or to_grid(obs.get("sem_map"))
        sem_patch = to_grid(obs.get("semantic_map_patch7"))
        save_semantic(sem_full, sess_dir / "init_sem_full.png")
        save_semantic(sem_patch, sess_dir / "init_sem_patch7.png")
        save_rgb(obs.get("observation_image"), sess_dir / "init_rgb.png")

        # Step a few times and save
        for i in range(args.steps):
            step = await client.post(
                f"{args.base_url}/env/CrafterClassic/step",
                json={"env_id": env_id, "action": {"tool_calls": [{"tool": "interact", "args": {"action": 2}}]}},
            )
            step.raise_for_status()
            sobs = step.json().get("observation", {})
            sem_full = to_grid(sobs.get("semantic_map")) or to_grid(sobs.get("sem_map"))
            sem_patch = to_grid(sobs.get("semantic_map_patch7"))
            save_semantic(sem_full, sess_dir / f"step_{i+1:02d}_sem_full.png")
            save_semantic(sem_patch, sess_dir / f"step_{i+1:02d}_sem_patch7.png")
            save_rgb(sobs.get("observation_image"), sess_dir / f"step_{i+1:02d}_rgb.png")

        # Cleanup
        try:
            await client.post(f"{args.base_url}/env/CrafterClassic/terminate", json={"env_id": env_id})
        except Exception:
            pass

    print(f"Wrote images under: {sess_dir}")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))


