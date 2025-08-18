"""Procedural Crafter taskset generation with seed filtering by world traits.
Run this to build a TaskInstanceSet with reproducible initial snapshots.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass, fields
from typing import Dict, List
from uuid import UUID, uuid4

import crafter
import numpy as np
from crafter import objects

from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    SplitInfo,
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
TASK = Task(
    global_premises="Procedural Crafter seed generation",
    global_constraints="",
    global_objectives="Survive and unlock achievements.",
    shared_env_params={},
)

AREA = (64, 64)
LEN = 10000
RADIUS = 10  # Manhattan distance for local trait count
SEED_START = 0
NUM_INSTANCES = 50

# Desired trait ranges per difficulty tier
TRAIT_BOUNDS = {
    "easy": {
        "min_trees": 4,
        "max_hostiles": 0,
    },
    "medium": {
        "min_trees": 2,
        "max_hostiles": 2,
    },
    "hard": {
        "min_trees": 0,
        "max_hostiles": 5,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Metadata + instance helpers
# ──────────────────────────────────────────────────────────────────────────────


from typing import Optional


@dataclass
class CrafterTaskInstanceMetadata(TaskInstanceMetadata):
    difficulty: str
    seed: int
    num_trees_radius: int
    num_cows_radius: int
    num_hostiles_radius: int
    world_config: Optional[str] = "normal"  # 'easy', 'normal', 'hard', 'peaceful'
    world_config_path: Optional[str] = None  # Path to custom JSON config


@dataclass
class CrafterTaskInstance(TaskInstance):
    async def serialize(self) -> dict:  # identical to Sokoban pattern
        data = asdict(self)
        if isinstance(data.get("id"), UUID):
            data["id"] = str(data["id"])
        if "intent" in data and data["intent"] is not None:
            data["intent"]["deterministic_eval_functions"] = []
        return data

    @classmethod
    async def deserialize(cls, data: dict) -> "CrafterTaskInstance":
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except Exception:
                pass
        if "impetus" in data and isinstance(data["impetus"], dict):
            impetus_data = data["impetus"]
            # Ensure instructions field exists with default if missing
            if "instructions" not in impetus_data:
                impetus_data["instructions"] = "Survive and unlock achievements"
            data["impetus"] = Impetus(**impetus_data)
        if "intent" in data and isinstance(data["intent"], dict):
            intent_data = data["intent"]
            # Ensure required fields exist with defaults if missing
            if "rubric" not in intent_data:
                intent_data["rubric"] = {"goal": "Unlock achievements"}
            if "gold_trajectories" not in intent_data:
                intent_data["gold_trajectories"] = None
            if "gold_state_diff" not in intent_data:
                intent_data["gold_state_diff"] = {}
            intent_data["deterministic_eval_functions"] = []
            data["intent"] = Intent(**intent_data)
        if "metadata" in data and isinstance(data["metadata"], dict):
            metadata_data = data["metadata"]
            # Ensure required fields exist with defaults if missing
            if "difficulty" not in metadata_data:
                metadata_data["difficulty"] = "medium"
            if "seed" not in metadata_data:
                metadata_data["seed"] = 0
            if "num_trees_radius" not in metadata_data:
                metadata_data["num_trees_radius"] = 0
            if "num_cows_radius" not in metadata_data:
                metadata_data["num_cows_radius"] = 0
            if "num_hostiles_radius" not in metadata_data:
                metadata_data["num_hostiles_radius"] = 0
            data["metadata"] = CrafterTaskInstanceMetadata(**metadata_data)
        keep = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in keep})


# ──────────────────────────────────────────────────────────────────────────────
# Trait extraction util
# ──────────────────────────────────────────────────────────────────────────────


def world_traits(env: crafter.Env, radius: int = RADIUS) -> Dict[str, int]:
    player = env._player  # type: ignore[attr-defined]
    pos = np.array(player.pos)
    counts = {"trees": 0, "cows": 0, "hostiles": 0}
    for obj in env._world._objects:  # type: ignore[attr-defined]
        if obj is None or obj is player:
            continue
        if np.abs(obj.pos - pos).sum() > radius:
            continue
        if isinstance(obj, objects.Plant) and getattr(obj, "kind", "") == "tree":
            counts["trees"] += 1
        elif isinstance(obj, objects.Cow):
            counts["cows"] += 1
        elif isinstance(obj, (objects.Zombie, objects.Skeleton)):
            counts["hostiles"] += 1
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────────────


async def create_crafter_taskset(num_instances: int = NUM_INSTANCES) -> TaskInstanceSet:
    instances: List[CrafterTaskInstance] = []
    seed = SEED_START
    while len(instances) < num_instances:
        env = crafter.Env(area=AREA, length=LEN, seed=seed)
        _ = env.reset()
        traits = world_traits(env)
        # assign difficulty tier first match
        difficulty: str | None = None
        for diff, bounds in TRAIT_BOUNDS.items():
            if (
                traits["trees"] >= bounds["min_trees"]
                and traits["hostiles"] <= bounds["max_hostiles"]
            ):
                difficulty = diff
                break
        if difficulty is None:
            seed += 1
            continue
        # build instance
        impetus = Impetus(instructions=f"Survive and unlock achievements. Difficulty={difficulty}.")
        intent = Intent(
            rubric={"goal": "Unlock as many achievements as possible."},
            gold_trajectories=None,
            gold_state_diff={},
        )
        metadata = CrafterTaskInstanceMetadata(
            difficulty=difficulty,
            seed=seed,
            num_trees_radius=traits["trees"],
            num_cows_radius=traits["cows"],
            num_hostiles_radius=traits["hostiles"],
        )
        instance = CrafterTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,  # will be filled lazily when env starts
        )
        instances.append(instance)
        seed += 1

    # simple random split 80/10/10
    random.shuffle(instances)
    n = len(instances)
    val_ids = {inst.id for inst in instances[int(0.8 * n) : int(0.9 * n)]}
    test_ids = {inst.id for inst in instances[int(0.9 * n) :]}
    split = SplitInfo(val_instance_ids=val_ids, test_instance_ids=test_ids, _is_split_defined=True)

    return TaskInstanceSet(
        name="Crafter Procedural TaskSet",
        description="Crafter seeds filtered by local world traits around spawn.",
        instances=instances,
        split_info=split,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI example
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import pathlib

    async def _main():
        ts = await create_crafter_taskset(30)
        serial = await asyncio.gather(*(inst.serialize() for inst in ts.instances))
        out = pathlib.Path("dataset/crafter_instances.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(serial, indent=2))
        print(f"Saved {len(serial)} instances → {out}")

    asyncio.run(_main())
