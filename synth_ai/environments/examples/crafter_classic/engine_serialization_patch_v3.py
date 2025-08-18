"""
Enhanced monkey patch to add save/load functionality to crafter.Env
This version handles player references for Zombie and Skeleton objects.
"""

import collections
import pickle
from typing import Any, Dict, Optional, Set

import crafter
import numpy as np
from crafter import objects

print("[PATCH] Attempting to apply Crafter serialization patch v3...")

# Check if already patched
if not hasattr(crafter.Env, "save"):
    print("[PATCH] Adding enhanced save/load methods to crafter.Env...")

    def _save(self) -> Dict[str, Any]:
        """Save complete environment state including all details."""
        # Save complete world state
        world_state = {
            "area": self._world.area,
            "daylight": self._world.daylight,
            "_mat_map": self._world._mat_map.tolist(),
            "_obj_map": self._world._obj_map.tolist(),
            "_mat_names": dict(self._world._mat_names),
            "_mat_ids": dict(self._world._mat_ids),
            "_chunk_size": self._world._chunk_size,
            "random_state": pickle.dumps(self._world.random.get_state()),
        }

        # Save all objects with complete state
        objects_data = []
        player_idx = None

        for i, obj in enumerate(self._world._objects):
            if obj is None:
                objects_data.append(None)
            else:
                # Get all object attributes
                obj_data = {
                    "type": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                    "_index": i,
                    "pos": obj.pos.tolist() if hasattr(obj, "pos") else None,
                    "removed": getattr(obj, "removed", False),
                }

                # Save ALL attributes based on object type
                if isinstance(obj, objects.Player):
                    player_idx = i
                    obj_data.update(
                        {
                            "facing": list(obj.facing) if hasattr(obj, "facing") else None,
                            "health": obj.health,
                            "inventory": dict(obj.inventory),
                            "achievements": dict(obj.achievements),
                            "_hunger": obj._hunger,
                            "_thirst": obj._thirst,
                            "_fatigue": obj._fatigue,
                            "_recover": obj._recover,
                            "action": obj.action,
                            "sleeping": obj.sleeping,
                            "_last_health": obj._last_health,
                        }
                    )
                elif isinstance(obj, objects.Zombie):
                    obj_data.update(
                        {
                            "health": obj.health,
                            "cooldown": getattr(obj, "cooldown", 0),
                        }
                    )
                elif isinstance(obj, objects.Skeleton):
                    obj_data.update(
                        {
                            "health": obj.health,
                            "reload": getattr(obj, "reload", 0),
                        }
                    )
                elif isinstance(obj, objects.Arrow):
                    obj_data.update(
                        {
                            "facing": list(obj.facing) if hasattr(obj, "facing") else None,
                        }
                    )
                elif isinstance(obj, objects.Plant):
                    obj_data.update(
                        {
                            "health": getattr(obj, "health", 1),
                            "grown": getattr(obj, "grown", 0),
                            "kind": getattr(obj, "kind", "unknown"),
                        }
                    )
                elif hasattr(obj, "health"):
                    obj_data["health"] = obj.health

                objects_data.append(obj_data)

        # Save chunks with proper references
        chunks_data = {}
        for key, chunk_objs in self._world._chunks.items():
            chunk_indices = []
            for obj in chunk_objs:
                if obj is not None:
                    try:
                        idx = self._world._objects.index(obj)
                        chunk_indices.append(idx)
                    except ValueError:
                        pass
            chunks_data[key] = chunk_indices

        # Add chunks data to world state
        world_state["chunks"] = chunks_data
        world_state["objects"] = objects_data

        return {
            "step": self._step,
            "seed": self._seed,
            "length": self._length,
            "episode": self._episode,
            "player_idx": player_idx,
            "world": world_state,
        }

    def _load(self, state: Dict[str, Any]) -> None:
        """Load environment state from saved data, preserving everything."""
        # Restore basic attributes
        self._step = state["step"]
        self._seed = state["seed"]
        self._length = state["length"]
        self._episode = state["episode"]

        # Restore world state
        world_data = state["world"]
        self._world.area = world_data["area"]
        self._world.daylight = world_data["daylight"]
        self._world._mat_map = np.array(world_data["_mat_map"])
        self._world._obj_map = np.array(world_data["_obj_map"])
        self._world._mat_names = world_data.get("_mat_names", {})
        self._world._mat_ids = world_data.get("_mat_ids", {})
        self._world._chunk_size = world_data.get("_chunk_size", (16, 16))

        # Clear existing objects
        self._world._objects = []
        self._world._chunks.clear()

        # First pass: Create player object first if it exists
        player_obj = None
        player_idx = state.get("player_idx")

        if player_idx is not None:
            player_data = world_data["objects"][player_idx]
            if player_data and "Player" in player_data["type"]:
                player_obj = objects.Player(self._world, np.array(player_data["pos"]))
                player_obj.facing = (
                    tuple(player_data["facing"]) if player_data.get("facing") else (0, 1)
                )
                player_obj.health = player_data["health"]
                player_obj.inventory = collections.Counter(player_data["inventory"])
                player_obj.achievements = dict(player_data["achievements"])
                player_obj._hunger = player_data["_hunger"]
                player_obj._thirst = player_data["_thirst"]
                player_obj._fatigue = player_data["_fatigue"]
                player_obj._recover = player_data["_recover"]
                player_obj.action = player_data["action"]
                player_obj.sleeping = player_data["sleeping"]
                player_obj._last_health = player_data["_last_health"]
                self._player = player_obj

        # Second pass: Create all objects
        objects_by_idx = {}

        for obj_data in world_data["objects"]:
            if obj_data is None:
                self._world._objects.append(None)
            else:
                idx = obj_data["_index"]

                # Skip player as we already created it
                if idx == player_idx and player_obj is not None:
                    while len(self._world._objects) <= idx:
                        self._world._objects.append(None)
                    self._world._objects[idx] = player_obj
                    objects_by_idx[idx] = player_obj
                    continue

                # Create object based on type
                type_str = obj_data["type"]
                module_name, cls_name = type_str.rsplit(".", 1)

                # Special handling for each object type
                if cls_name == "Zombie":
                    obj = objects.Zombie(self._world, np.array(obj_data["pos"]), player_obj)
                    obj.health = obj_data.get("health", 3)
                    obj.cooldown = obj_data.get("cooldown", 0)
                elif cls_name == "Skeleton":
                    obj = objects.Skeleton(self._world, np.array(obj_data["pos"]), player_obj)
                    obj.health = obj_data.get("health", 2)
                    obj.reload = obj_data.get("reload", 0)
                elif cls_name == "Cow":
                    obj = objects.Cow(self._world, np.array(obj_data["pos"]))
                    obj.health = obj_data.get("health", 3)
                elif cls_name == "Plant":
                    obj = objects.Plant(self._world, np.array(obj_data["pos"]))
                    obj.health = obj_data.get("health", 1)
                    obj.grown = obj_data.get("grown", 0)
                    obj.kind = obj_data.get("kind", "tree")
                elif cls_name == "Arrow":
                    obj = objects.Arrow(
                        self._world,
                        np.array(obj_data["pos"]),
                        tuple(obj_data.get("facing", [0, 1])),
                    )
                elif cls_name in ["Stone", "Coal", "Iron", "Diamond"]:
                    # These are material objects
                    cls = getattr(objects, cls_name)
                    obj = cls(self._world, np.array(obj_data["pos"]))
                elif cls_name in ["Table", "Furnace"]:
                    cls = getattr(objects, cls_name)
                    obj = cls(self._world, np.array(obj_data["pos"]))
                else:
                    # Generic object creation
                    import importlib

                    module = importlib.import_module(module_name)
                    cls = getattr(module, cls_name)
                    obj = cls(self._world, np.array(obj_data["pos"]))
                    if hasattr(obj, "health") and "health" in obj_data:
                        obj.health = obj_data["health"]

                obj.removed = obj_data.get("removed", False)

                # Ensure list is long enough
                while len(self._world._objects) <= idx:
                    self._world._objects.append(None)

                self._world._objects[idx] = obj
                objects_by_idx[idx] = obj

        # Restore chunks
        for key, obj_indices in world_data["chunks"].items():
            # Convert string key back to tuple if needed
            if isinstance(key, str):
                key = eval(key)  # Safe here as we control the format

            chunk_objects = set()
            for idx in obj_indices:
                if idx in objects_by_idx:
                    chunk_objects.add(objects_by_idx[idx])
            if chunk_objects:
                self._world._chunks[key] = chunk_objects

        # Restore random state
        random_state = pickle.loads(world_data["random_state"])
        self._world.random.set_state(random_state)

    # Attach methods to Env class
    crafter.Env.save = _save
    crafter.Env.load = _load

    print("[PATCH] crafter.Env.save() and load() methods added (v3).")
else:
    print("[PATCH] crafter.Env already has save/load methods.")

print("[PATCH] Crafter serialization patch v3 complete.")
