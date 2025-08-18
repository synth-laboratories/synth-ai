"""
astar_common.py – one A* routine usable by both engine-level and
environment-level unit tests.
"""

import heapq
import itertools
import json
from typing import Any, Awaitable, Callable, List, Tuple

import numpy as np


# ---------- generic utilities ------------------------------------ #
def _boxes_left(env_pkg) -> int:
    """#targets – #boxes-on-targets (uses raw grids, never the counter)."""
    return int(np.sum(env_pkg.room_fixed == 2) - np.sum(env_pkg.room_state == 3))


def solved(obj: Any) -> bool:
    """Expects obj to have a .package_sokoban_env attribute."""
    return _boxes_left(obj.package_sokoban_env) == 0


def heuristic(obj: Any) -> int:
    """Expects obj to have a .package_sokoban_env attribute."""
    return _boxes_left(obj.package_sokoban_env)


# ---------- single reusable A* ----------------------------------- #
async def astar(
    root_obj: Any,
    step_fn: Callable[[Any, int], Awaitable[None]],
    deserialize_fn: Callable[[Any], Awaitable[Any]],
    max_nodes: int = 1000,
) -> List[int]:
    """
    Generic A* over Sokoban snapshots.

    • `root_obj` - current engine *or* environment
    • `step_fn(obj, action)` - async: apply one move to *obj*
    • `deserialize_fn(snapshot)` - async: new obj from snapshot
    """
    start_snap = await root_obj._serialize_engine()

    frontier: List[Tuple[int, int, Any, List[int]]] = []
    counter = itertools.count()
    frontier.append((heuristic(root_obj), next(counter), start_snap, []))
    seen: set[str] = set()

    nodes = 0
    while frontier and nodes < max_nodes:
        f, _, snap, path = heapq.heappop(frontier)
        cur = await deserialize_fn(snap)
        key = json.dumps(snap.engine_snapshot, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if solved(cur):
            return path

        nodes += 1
        for action in range(cur.package_sokoban_env.action_space.n):
            child = await deserialize_fn(snap)  # fresh copy
            try:
                await step_fn(child, action)
            except Exception:  # illegal/off-board
                continue

            child_snap = await child._serialize_engine()
            g = len(path) + 1
            heapq.heappush(
                frontier,
                (g + heuristic(child), next(counter), child_snap, path + [action]),
            )
    return []


# convenience lambdas for the two concrete APIs
async def _engine_step(e, a):  # `SokobanEngine`
    await e._step_engine(a)


async def _env_step(env, a):  # `SokobanEnvironment` (expects Move wrapper)
    from synth_ai.environments.examples.sokoban.units.test_sokoban_environment import Move

    await env.step([[Move(a)]])


ENGINE_ASTAR = lambda eng, **kw: astar(eng, _engine_step, eng.__class__._deserialize_engine, **kw)
ENV_ASTAR = lambda env, **kw: astar(
    env.engine, _env_step, env.engine.__class__._deserialize_engine, **kw
)

# ----------------------------------------------------------------- #
