"""
test_sokoban_environment.py – A*‑style search and replay, but through the
*SokobanEnvironment* API (initialize/step/checkpoint) rather than talking to
SokobanEngine directly.
"""

import asyncio
from typing import List, Dict, Any
from uuid import uuid4

import pytest

# ––––– app imports ––––– #
from synth_ai.environments.examples.sokoban.environment import SokobanEnvironment
from synth_ai.environments.examples.sokoban.engine import SokobanEngineSnapshot
from synth_ai.environments.environment.tools import EnvToolCall

# shared A* / heuristic utilities
from synth_ai.environments.examples.sokoban.units.astar_common import astar, solved

from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstanceMetadata,
    SokobanTaskInstance,
)
from synth_ai.environments.tasks.core import Impetus, Intent


# ---------------- test fixture snapshot ---------------------------------- #
SIMPLE_SNAPSHOT: Dict[str, Any] = {
    "dim_room": [4, 4],
    "room_fixed": [[0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
    "room_state": [[0, 0, 0, 0], [0, 1, 4, 0], [0, 5, 0, 0], [0, 0, 0, 0]],
    "boxes_on_target": 0,
    "max_steps": 10,
    "num_boxes": 1,
}


# helper: tiny wrapper so we don't depend on full EnvToolCall implementation
class Move(EnvToolCall):  # type: ignore[misc]
    def __init__(self, action: int):
        self.action = action


async def replay(env: SokobanEnvironment, start: SokobanEngineSnapshot, plan: List[int]) -> bool:
    env = await SokobanEnvironment._deserialize_engine(start)
    for a in plan:
        await env.step([[Move(a)]])
    return solved(env)


# ----------------------------- test -------------------------------------- #
@pytest.mark.asyncio
async def test_environment_solve_and_replay():
    # build minimal TaskInstance
    meta = SokobanTaskInstanceMetadata(
        difficulty="easy",
        num_boxes=1,
        dim_room=(4, 4),
        max_steps=10,
        shortest_path_length=-1,
        seed=-1,
        generation_params="unit‑test",
    )
    ti = SokobanTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="solve"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=meta,
        is_reproducible=True,
        initial_engine_snapshot=SIMPLE_SNAPSHOT,
    )

    env = SokobanEnvironment(ti)
    await env.initialize()

    # speed-up: disable image rendering inside gym-sokoban
    env.engine.package_sokoban_env.observation_mode = "raw"

    root_snapshot = await env._serialize_engine()

    # plan search (pass custom step_fn using our Move wrapper)
    plan = await astar(
        root_obj=env,
        step_fn=lambda e, act: e.step([[Move(act)]]),  # Renamed action to act to avoid conflict
        deserialize_fn=SokobanEnvironment._deserialize_engine,
        max_nodes=500,  # circuit-breaker
    )
    assert plan, "Environment A* failed to find a plan"
    assert len(plan) == 2  # expect the 2-move solution

    # verify replay
    assert await replay(env, root_snapshot, plan)


if __name__ == "__main__":
    asyncio.run(test_environment_solve_and_replay())
    pass
