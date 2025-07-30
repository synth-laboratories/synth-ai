"""
test_unsolvable_sokoban.py – make sure A* cannot "solve" an unsolvable level
and that the environment never reports a false positive.
"""

import asyncio
import uuid
from typing import Dict, Any

import pytest

from synth_ai.environments.examples.sokoban.environment import SokobanEnvironment
from synth_ai.environments.examples.sokoban.units.astar_common import (
    astar,
    solved,
)  # buggy solved()!
from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstance,
    SokobanTaskInstanceMetadata,
)
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent

# ───────────────────────────────── snapshot ───────────────────────────── #
UNSOLVABLE_SNAPSHOT: Dict[str, Any] = {
    #  #  #  #  #
    #  _  X  #  #
    #  P  #  #  #
    #  #  #  #  #
    "dim_room": [4, 4],
    "room_fixed": [
        [0, 0, 0, 0],
        [0, 2, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ],
    "room_state": [
        [0, 0, 0, 0],
        [0, 1, 4, 0],
        [0, 5, 0, 0],
        [0, 0, 0, 0],
    ],
    "boxes_on_target": 0,
    "max_steps": 10,
    "num_boxes": 1,
}


# ───────────────────────── helper object wrapper ──────────────────────── #
class Move(EnvToolCall):  # type: ignore[misc]
    def __init__(self, action: int):
        self.action = action


# ──────────────────────────────── test ────────────────────────────────── #
@pytest.mark.asyncio
async def test_unsolvable_level_not_solved():
    """A* should *not* find a solution and the env should *not* claim success."""
    meta = SokobanTaskInstanceMetadata(
        difficulty="unsolvable-unit",
        num_boxes=1,
        dim_room=(4, 4),
        max_steps=10,
        shortest_path_length=-1,
        seed=-1,
        generation_params="unsolvable-test",
    )
    ti = SokobanTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="prove unsolvable"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=meta,
        is_reproducible=True,
        initial_engine_snapshot=UNSOLVABLE_SNAPSHOT,
    )

    env = SokobanEnvironment(ti)
    await env.initialize()
    env.engine.package_sokoban_env.observation_mode = "raw"

    # ★ give the Environment the attribute the heuristic expects
    env.package_sokoban_env = env.engine.package_sokoban_env  # type: ignore[attr-defined]

    root_snapshot = await env._serialize_engine()

    async def custom_deserialize(snapshot: Any) -> SokobanEnvironment:
        new_env = await SokobanEnvironment._deserialize_engine(snapshot)
        new_env.package_sokoban_env = new_env.engine.package_sokoban_env  # type: ignore[attr-defined]
        return new_env

    plan = await astar(
        root_obj=env,
        step_fn=lambda e, a: e.step([[Move(a)]]),
        deserialize_fn=custom_deserialize,  # Use the wrapper
        max_nodes=500,
    )

    # ---------------- expected behaviour ---------------- #
    # (1) Search should *not* find a plan.
    assert not plan, f"A* unexpectedly found a plan: {plan}"

    # (2) Even if a buggy plan exists, replay must not mark the puzzle solved.
    # Re-initialize environment for replay to ensure a clean state
    env_for_replay = await SokobanEnvironment._deserialize_engine(root_snapshot)
    # ★ give the new Environment instance the attribute the heuristic expects
    env_for_replay.package_sokoban_env = env_for_replay.engine.package_sokoban_env  # type: ignore[attr-defined]

    for action_code in plan or []:
        await env_for_replay.step([[Move(action_code)]])

    solved_after_replay = solved(env_for_replay)  # Use the existing buggy solved()
    assert not solved_after_replay, "Environment incorrectly reports a solved state"

    # If the current code base still has the buggy `solved()` that checks the
    # player instead of the boxes, this test will **fail** here — that's the
    # signal to apply the earlier fixes.


if __name__ == "__main__":
    asyncio.run(test_unsolvable_level_not_solved())
