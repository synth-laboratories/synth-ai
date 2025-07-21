#!/usr/bin/env python3
"""
mcts_sokoban_env_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tiny Monte-Carlo-Tree-Search demo that
  • wraps a 4×4 toy Sokoban level in `SokobanEnvironment`
  • stores every state in a FilesystemSnapshotStore
  • expands / rolls-out with a TrajectoryTreeStore
  • returns the most visited root-child as the "plan"

Run with pytest: pytest Environments/examples/sokoban/units/test_tree.py
"""

import asyncio
import gzip
import pickle
import time
import logging
import uuid
from pathlib import Path

import numpy as np
import pytest

from synth_ai.environments.reproducibility.tree import FilesystemSnapshotStore, TrajectoryTreeStore
from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstance,
    SokobanTaskInstanceMetadata,
)

# from examples.sokoban.engine import SokobanEngineSnapshot  # only a type
from synth_ai.environments.tasks.core import Impetus, Intent
from synth_ai.environments.examples.sokoban.environment import SokobanEnvironment
from synth_ai.environments.examples.sokoban.units.astar_common import ENGINE_ASTAR  # A* helper
from gym_sokoban.envs.sokoban_env import ACTION_LOOKUP  # Added for full action set

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
LOG = logging.getLogger("mcts-debug")

# ─────────────────────────── toy level ──────────────────────────────── #

SNAP = {
    "dim_room": [4, 4],
    "room_fixed": [
        [0, 0, 0, 0],
        [0, 1, 2, 1],  # target at (1,2)
        [0, 1, 1, 1],
        [0, 0, 0, 0],
    ],
    "room_state": [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 4, 1],  # box at (2,2)
        [0, 5, 1, 1],  # player at (3,1)
    ],
    "boxes_on_target": 0,
    "max_steps": 10,
    "num_boxes": 1,
}


# ─────────────────────────── env wrapper ────────────────────────────── #
# (import placed here to avoid circulars; uses the code you pasted)

# ─────────────────────────── helpers ─────────────────────────────────── #


def solved(env: SokobanEnvironment) -> bool:
    """All targets covered?"""
    eng = env.engine
    return eng.package_sokoban_env.boxes_on_target == np.sum(
        eng.package_sokoban_env.room_fixed == 2
    )


async def debug_basic_actions():
    """Debug function to test basic action execution"""
    inst = SokobanTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="solve"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=SokobanTaskInstanceMetadata(
            difficulty="easy",
            num_boxes=1,
            dim_room=(4, 4),
            max_steps=10,
            shortest_path_length=-1,
            seed=0,
            generation_params="debug",
        ),
        is_reproducible=True,
        initial_engine_snapshot=SNAP,
    )
    env = SokobanEnvironment(inst)
    await env.initialize()

    LOG.debug("Initial state:")
    LOG.debug(f"player @ {env.engine.package_sokoban_env.player_position}")
    LOG.debug(f"room_state:\n{env.engine.package_sokoban_env.room_state}")
    LOG.debug(f"room_fixed:\n{env.engine.package_sokoban_env.room_fixed}")
    LOG.debug(f"boxes_on_target AFTER INIT: {env.engine.package_sokoban_env.boxes_on_target}")
    LOG.debug(
        f"Actual count of boxes on target (value 3): {np.sum(env.engine.package_sokoban_env.room_state == 3)}"
    )
    LOG.debug(f"Boxes at value 4: {np.sum(env.engine.package_sokoban_env.room_state == 4)}")
    LOG.debug(f"Targets at value 2: {np.sum(env.engine.package_sokoban_env.room_fixed == 2)}")

    # Try each action individually
    for a in range(len(ACTION_LOOKUP)):
        LOG.debug(f"\nTrying action {a} ({ACTION_LOOKUP[a]}):")
        snapshot = await env._serialize_engine()
        try:
            test_env = await SokobanEnvironment._deserialize_engine(snapshot, inst)
            LOG.debug(
                f"  Before: player @ {test_env.engine.package_sokoban_env.player_position}, boxes @ {np.argwhere((test_env.engine.package_sokoban_env.room_state == 3) | (test_env.engine.package_sokoban_env.room_state == 4))}"
            )
            await test_env.engine._step_engine(a)
            LOG.debug(
                f"  After:  player @ {test_env.engine.package_sokoban_env.player_position}, boxes @ {np.argwhere((test_env.engine.package_sokoban_env.room_state == 3) | (test_env.engine.package_sokoban_env.room_state == 4))}"
            )
            LOG.debug(
                f"  Action {a} succeeded, boxes_on_target: {test_env.engine.package_sokoban_env.boxes_on_target}"
            )
        except Exception as e:
            LOG.debug(f"  Action {a} failed: {e}")

    return env


# ╰────────────────────────────────────────────────────────────────────╯ #


# ───────── greedy search that *writes/reads* via TrajectoryTreeStore ─────────
async def greedy_tree_mcts_plan(
    tree: TrajectoryTreeStore,
    root_id: str,
    task_instance: SokobanTaskInstance,
    *,
    rollouts_per_action: int = 50,
    max_depth: int = 30,
    timeout_s: float | None = None,  # Added timeout parameter
) -> tuple[list[int], list[dict[int, float]]]:
    start = time.monotonic()  # Start timer
    plan, q_hist, node_id = [], [], root_id

    for depth in range(max_depth):
        LOG.debug(f"\n--- depth {depth} --- node={node_id[:6]}")  # LOGGING
        if timeout_s is not None and time.monotonic() - start >= timeout_s:
            break  # time budget exhausted

        env_blob = tree.load_snapshot_blob(node_id)
        env = await SokobanEnvironment._deserialize_engine(
            pickle.loads(gzip.decompress(env_blob)), task_instance
        )
        LOG.debug(
            f"player @ {env.engine.package_sokoban_env.player_position} boxes @ {np.argwhere((env.engine.package_sokoban_env.room_state == 3) | (env.engine.package_sokoban_env.room_state == 4))}"
        )  # LOGGING

        # legal_n = env.engine.package_sokoban_env.action_space.n # Old way
        q_vals: dict[int, float] = {}  # Initialize q_vals here
        # enumerate every Sokoban action (4 moves + 4 pushes + no-op = 9)
        for a in range(len(ACTION_LOOKUP)):  # Use full ACTION_LOOKUP length
            if timeout_s is not None and time.monotonic() - start >= timeout_s:
                break  # time budget exhausted in inner loop

            action_type_log = ""
            child_id = next(
                (
                    cid
                    for cid in tree.get_children(node_id)
                    if tree.graph[node_id][cid]["action"] == a
                ),
                None,
            )

            if child_id is None:  # expand once
                action_type_log = f"expand a={a}"  # Store log message
                # Create a new environment from the current env state for stepping
                tmp_env_for_step = await SokobanEnvironment._deserialize_engine(
                    pickle.loads(gzip.decompress(env_blob)),
                    task_instance,  # Re-deserialize parent to ensure clean state for step
                )
                try:
                    await tmp_env_for_step.engine._step_engine(a)
                except Exception:  # Catch potential errors from illegal actions
                    # q_vals[a] = -1.0 # No q-value assigned if action is illegal and cannot be expanded
                    LOG.debug(f"illegal expand a={a}, skipping")  # Log illegal action here
                    continue  # illegal → skip
                cid_blob = gzip.compress(pickle.dumps(await tmp_env_for_step._serialize_engine()))
                child_id = tree.add_child(
                    node_id,
                    cid_blob,
                    action=a,
                    reward=0.0,
                    terminated=solved(tmp_env_for_step),
                    info={},
                )
            else:
                action_type_log = f"reuse   a={a}"  # Store log message

            # deterministic rollout: A* from child snapshot
            if child_id is None:
                # This case should ideally be hit if the 'continue' for illegal expansion was triggered.
                # No valid child_id means no Q-value can be computed.
                continue

            child_env = await SokobanEnvironment._deserialize_engine(
                pickle.loads(gzip.decompress(tree.load_snapshot_blob(child_id))),
                task_instance,
            )
            # run A* on the *engine*, not the env wrapper
            path = await ENGINE_ASTAR(child_env.engine, max_nodes=1_000)  # try to solve

            # Calculate Q-value considering the cost of the current action 'a'
            if path is None:  # search failed / gave up
                q_value_for_a = 0.0
            elif len(path) == 0:  # child state already solved
                # 1 for the current action 'a', but state is solved, so highest Q.
                # The '1 + len(path)' for total_len doesn't strictly apply here in the same way,
                # as no further steps are needed from A*.
                # Assigning 1.0 directly makes it the best possible Q.
                q_value_for_a = (
                    1.0  # best possible, accounts for the step taken to reach solved state.
                )
            else:  # solution in |path| further steps
                # cost of taking the action itself  ↓
                total_len = 1 + len(path)  # 1 = the step 'a' + A* path from child
                q_value_for_a = 1.0 / (
                    1 + total_len
                )  # shorter total → higher Q. Effectively 1.0 / (2 + len(path))

            q_vals[a] = q_value_for_a

            # Prepare path string for logging
            if path is None:
                path_str = "No solution found / A* failed"
            elif path == []:
                path_str = "✓ (already solved)"
            else:
                path_str = str(path)
            LOG.debug(
                f"{action_type_log}, Q={q_value_for_a:.4f}, Path={path_str}"
            )  # Log action type, Q, and A* path

        if not q_vals:  # No actions evaluated, possibly due to timeout or all actions illegal
            break
        LOG.debug(f"Q={q_vals}")  # LOGGING

        q_hist.append(q_vals)

        current_children_ids = tree.get_children(node_id)
        if current_children_ids is None:
            current_children_ids = []  # Ensure it's an iterable for the comprehension

        valid_actions = {
            action_key: q_value
            for action_key, q_value in q_vals.items()
            # Ensure that the action resulted in an actual child node being added to the tree
            if any(
                tree.graph[node_id][child_id_loop]["action"] == action_key
                for child_id_loop in current_children_ids
            )
        }

        if not valid_actions:
            # This means no actions evaluated (or timed out) or none of the evaluated actions
            # correspond to an actual created child node (e.g., all were illegal).
            LOG.debug(f"No valid actions available at depth {depth}. Stopping MCTS.")
            break

        best_a = max(valid_actions, key=valid_actions.get)
        plan.append(best_a)

        # Move to child node
        node_id = next(
            cid_loop
            for cid_loop in current_children_ids
            if tree.graph[node_id][cid_loop]["action"] == best_a
        )
        LOG.debug(f"best={best_a} → new node={node_id[:6]}")  # LOGGING
        # Check if the chosen child is a terminal state, and break after adding final action
        child_blob = tree.load_snapshot_blob(node_id)
        child_env = await SokobanEnvironment._deserialize_engine(
            pickle.loads(gzip.decompress(child_blob)), task_instance
        )
        if solved(child_env):
            LOG.debug("Child node is terminal. Ending plan.")
            break

    return plan, q_hist


# ───────────────────── pytest driver (add this AFTER helpers) ─────────────
@pytest.mark.asyncio
async def test_mcts_sokoban_run(tmp_path: Path) -> None:
    # 1) build an env around the tiny 4×4 level
    inst = SokobanTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="solve"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=SokobanTaskInstanceMetadata(
            difficulty="easy",
            num_boxes=1,
            dim_room=(4, 4),
            max_steps=10,
            shortest_path_length=-1,
            seed=0,
            generation_params="demo",
        ),
        is_reproducible=True,
        initial_engine_snapshot=SNAP,
    )
    env = SokobanEnvironment(inst)
    await env.initialize()

    # 2) root snapshot → tree
    snap_store_path = tmp_path / "mcts_snaps"
    tree = TrajectoryTreeStore(FilesystemSnapshotStore(snap_store_path))
    root_blob = gzip.compress(pickle.dumps(await env._serialize_engine()))
    root_id = tree.add_root(root_blob)

    # Diagnostic: Test A* directly on the root SNAP state
    diag_env = await SokobanEnvironment._deserialize_engine(
        pickle.loads(gzip.decompress(root_blob)), inst
    )
    LOG.debug("Diagnostic A* on initial state with max_nodes=5000:")
    diag_path = await ENGINE_ASTAR(diag_env.engine, max_nodes=5000)
    LOG.debug(f"Diagnostic A* path from root: {diag_path if diag_path else 'No solution found'}")

    # 3) greedy tree search
    plan, q_hist = await greedy_tree_mcts_plan(
        tree,
        root_id,
        inst,
        rollouts_per_action=50,
        max_depth=30,
        timeout_s=30.0,
    )
    print("plan:", plan)
    print("q-history:", q_hist)
    assert plan, "empty plan"

    # 4) verify the plan solves the puzzle
    checker_env = await SokobanEnvironment._deserialize_engine(
        pickle.loads(gzip.decompress(root_blob)), inst
    )
    for a in plan:
        await checker_env.engine._step_engine(a)
    assert solved(checker_env), "plan did not solve the puzzle"


# Removed if __name__ == "__main__": block as pytest handles execution
if __name__ == "__main__":
    import tempfile

    async def main():
        # First run debug to understand action issues
        await debug_basic_actions()

        # Then run the main test
        with tempfile.TemporaryDirectory() as tmpdir:
            await test_mcts_sokoban_run(Path(tmpdir))

    asyncio.run(main())
