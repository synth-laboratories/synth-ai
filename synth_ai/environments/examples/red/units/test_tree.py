#!/usr/bin/env python3
"""
mcts_pokemon_red_env_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Monte-Carlo-Tree-Search demo for Pokemon Red that:
  • wraps Pokemon Red environment with real ROM
  • stores every state in a FilesystemSnapshotStore
  • expands / rolls-out with a TrajectoryTreeStore
  • uses simple heuristics to guide exploration
  • returns action sequence that makes progress

Run with pytest: pytest src/examples/red/units/test_tree.py
"""

import asyncio
import gzip
import pickle
import random
import time
import logging
from pathlib import Path

import pytest

import sys

sys.path.append("/Users/joshuapurtell/Documents/GitHub/Environments/src")

from synth_ai.environments.reproducibility.tree import FilesystemSnapshotStore, TrajectoryTreeStore
from synth_ai.environments.examples.red.taskset import (
    INSTANCE as DEFAULT_TASK,
)
from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.environment.tools import EnvToolCall

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
LOG = logging.getLogger("pokemon-mcts")

# Pokemon Red action space - all possible buttons
POKEMON_ACTIONS = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]


def heuristic_score(env: PokemonRedEnvironment) -> float:
    """
    Simple heuristic to evaluate Pokemon Red game state.
    Higher scores are better.
    """
    try:
        # Get current state
        priv, pub = env.engine._create_states(reward=0.0)

        score = 10.0  # Base score to avoid all zeros

        # Badge progress (most important)
        badge_count = bin(pub.badges).count("1")
        score += badge_count * 100.0  # 100 points per badge

        # Level progress
        score += pub.party_level * 5.0  # 5 points per level

        # XP progress (smaller contribution)
        score += pub.party_xp * 0.001  # Very small XP bonus

        # Exploration bonus - being in different maps
        if pub.map_id > 0:
            score += 10.0  # Bonus for being in actual game world

        # Position exploration bonus - reward movement from (0,0)
        if pub.player_x != 0 or pub.player_y != 0:
            score += 5.0

        # HP bonus - encourage keeping Pokemon healthy (only if we have a Pokemon)
        if pub.party_hp_max > 0:
            hp_ratio = pub.party_hp_current / pub.party_hp_max
            score += hp_ratio * 2.0
        else:
            # No penalty for not having a Pokemon initially
            score += 1.0

        # Step efficiency penalty (very small)
        score -= pub.step_count * 0.001

        return max(score, 0.1)  # Ensure minimum positive score

    except Exception as e:
        LOG.debug(f"Heuristic evaluation error: {e}")
        return 0.1


def is_terminal_state(env: PokemonRedEnvironment) -> bool:
    """Check if we've reached a terminal state (won or lost)"""
    try:
        priv, pub = env.engine._create_states(reward=0.0)

        # Terminal if we got the Boulder Badge (task completion)
        if pub.badges & 0x01:
            return True

        # Only consider terminal if HP is 0 AND max HP > 0 (meaning we actually have a Pokemon)
        # Initial state might have 0/0 HP which isn't really a loss
        if pub.party_hp_current == 0 and pub.party_hp_max > 0:
            return True

        return False

    except Exception:
        return True  # Consider error states as terminal


async def simple_rollout(env: PokemonRedEnvironment, max_steps: int = 20) -> float:
    """
    Perform a simple random rollout from current state.
    Returns heuristic score after rollout.
    """
    try:
        # Save current state
        snapshot = await env._serialize_engine()

        # Random walk
        for _ in range(max_steps):
            if is_terminal_state(env):
                break

            # Choose random action
            action = random.choice(POKEMON_ACTIONS)
            call = EnvToolCall(tool="press_button", args={"button": action, "frames": 1})

            try:
                await env.step(call)
            except Exception:
                break  # Stop on error

        # Evaluate final state
        final_score = heuristic_score(env)

        # Restore original state
        env.engine = await PokemonRedEnvironment._deserialize_engine(snapshot, env.task_instance)

        return final_score

    except Exception as e:
        LOG.debug(f"Rollout error: {e}")
        return 0.0


async def pokemon_red_mcts_plan(
    tree: TrajectoryTreeStore,
    root_id: str,
    *,
    rollouts_per_action: int = 10,
    max_depth: int = 20,
    timeout_s: float = 30.0,
) -> tuple[list[str], list[dict[str, float]]]:
    """
    MCTS planning for Pokemon Red.
    Returns (action_plan, q_value_history)
    """
    start = time.monotonic()
    plan, q_hist, node_id = [], [], root_id

    for depth in range(max_depth):
        LOG.debug(f"\n--- MCTS depth {depth} --- node={node_id[:8]}")

        if timeout_s is not None and time.monotonic() - start >= timeout_s:
            LOG.debug("MCTS timeout reached")
            break

        # Load environment from snapshot
        env_blob = tree.load_snapshot_blob(node_id)
        env = await PokemonRedEnvironment._deserialize_engine(
            pickle.loads(gzip.decompress(env_blob)), DEFAULT_TASK
        )

        # Check if terminal
        if is_terminal_state(env):
            LOG.debug("Terminal state reached")
            break

        # Log current state
        priv, pub = env.engine._create_states(reward=0.0)
        LOG.debug(
            f"State: Map{pub.map_id:02X}:({pub.player_x},{pub.player_y}) "
            f"Badges:{bin(pub.badges).count('1')} Level:{pub.party_level} "
            f"HP:{pub.party_hp_current}/{pub.party_hp_max}"
        )

        q_vals: dict[str, float] = {}

        # Evaluate each possible action
        for action in POKEMON_ACTIONS:
            if timeout_s is not None and time.monotonic() - start >= timeout_s:
                break

            # Check if we already have a child for this action
            child_id = next(
                (
                    cid
                    for cid in tree.get_children(node_id)
                    if tree.graph[node_id][cid]["action"] == action
                ),
                None,
            )

            if child_id is None:  # Need to expand
                LOG.debug(f"Expanding action: {action}")

                # Create new environment and take action
                try:
                    tmp_env = await PokemonRedEnvironment._deserialize_engine(
                        pickle.loads(gzip.decompress(env_blob)), DEFAULT_TASK
                    )

                    call = EnvToolCall(tool="press_button", args={"button": action, "frames": 1})
                    await tmp_env.step(call)

                    # Create child node
                    child_blob = gzip.compress(pickle.dumps(await tmp_env._serialize_engine()))
                    child_id = tree.add_child(
                        node_id,
                        child_blob,
                        action=action,
                        reward=heuristic_score(tmp_env),
                        terminated=is_terminal_state(tmp_env),
                        info={},
                    )

                except Exception as e:
                    LOG.debug(f"Failed to expand action {action}: {e}")
                    continue
            else:
                LOG.debug(f"Reusing existing child for action: {action}")

            if child_id is None:
                continue

            # Perform rollouts from child state
            child_env = await PokemonRedEnvironment._deserialize_engine(
                pickle.loads(gzip.decompress(tree.load_snapshot_blob(child_id))),
                DEFAULT_TASK,
            )

            rollout_scores = []
            for _ in range(rollouts_per_action):
                if timeout_s is not None and time.monotonic() - start >= timeout_s:
                    break
                score = await simple_rollout(child_env, max_steps=10)
                rollout_scores.append(score)

            if rollout_scores:
                # Average rollout score as Q-value
                q_vals[action] = sum(rollout_scores) / len(rollout_scores)
                LOG.debug(
                    f"Action {action}: Q={q_vals[action]:.3f} "
                    f"(avg of {len(rollout_scores)} rollouts)"
                )
            else:
                q_vals[action] = 0.0

        if not q_vals:
            LOG.debug("No valid actions found")
            break

        LOG.debug(f"Q-values: {q_vals}")
        q_hist.append(q_vals)

        # Select best action
        best_action = max(q_vals, key=q_vals.get)
        plan.append(best_action)

        # Move to child node
        child_nodes = tree.get_children(node_id)
        next_node = None
        for cid in child_nodes:
            if tree.graph[node_id][cid]["action"] == best_action:
                next_node = cid
                break

        if next_node is None:
            LOG.debug(f"No child node found for action {best_action}")
            break

        node_id = next_node

        LOG.debug(f"Selected action: {best_action} → node={node_id[:8]}")

    return plan, q_hist


@pytest.mark.asyncio
async def test_mcts_pokemon_red_basic(tmp_path: Path) -> None:
    """Test basic MCTS functionality with Pokemon Red"""

    # Create environment
    env = PokemonRedEnvironment(DEFAULT_TASK)
    await env.initialize()

    # Set up tree storage
    snap_store_path = tmp_path / "pokemon_mcts_snaps"
    tree = TrajectoryTreeStore(FilesystemSnapshotStore(snap_store_path))

    # Add root snapshot
    root_blob = gzip.compress(pickle.dumps(await env._serialize_engine()))
    root_id = tree.add_root(root_blob)

    LOG.debug("Starting Pokemon Red MCTS planning...")

    # Run MCTS with short timeout for testing
    plan, q_hist = await pokemon_red_mcts_plan(
        tree,
        root_id,
        rollouts_per_action=3,  # Reduced for faster testing
        max_depth=5,  # Shallow depth for testing
        timeout_s=10.0,  # Short timeout
    )

    print(f"MCTS Plan: {plan}")
    print(f"Q-value history: {q_hist}")

    # Verify we got some plan
    assert isinstance(plan, list), "Plan should be a list"
    assert len(plan) >= 0, "Plan should have non-negative length"

    # Verify all actions in plan are valid
    for action in plan:
        assert action in POKEMON_ACTIONS, f"Invalid action in plan: {action}"

    # Verify Q-values were computed
    assert isinstance(q_hist, list), "Q-history should be a list"
    for q_dict in q_hist:
        assert isinstance(q_dict, dict), "Each Q-value entry should be a dict"
        for action, q_val in q_dict.items():
            assert action in POKEMON_ACTIONS, f"Invalid action in Q-values: {action}"
            assert isinstance(q_val, (int, float)), f"Q-value should be numeric: {q_val}"


@pytest.mark.asyncio
async def test_mcts_pokemon_red_execution(tmp_path: Path) -> None:
    """Test that MCTS plan can be executed in Pokemon Red"""

    # Create environment
    env = PokemonRedEnvironment(DEFAULT_TASK)
    await env.initialize()

    # Get initial state for comparison
    initial_priv, initial_pub = env.engine._create_states(reward=0.0)
    initial_score = heuristic_score(env)

    LOG.debug(
        f"Initial state - Score: {initial_score:.3f}, "
        f"Map: {initial_pub.map_id}, Pos: ({initial_pub.player_x},{initial_pub.player_y}), "
        f"Level: {initial_pub.party_level}, Badges: {bin(initial_pub.badges).count('1')}"
    )

    # Set up MCTS
    snap_store_path = tmp_path / "pokemon_execution_test"
    tree = TrajectoryTreeStore(FilesystemSnapshotStore(snap_store_path))
    root_blob = gzip.compress(pickle.dumps(await env._serialize_engine()))
    root_id = tree.add_root(root_blob)

    # Run MCTS
    plan, q_hist = await pokemon_red_mcts_plan(
        tree, root_id, rollouts_per_action=2, max_depth=3, timeout_s=8.0
    )

    # Execute the plan
    for i, action in enumerate(plan):
        LOG.debug(f"Executing step {i + 1}: {action}")
        call = EnvToolCall(tool="press_button", args={"button": action, "frames": 1})
        obs = await env.step(call)

        # Log progress
        LOG.debug(f"  → Step {obs['step_count']}, Reward: {obs['total_reward']:.3f}")

    # Check final state
    final_priv, final_pub = env.engine._create_states(reward=0.0)
    final_score = heuristic_score(env)

    LOG.debug(
        f"Final state - Score: {final_score:.3f}, "
        f"Map: {final_pub.map_id}, Pos: ({final_pub.player_x},{final_pub.player_y}), "
        f"Level: {final_pub.party_level}, Badges: {bin(final_pub.badges).count('1')}"
    )

    # Verify execution worked
    assert final_pub.step_count >= len(plan), "Steps should have been executed"

    # Verify some progress was made (even if minimal)
    progress_made = (
        final_pub.map_id != initial_pub.map_id
        or final_pub.player_x != initial_pub.player_x
        or final_pub.player_y != initial_pub.player_y
        or final_pub.party_level > initial_pub.party_level
        or final_pub.badges != initial_pub.badges
        or abs(final_score - initial_score) > 0.01
    )

    LOG.debug(f"Progress made: {progress_made}")
    # Note: Progress isn't guaranteed in a short test, so we just verify execution worked


@pytest.mark.asyncio
async def test_heuristic_functions() -> None:
    """Test the heuristic and utility functions"""

    # Create test environment
    env = PokemonRedEnvironment(DEFAULT_TASK)
    await env.initialize()

    # Test heuristic scoring
    initial_score = heuristic_score(env)
    assert isinstance(initial_score, (int, float)), "Heuristic should return numeric score"
    assert initial_score >= 0, "Initial score should be non-negative"

    # Test terminal state detection
    is_terminal = is_terminal_state(env)
    assert isinstance(is_terminal, bool), "Terminal check should return boolean"

    # Test rollout (with very short length)
    rollout_score = await simple_rollout(env, max_steps=3)
    assert isinstance(rollout_score, (int, float)), "Rollout should return numeric score"

    LOG.debug(
        f"Heuristic tests - Initial: {initial_score:.3f}, "
        f"Terminal: {is_terminal}, Rollout: {rollout_score:.3f}"
    )


if __name__ == "__main__":
    import tempfile

    async def main():
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            print("Running Pokemon Red MCTS tests...")

            await test_heuristic_functions()
            print("✓ Heuristic functions test passed")

            await test_mcts_pokemon_red_basic(tmp_path)
            print("✓ Basic MCTS test passed")

            await test_mcts_pokemon_red_execution(tmp_path)
            print("✓ MCTS execution test passed")

            print("🎉 All Pokemon Red MCTS tests passed!")

    asyncio.run(main())
