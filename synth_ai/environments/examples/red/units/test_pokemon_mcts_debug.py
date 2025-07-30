#!/usr/bin/env python3
"""Debug Pokemon Red MCTS to see what's happening"""

import sys

sys.path.append("/Users/joshuapurtell/Documents/GitHub/Environments/src")

import asyncio
import logging
from pathlib import Path
import tempfile
import gzip
import pickle

from synth_ai.environments.reproducibility.tree import FilesystemSnapshotStore, TrajectoryTreeStore
from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.examples.red.taskset import INSTANCE as DEFAULT_TASK
from synth_ai.environments.environment.tools import EnvToolCall

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("pokemon-debug")


async def debug_pokemon_mcts():
    """Debug what's happening in Pokemon Red MCTS"""

    print("=== Pokemon Red MCTS Debug ===")

    # Create environment
    env = PokemonRedEnvironment(DEFAULT_TASK)
    await env.initialize()

    # Check initial state
    priv, pub = env.engine._create_states(reward=0.0)
    print("Initial state:")
    print(f"  Map: {pub.map_id} ({pub.map_id:02X})")
    print(f"  Position: ({pub.player_x}, {pub.player_y})")
    print(f"  Badges: {pub.badges} (count: {bin(pub.badges).count('1')})")
    print(f"  Level: {pub.party_level}")
    print(f"  HP: {pub.party_hp_current}/{pub.party_hp_max}")
    print(f"  XP: {pub.party_xp}")
    print(f"  Steps: {pub.step_count}")
    print(f"  Terminated: {priv.terminated}")

    # Test heuristic
    from synth_ai.environments.examples.red.units.test_tree import (
        heuristic_score,
        is_terminal_state,
    )

    score = heuristic_score(env)
    terminal = is_terminal_state(env)
    print(f"  Heuristic score: {score}")
    print(f"  Is terminal: {terminal}")

    print("\n=== Testing Actions ===")

    # Test each action to see what happens
    actions = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]

    for action in actions:
        # Save state
        snapshot = await env._serialize_engine()

        print(f"\nTesting action: {action}")

        try:
            # Take action
            call = EnvToolCall(tool="press_button", args={"button": action, "frames": 1})
            obs = await env.step(call)

            # Check what changed
            new_priv, new_pub = env.engine._create_states(reward=0.0)
            new_score = heuristic_score(env)

            changes = []
            if new_pub.map_id != pub.map_id:
                changes.append(f"map: {pub.map_id} → {new_pub.map_id}")
            if new_pub.player_x != pub.player_x:
                changes.append(f"x: {pub.player_x} → {new_pub.player_x}")
            if new_pub.player_y != pub.player_y:
                changes.append(f"y: {pub.player_y} → {new_pub.player_y}")
            if new_pub.party_level != pub.party_level:
                changes.append(f"level: {pub.party_level} → {new_pub.party_level}")
            if new_pub.badges != pub.badges:
                changes.append(f"badges: {pub.badges} → {new_pub.badges}")
            if new_pub.party_hp_current != pub.party_hp_current:
                changes.append(f"hp: {pub.party_hp_current} → {new_pub.party_hp_current}")

            print(f"  Changes: {changes if changes else 'None'}")
            print(f"  Reward: {obs.get('reward_last_step', 'N/A')}")
            print(f"  Score: {pub_score:.3f} → {new_score:.3f} (Δ{new_score - score:.3f})")
            print(f"  Steps: {new_pub.step_count}")

        except Exception as e:
            print(f"  ERROR: {e}")

        # Restore state
        env.engine = await PokemonRedEnvironment._deserialize_engine(snapshot, env.task_instance)
        pub_score = score  # Reset for next iteration

    print("\n=== Testing Tree Operations ===")

    # Test tree operations
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_store_path = Path(tmpdir) / "debug_snaps"
        tree = TrajectoryTreeStore(FilesystemSnapshotStore(snap_store_path))

        # Add root
        root_blob = gzip.compress(pickle.dumps(await env._serialize_engine()))
        root_id = tree.add_root(root_blob)
        print(f"Root ID: {root_id[:8]}...")

        # Test expanding one action
        action = "A"
        print(f"\nExpanding action: {action}")

        try:
            # Load env from blob
            test_env = await PokemonRedEnvironment._deserialize_engine(
                pickle.loads(gzip.decompress(root_blob)), DEFAULT_TASK
            )

            call = EnvToolCall(tool="press_button", args={"button": action, "frames": 1})
            await test_env.step(call)

            # Add child
            child_blob = gzip.compress(pickle.dumps(await test_env._serialize_engine()))
            child_id = tree.add_child(
                root_id,
                child_blob,
                action=action,
                reward=heuristic_score(test_env),
                terminated=is_terminal_state(test_env),
                info={},
            )

            print(f"Child ID: {child_id[:8]}...")
            print(f"Tree has {len(tree.get_children(root_id))} children")

            # Test rollout from child
            print("\nTesting rollout from child...")
            child_env = await PokemonRedEnvironment._deserialize_engine(
                pickle.loads(gzip.decompress(child_blob)), DEFAULT_TASK
            )

            from synth_ai.environments.examples.red.units.test_tree import simple_rollout

            rollout_score = await simple_rollout(child_env, max_steps=5)
            print(f"Rollout score: {rollout_score}")

        except Exception as e:
            print(f"Tree operation failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n=== Debug Complete ===")


if __name__ == "__main__":
    asyncio.run(debug_pokemon_mcts())
