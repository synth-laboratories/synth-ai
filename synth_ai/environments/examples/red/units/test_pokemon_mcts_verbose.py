#!/usr/bin/env python3
"""Verbose Pokemon Red MCTS test to see detailed operation"""

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

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")


async def verbose_mcts_test():
    """Run MCTS with verbose output"""

    print("🎮 Pokemon Red MCTS - Verbose Test")
    print("=" * 50)

    # Create environment
    env = PokemonRedEnvironment(DEFAULT_TASK)
    await env.initialize()

    # Check initial state
    priv, pub = env.engine._create_states(reward=0.0)
    print("Initial State:")
    print(f"  Map: {pub.map_id:02X}, Position: ({pub.player_x},{pub.player_y})")
    print(f"  Badges: {bin(pub.badges).count('1')}, Level: {pub.party_level}")
    print(f"  HP: {pub.party_hp_current}/{pub.party_hp_max}")
    print(f"  Steps: {pub.step_count}")

    # Set up MCTS
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_store_path = Path(tmpdir) / "verbose_mcts"
        tree = TrajectoryTreeStore(FilesystemSnapshotStore(snap_store_path))

        root_blob = gzip.compress(pickle.dumps(await env._serialize_engine()))
        root_id = tree.add_root(root_blob)

        print(f"\n🌳 MCTS Tree initialized, root: {root_id[:8]}...")

        # Run MCTS with detailed settings
        from synth_ai.environments.examples.red.units.test_tree import pokemon_red_mcts_plan

        plan, q_hist = await pokemon_red_mcts_plan(
            tree,
            root_id,
            rollouts_per_action=5,  # More rollouts
            max_depth=8,  # Deeper search
            timeout_s=20.0,  # Longer timeout
        )

        print("\n📋 MCTS Results:")
        print(f"Plan length: {len(plan)}")
        print(f"Action sequence: {plan}")
        print(f"Q-value history length: {len(q_hist)}")

        for i, q_dict in enumerate(q_hist):
            print(f"\nDepth {i} Q-values:")
            sorted_actions = sorted(q_dict.items(), key=lambda x: x[1], reverse=True)
            for action, q_val in sorted_actions:
                print(f"  {action}: {q_val:.4f}")

        print("\n🎯 Tree Statistics:")
        print(f"Root children: {len(tree.get_children(root_id))}")

        total_nodes = 1  # Root
        for child_id in tree.get_children(root_id):
            total_nodes += 1 + len(tree.get_children(child_id))
        print(f"Total nodes: {total_nodes}")

        # Execute the plan and see what happens
        print("\n🎮 Executing Plan:")
        from synth_ai.environments.environment.tools import EnvToolCall

        for i, action in enumerate(plan):
            print(f"\nStep {i + 1}: {action}")

            call = EnvToolCall(tool="press_button", args={"button": action, "frames": 1})
            obs = await env.step(call)

            new_priv, new_pub = env.engine._create_states(reward=0.0)

            print(f"  Map: {pub.map_id:02X} → {new_pub.map_id:02X}")
            print(
                f"  Pos: ({pub.player_x},{pub.player_y}) → ({new_pub.player_x},{new_pub.player_y})"
            )
            print(f"  Level: {pub.party_level} → {new_pub.party_level}")
            print(f"  Badges: {bin(pub.badges).count('1')} → {bin(new_pub.badges).count('1')}")
            print(f"  Reward: {obs.get('reward_last_step', 'N/A')}")
            print(f"  Total Reward: {obs.get('total_reward', 'N/A')}")

            # Update for next iteration
            pub = new_pub

        # Final assessment
        from synth_ai.environments.examples.red.units.test_tree import heuristic_score

        final_score = heuristic_score(env)
        print("\n📊 Final Assessment:")
        print(f"Final heuristic score: {final_score:.3f}")
        print(f"Total steps taken: {pub.step_count}")

        print("\n✅ MCTS Test Complete!")


if __name__ == "__main__":
    asyncio.run(verbose_mcts_test())
