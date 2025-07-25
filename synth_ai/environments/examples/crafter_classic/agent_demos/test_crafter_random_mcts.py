#!/usr/bin/env python3
"""
Random-policy MCTS for Crafter
==============================
Pure random rollouts to see if MCTS can stumble into achievements.
No LLM calls, just random exploration with tree search.
"""

import asyncio
import gzip
import pickle
import random
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from uuid import uuid4

from synth_ai.environments.reproducibility.tree import FilesystemSnapshotStore, TrajectoryTreeStore
from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("crafter-mcts")

# Crafter action space (17 actions)
CRAFTER_ACTIONS = list(range(17))
ACTION_NAMES = [
    "noop", "left", "right", "up", "down", "do",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword", "rest"
]


def heuristic_score(env: CrafterClassicEnvironment) -> float:
    """
    Heuristic evaluation of Crafter state.
    Higher scores = better states.
    """
    try:
        pub = env.engine._get_public_state_from_env()
        priv = env.engine._get_private_state_from_env(0, False, False)
        
        score = 10.0  # Base score
        
        # Achievement bonus (primary objective)
        achievements_unlocked = sum(1 for v in pub.achievements_status.values() if v)
        score += achievements_unlocked * 100.0
        
        # Count specific achievements for bonus
        if pub.achievements_status.get("collect_wood", False):
            score += 20.0
        if pub.achievements_status.get("defeat_zombie", False):
            score += 50.0
        if pub.achievements_status.get("make_wood_pickaxe", False):
            score += 30.0
        
        # Survival metrics
        health = priv.player_internal_stats.get("health", 0)
        hunger = priv.player_internal_stats.get("_hunger", 0)
        thirst = priv.player_internal_stats.get("_thirst", 0)
        
        score += health * 5.0
        score += (9 - hunger) * 2.0  # Lower hunger is better
        score += (9 - thirst) * 2.0  # Lower thirst is better
        
        # Inventory value
        for item, count in pub.inventory.items():
            if "pickaxe" in item:
                score += count * 20.0
            elif "sword" in item:
                score += count * 15.0
            elif item == "wood":
                score += count * 5.0
            elif item == "stone":
                score += count * 3.0
            else:
                score += count * 1.0
        
        # Small penalty for time
        score -= pub.num_steps_taken * 0.01
        
        return max(score, 0.1)
    
    except Exception as e:
        LOG.debug(f"Heuristic error: {e}")
        return 0.1


def is_terminal_state(env: CrafterClassicEnvironment) -> bool:
    """Check if we've reached a terminal state."""
    try:
        priv = env.engine._get_private_state_from_env(0, False, False)
        
        # Terminal if dead
        if priv.player_internal_stats.get("health", 0) <= 0:
            return True
        
        # Terminal if episode ended
        if priv.terminated or priv.truncated:
            return True
        
        return False
    
    except Exception:
        return True


async def random_rollout(env: CrafterClassicEnvironment, max_steps: int = 50) -> float:
    """
    Perform a random rollout from current state.
    Returns heuristic score after rollout.
    """
    try:
        # Save current state
        snapshot = await env._serialize_engine()
        
        # Random walk
        for _ in range(max_steps):
            if is_terminal_state(env):
                break
            
            # Choose random action (with slight bias)
            if random.random() < 0.3:  # 30% chance to do useful actions
                # Bias towards movement and 'do' action
                action = random.choice([0, 1, 2, 3, 4, 5, 5, 5])  # Extra weight on 'do'
            else:
                # Fully random
                action = random.choice(CRAFTER_ACTIONS)
            
            call = EnvToolCall(tool="interact", args={"action": action})
            
            try:
                await env.step(call)
            except Exception:
                break
        
        # Evaluate final state
        final_score = heuristic_score(env)
        
        # Restore original state by deserializing entire environment
        restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, env.task_instance)
        # Copy the restored engine back
        env.engine = restored_env.engine
        
        return final_score
    
    except Exception as e:
        LOG.debug(f"Rollout error: {e}")
        return 0.0


async def crafter_mcts_plan(
    tree: TrajectoryTreeStore,
    root_id: str,
    task: CrafterTaskInstance,
    *,
    rollouts_per_action: int = 5,
    max_depth: int = 20,
    timeout_s: float = 60.0,
    exploration_bonus: float = 1.0,
) -> Tuple[List[int], List[Dict[int, float]], Dict[str, int]]:
    """
    MCTS planning for Crafter with random rollouts.
    Returns (action_plan, q_value_history, achievements_found)
    """
    start = time.monotonic()
    plan = []
    q_hist = []
    node_id = root_id
    achievements_found = {}
    
    for depth in range(max_depth):
        LOG.info(f"\n--- MCTS depth {depth} --- node={node_id[:8]}")
        
        if timeout_s is not None and time.monotonic() - start >= timeout_s:
            LOG.info("MCTS timeout reached")
            break
        
        # Load environment from snapshot
        env_blob = tree.load_snapshot_blob(node_id)
        env_snapshot = pickle.loads(gzip.decompress(env_blob))
        env = await CrafterClassicEnvironment._deserialize_engine(env_snapshot, task)
        
        # Check if terminal
        if is_terminal_state(env):
            LOG.info("Terminal state reached")
            break
        
        # Log current state
        pub = env.engine._get_public_state_from_env()
        priv = env.engine._get_private_state_from_env(0, False, False)
        
        # Track achievements
        for ach, status in pub.achievements_status.items():
            if status and ach not in achievements_found:
                achievements_found[ach] = depth
                LOG.info(f"🎯 NEW ACHIEVEMENT: {ach} at depth {depth}!")
        
        LOG.info(
            f"State: Pos{pub.player_position} "
            f"Health:{priv.player_internal_stats.get('health', 0)} "
            f"Achievements:{sum(1 for v in pub.achievements_status.values() if v)} "
            f"Inventory:{dict(pub.inventory)}"
        )
        
        q_vals: Dict[int, float] = {}
        visit_counts: Dict[int, int] = {}
        
        # Evaluate each possible action
        for action in CRAFTER_ACTIONS:
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
                LOG.debug(f"Expanding action: {action} ({ACTION_NAMES[action]})")
                
                # Create new environment and take action
                try:
                    tmp_env = await CrafterClassicEnvironment._deserialize_engine(env_snapshot, task)
                    
                    call = EnvToolCall(tool="interact", args={"action": action})
                    obs = await tmp_env.step(call)
                    
                    # Create child node
                    child_blob = gzip.compress(pickle.dumps(await tmp_env._serialize_engine()))
                    child_id = tree.add_child(
                        node_id,
                        child_blob,
                        action=action,
                        reward=obs.get("reward_last_step", 0.0),
                        terminated=is_terminal_state(tmp_env),
                        info={"heuristic": heuristic_score(tmp_env)},
                    )
                    
                except Exception as e:
                    LOG.debug(f"Failed to expand action {action}: {e}")
                    continue
            else:
                LOG.debug(f"Reusing existing child for action: {action} ({ACTION_NAMES[action]})")
            
            if child_id is None:
                continue
            
            # Perform rollouts from child state
            child_env_snapshot = pickle.loads(gzip.decompress(tree.load_snapshot_blob(child_id)))
            child_env = await CrafterClassicEnvironment._deserialize_engine(child_env_snapshot, task)
            
            rollout_scores = []
            for i in range(rollouts_per_action):
                if timeout_s is not None and time.monotonic() - start >= timeout_s:
                    break
                score = await random_rollout(child_env, max_steps=30)
                rollout_scores.append(score)
            
            if rollout_scores:
                # Average rollout score as Q-value
                q_vals[action] = sum(rollout_scores) / len(rollout_scores)
                visit_counts[action] = len(rollout_scores)
                LOG.debug(
                    f"Action {action} ({ACTION_NAMES[action]}): "
                    f"Q={q_vals[action]:.3f} "
                    f"(avg of {len(rollout_scores)} rollouts)"
                )
            else:
                q_vals[action] = 0.0
                visit_counts[action] = 0
        
        if not q_vals:
            LOG.info("No valid actions found")
            break
        
        LOG.info(f"Q-values: {[(ACTION_NAMES[a], f'{q:.1f}') for a, q in sorted(q_vals.items(), key=lambda x: x[1], reverse=True)[:5]]}")
        q_hist.append(q_vals)
        
        # Select action with UCB1
        total_visits = sum(visit_counts.values())
        best_action = None
        best_ucb = -float('inf')
        
        for action, q_val in q_vals.items():
            visits = visit_counts.get(action, 1)
            # UCB1 formula: Q + c * sqrt(ln(N) / n)
            if total_visits > 0 and visits > 0:
                ucb = q_val + exploration_bonus * (2 * (total_visits / visits) ** 0.5)
            else:
                ucb = q_val + exploration_bonus * 100  # High bonus for unvisited
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
        
        if best_action is None:
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
            LOG.info(f"No child node found for action {best_action}")
            break
        
        node_id = next_node
        
        LOG.info(f"Selected action: {best_action} ({ACTION_NAMES[best_action]}) → node={node_id[:8]}")
    
    return plan, q_hist, achievements_found


async def test_random_mcts_crafter():
    """Test if random MCTS can discover achievements in Crafter."""
    
    # Create task
    metadata = CrafterTaskInstanceMetadata(
        difficulty="easy",
        seed=random.randint(0, 10000),
        num_trees_radius=5,
        num_cows_radius=2,
        num_hostiles_radius=0
    )
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Discover achievements with MCTS"),
        intent=Intent(rubric={"goal": "Unlock achievements"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    
    # Create environment
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Get initial state
    initial_pub = env.engine._get_public_state_from_env()
    initial_achievements = sum(1 for v in initial_pub.achievements_status.values() if v)
    
    LOG.info("🎮 Starting Random MCTS for Crafter")
    LOG.info(f"Seed: {metadata.seed}")
    LOG.info(f"Initial achievements: {initial_achievements}")
    LOG.info("=" * 60)
    
    # Set up tree storage
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_store_path = Path(tmpdir) / "crafter_mcts"
        tree = TrajectoryTreeStore(FilesystemSnapshotStore(snap_store_path))
        
        # Add root snapshot
        root_snapshot = await env._serialize_engine()
        root_blob = gzip.compress(pickle.dumps(root_snapshot))
        root_id = tree.add_root(root_blob)
        
        # Run MCTS
        plan, q_hist, achievements_found = await crafter_mcts_plan(
            tree,
            root_id,
            task,
            rollouts_per_action=10,
            max_depth=30,
            timeout_s=120.0,
            exploration_bonus=2.0,  # Higher exploration
        )
        
        LOG.info("\n" + "=" * 60)
        LOG.info("📊 MCTS Results:")
        LOG.info(f"Plan length: {len(plan)}")
        LOG.info(f"Actions: {[ACTION_NAMES[a] for a in plan[:10]]}...")
        
        if achievements_found:
            LOG.info(f"\n🎯 Achievements discovered: {len(achievements_found)}")
            for ach, depth in sorted(achievements_found.items(), key=lambda x: x[1]):
                LOG.info(f"  - {ach} (depth {depth})")
        else:
            LOG.info("\n❌ No achievements discovered")
        
        # Execute the plan to verify
        LOG.info("\n🎮 Executing plan...")
        for i, action in enumerate(plan[:20]):  # Execute first 20 actions
            call = EnvToolCall(tool="interact", args={"action": action})
            obs = await env.step(call)
            
            if i % 5 == 0:
                pub = env.engine._get_public_state_from_env()
                achievements_now = sum(1 for v in pub.achievements_status.values() if v)
                LOG.info(f"Step {i}: {ACTION_NAMES[action]}, Achievements: {achievements_now}")
        
        # Final state
        final_pub = env.engine._get_public_state_from_env()
        final_achievements = sum(1 for v in final_pub.achievements_status.values() if v)
        
        LOG.info("\n" + "=" * 60)
        LOG.info("📈 Summary:")
        LOG.info(f"Initial achievements: {initial_achievements}")
        LOG.info(f"Final achievements: {final_achievements}")
        LOG.info(f"Achievements gained: {final_achievements - initial_achievements}")
        
        # Show which achievements were unlocked
        for ach, status in final_pub.achievements_status.items():
            if status:
                LOG.info(f"  ✓ {ach}")
        
        # Tree statistics
        LOG.info(f"\n🌳 Tree statistics:")
        LOG.info(f"Root children: {len(tree.get_children(root_id))}")
        
        total_nodes = 1
        to_visit = list(tree.get_children(root_id))
        while to_visit:
            node = to_visit.pop()
            total_nodes += 1
            to_visit.extend(tree.get_children(node))
        
        LOG.info(f"Total nodes explored: {total_nodes}")
        
        LOG.info("\n✅ Random MCTS test complete!")
        
        return final_achievements > initial_achievements


if __name__ == "__main__":
    # Run multiple trials
    successes = 0
    trials = 3
    
    print("Running Random MCTS trials for Crafter...")
    print("Can random exploration discover achievements?")
    print("=" * 60)
    
    for i in range(trials):
        print(f"\nTrial {i+1}/{trials}:")
        success = asyncio.run(test_random_mcts_crafter())
        if success:
            successes += 1
            print("✅ Success! Discovered new achievements.")
        else:
            print("❌ No new achievements discovered.")
    
    print("\n" + "=" * 60)
    print(f"Success rate: {successes}/{trials} ({100*successes/trials:.0f}%)")
    print("🎉 Random MCTS experiment complete!")