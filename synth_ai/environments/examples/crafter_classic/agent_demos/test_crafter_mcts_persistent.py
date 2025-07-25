#!/usr/bin/env python3
"""
Random MCTS for Crafter with Persistent State Storage
=====================================================
Saves all explored states to env_states directory for analysis.
"""

import asyncio
import gzip
import pickle
import random
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from uuid import uuid4
from datetime import datetime

from synth_ai.environments.reproducibility.tree import FilesystemSnapshotStore, TrajectoryTreeStore
from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("persistent-mcts")

# Crafter action space
CRAFTER_ACTIONS = list(range(17))
ACTION_NAMES = [
    "noop", "left", "right", "up", "down", "do",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword", "rest"
]

# Persistent storage path
STATES_DIR = Path("synth_ai/environments/examples/crafter_classic/env_states")


async def crafter_mcts_with_persistent_storage(
    seed: int = 42,
    max_depth: int = 20,
    rollouts_per_action: int = 5,
    timeout_s: float = 60.0,
):
    """Run MCTS and save all states to persistent storage."""
    
    # Create task
    metadata = CrafterTaskInstanceMetadata(
        difficulty="easy",
        seed=seed,
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
    
    # Create persistent storage directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = STATES_DIR / f"mcts_seed{seed}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    LOG.info(f"🎮 Starting MCTS with Persistent Storage")
    LOG.info(f"Seed: {seed}")
    LOG.info(f"📁 Saving states to: {session_dir}")
    LOG.info("=" * 60)
    
    # Set up tree storage in persistent directory
    tree = TrajectoryTreeStore(FilesystemSnapshotStore(session_dir))
    
    # Add root snapshot
    root_snapshot = await env._serialize_engine()
    root_blob = gzip.compress(pickle.dumps(root_snapshot))
    root_id = tree.add_root(root_blob)
    
    # Save metadata about the session
    metadata_file = session_dir / "session_metadata.json"
    import json
    metadata_file.write_text(json.dumps({
        "seed": seed,
        "timestamp": timestamp,
        "root_node_id": root_id,
        "task_id": str(task.id),
        "max_depth": max_depth,
        "rollouts_per_action": rollouts_per_action,
    }, indent=2))
    
    LOG.info(f"Root node saved: {root_id[:8]}...")
    
    # Track statistics
    nodes_created = 1
    achievements_found = {}
    state_sizes = []
    
    # Simple MCTS loop
    start_time = time.monotonic()
    node_id = root_id
    plan = []
    
    for depth in range(max_depth):
        if time.monotonic() - start_time > timeout_s:
            LOG.info("Timeout reached")
            break
            
        LOG.info(f"\n--- Depth {depth} --- Node: {node_id[:8]}")
        
        # Load current state
        env_blob = tree.load_snapshot_blob(node_id)
        state_sizes.append(len(env_blob))
        env_snapshot = pickle.loads(gzip.decompress(env_blob))
        env = await CrafterClassicEnvironment._deserialize_engine(env_snapshot, task)
        
        # Check current state
        pub = env.engine._get_public_state_from_env()
        priv = env.engine._get_private_state_from_env(0, False, False)
        
        # Track new achievements
        for ach, status in pub.achievements_status.items():
            if status and ach not in achievements_found:
                achievements_found[ach] = depth
                LOG.info(f"🎯 NEW ACHIEVEMENT: {ach}")
        
        LOG.info(f"Pos: {pub.player_position}, Health: {priv.player_internal_stats.get('health', 0)}")
        LOG.info(f"Inventory: {[(k,v) for k,v in pub.inventory.items() if v > 0 and k not in ['health','food','drink','energy']]}")
        
        # Try a few random actions and pick the best
        best_action = None
        best_score = -float('inf')
        
        for _ in range(min(5, len(CRAFTER_ACTIONS))):  # Sample 5 random actions
            action = random.choice(CRAFTER_ACTIONS)
            
            # Skip if we already have this child
            existing_child = next(
                (cid for cid in tree.get_children(node_id) 
                 if tree.graph[node_id][cid]["action"] == action),
                None
            )
            if existing_child:
                continue
            
            try:
                # Try the action
                test_env = await CrafterClassicEnvironment._deserialize_engine(env_snapshot, task)
                call = EnvToolCall(tool="interact", args={"action": action})
                obs = await test_env.step(call)
                
                # Simple heuristic score
                test_pub = test_env.engine._get_public_state_from_env()
                score = sum(1 for v in test_pub.achievements_status.values() if v)
                score += len([v for k,v in test_pub.inventory.items() if v > 0]) * 0.1
                
                # Save this state
                child_snapshot = await test_env._serialize_engine()
                child_blob = gzip.compress(pickle.dumps(child_snapshot))
                child_id = tree.add_child(
                    node_id,
                    child_blob,
                    action=action,
                    reward=obs.get("reward_last_step", 0.0),
                    terminated=obs.get("terminated", False),
                    info={"score": score, "depth": depth + 1}
                )
                nodes_created += 1
                
                LOG.debug(f"  Action {action} ({ACTION_NAMES[action]}): score={score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child_id = child_id
                    
            except Exception as e:
                LOG.debug(f"  Action {action} failed: {e}")
        
        if best_action is None:
            LOG.info("No valid actions found")
            break
        
        # Move to best child
        plan.append(best_action)
        node_id = best_child_id
        LOG.info(f"Selected: {ACTION_NAMES[best_action]} (score: {best_score:.2f})")
    
    # Save summary
    summary_file = session_dir / "summary.txt"
    summary = f"""MCTS Session Summary
====================
Seed: {seed}
Timestamp: {timestamp}
Directory: {session_dir}

Tree Statistics:
- Total nodes created: {nodes_created}
- Max depth reached: {len(plan)}
- Average state size: {sum(state_sizes)/len(state_sizes):.0f} bytes (compressed)
- Total storage used: {sum(state_sizes)/1024/1024:.2f} MB

Achievements Found: {len(achievements_found)}
{chr(10).join(f'  - {ach} (depth {d})' for ach, d in sorted(achievements_found.items(), key=lambda x: x[1]))}

Action Plan ({len(plan)} actions):
{' -> '.join(ACTION_NAMES[a] for a in plan[:10])}{'...' if len(plan) > 10 else ''}
"""
    summary_file.write_text(summary)
    
    LOG.info("\n" + "=" * 60)
    LOG.info(summary)
    LOG.info(f"\n📁 All states saved to: {session_dir}")
    LOG.info(f"   Total files: {len(list(session_dir.glob('*.pkl.gz')))}")
    
    return session_dir, achievements_found


async def explore_saved_states(session_dir: Path):
    """Load and analyze previously saved states."""
    
    LOG.info(f"\n📂 Exploring saved states in: {session_dir}")
    
    # Load metadata
    metadata_file = session_dir / "session_metadata.json"
    if metadata_file.exists():
        import json
        metadata = json.loads(metadata_file.read_text())
        LOG.info(f"Session seed: {metadata['seed']}")
        LOG.info(f"Session time: {metadata['timestamp']}")
    
    # Count state files
    state_files = list(session_dir.glob("*.pkl.gz"))
    LOG.info(f"Found {len(state_files)} state files")
    
    # Load and analyze a few states
    for i, state_file in enumerate(state_files[:3]):
        LOG.info(f"\n--- State {i+1}: {state_file.name} ---")
        LOG.info(f"Size: {state_file.stat().st_size / 1024:.1f} KB")
        
        # Load the state
        env_blob = gzip.decompress(state_file.read_bytes())
        env_snapshot = pickle.loads(env_blob)
        
        # Create dummy task for deserialization
        task = CrafterTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Analysis"),
            intent=Intent(rubric={"goal": "Analysis"}, gold_trajectories=None, gold_state_diff={}),
            metadata=CrafterTaskInstanceMetadata(
                difficulty="easy", seed=0, num_trees_radius=0, 
                num_cows_radius=0, num_hostiles_radius=0
            ),
            is_reproducible=True,
            initial_engine_snapshot=None
        )
        
        env = await CrafterClassicEnvironment._deserialize_engine(env_snapshot, task)
        pub = env.engine._get_public_state_from_env()
        
        LOG.info(f"Position: {pub.player_position}")
        LOG.info(f"Achievements: {sum(1 for v in pub.achievements_status.values() if v)}")
        LOG.info(f"Inventory: {[(k,v) for k,v in pub.inventory.items() if v > 0 and k not in ['health','food','drink','energy']]}")


async def cleanup_old_sessions(keep_last_n: int = 5):
    """Clean up old session directories, keeping only the most recent ones."""
    
    if not STATES_DIR.exists():
        return
    
    # Find all session directories
    session_dirs = sorted(
        [d for d in STATES_DIR.iterdir() if d.is_dir() and d.name.startswith("mcts_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    
    if len(session_dirs) <= keep_last_n:
        LOG.info(f"Only {len(session_dirs)} sessions found, keeping all")
        return
    
    # Remove old sessions
    for old_dir in session_dirs[keep_last_n:]:
        LOG.info(f"Removing old session: {old_dir.name}")
        shutil.rmtree(old_dir)
    
    LOG.info(f"Kept {keep_last_n} most recent sessions")


if __name__ == "__main__":
    async def main():
        # Run MCTS with persistent storage
        seed = random.randint(0, 10000)
        session_dir, achievements = await crafter_mcts_with_persistent_storage(
            seed=seed,
            max_depth=30,
            rollouts_per_action=3,
            timeout_s=60.0
        )
        
        # Explore the saved states
        await explore_saved_states(session_dir)
        
        # Optional: Clean up old sessions
        # await cleanup_old_sessions(keep_last_n=5)
        
        LOG.info("\n✅ Done! Check the env_states directory for saved states.")
    
    asyncio.run(main())