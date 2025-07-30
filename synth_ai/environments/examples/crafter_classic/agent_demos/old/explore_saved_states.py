#!/usr/bin/env python3
"""
Explore and visualize saved Crafter states
==========================================
"""

import gzip
import pickle
import json
from pathlib import Path
from datetime import datetime
import asyncio
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.tasks.core import Impetus, Intent

STATES_DIR = Path("synth_ai/environments/examples/crafter_classic/env_states")


def list_sessions():
    """List all saved MCTS sessions."""
    if not STATES_DIR.exists():
        print("No env_states directory found!")
        return []
    
    sessions = []
    for session_dir in sorted(STATES_DIR.iterdir()):
        if session_dir.is_dir() and session_dir.name.startswith("mcts_"):
            metadata_file = session_dir / "session_metadata.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text())
                sessions.append((session_dir, metadata))
    
    return sessions


async def explore_state_file(state_file: Path):
    """Load and display info about a single state file."""
    print(f"\n📄 State file: {state_file.name}")
    print(f"   Size: {state_file.stat().st_size / 1024:.1f} KB")
    
    # Load the state
    try:
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
        priv = env.engine._get_private_state_from_env(0, False, False)
        
        print(f"   Position: {pub.player_position}")
        print(f"   Health: {priv.player_internal_stats.get('health', 0)}")
        print(f"   Steps: {pub.num_steps_taken}")
        
        # Show achievements
        achievements = [k for k, v in pub.achievements_status.items() if v]
        if achievements:
            print(f"   Achievements ({len(achievements)}): {', '.join(achievements[:3])}{'...' if len(achievements) > 3 else ''}")
        
        # Show inventory
        inventory = [(k, v) for k, v in pub.inventory.items() 
                    if v > 0 and k not in ['health', 'food', 'drink', 'energy']]
        if inventory:
            print(f"   Inventory: {dict(inventory)}")
            
    except Exception as e:
        print(f"   ❌ Error loading state: {e}")


async def main():
    print("🔍 Exploring Saved Crafter States")
    print("=" * 60)
    
    sessions = list_sessions()
    
    if not sessions:
        print("No saved sessions found!")
        return
    
    print(f"Found {len(sessions)} saved sessions:\n")
    
    for i, (session_dir, metadata) in enumerate(sessions):
        print(f"{i+1}. {session_dir.name}")
        print(f"   Seed: {metadata['seed']}")
        print(f"   Time: {metadata['timestamp']}")
        print(f"   States: {len(list(session_dir.glob('*.snapshot.gz')))}")
    
    # Explore the most recent session
    if sessions:
        latest_session, latest_metadata = sessions[-1]
        print(f"\n📂 Exploring latest session: {latest_session.name}")
        print("-" * 60)
        
        # Get all state files
        state_files = sorted(latest_session.glob("*.snapshot.gz"))
        print(f"Total states saved: {len(state_files)}")
        
        # Show the root state
        root_id = latest_metadata['root_node_id']
        root_file = latest_session / f"{root_id}.snapshot.gz"
        if root_file.exists():
            print("\n🌱 Root state:")
            await explore_state_file(root_file)
        
        # Show a few random states
        import random
        sample_size = min(3, len(state_files) - 1)
        if sample_size > 0:
            print("\n🎲 Random sample states:")
            for state_file in random.sample(state_files[1:], sample_size):
                await explore_state_file(state_file)
        
        # Show summary if it exists
        summary_file = latest_session / "summary.txt"
        if summary_file.exists():
            print("\n📊 Session Summary:")
            print("-" * 40)
            print(summary_file.read_text())
    
    print("\n✅ Done exploring saved states!")
    print(f"\n💡 Tip: States are saved in: {STATES_DIR}")
    print("   Each .snapshot.gz file contains a complete game state")
    print("   You can analyze these to understand MCTS exploration patterns")


if __name__ == "__main__":
    asyncio.run(main())