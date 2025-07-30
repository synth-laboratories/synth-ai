"""
MCTS Implementation Guide for Crafter Environment
================================================

Based on analysis of Pokemon RED and Sokoban MCTS implementations, here's how we could 
implement Monte Carlo Tree Search for the Crafter environment:

1. **State Serialization & Tree Storage**
   - Use FilesystemSnapshotStore and TrajectoryTreeStore (like RED)
   - Serialize Crafter environment state using pickle + gzip compression
   - Each tree node stores a complete environment snapshot
   - Child nodes are created by taking actions from parent states

2. **Action Space**
   - Crafter has 17 discrete actions: noop, move(4), do, place_stone, place_table, 
     place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, 
     make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword
   - Much larger than Pokemon's 8 buttons, so may need action filtering

3. **Heuristic Evaluation Function**
   Key metrics to consider for Crafter:
   - Achievements unlocked (primary objective)
   - Health, hunger, thirst levels (survival)
   - Inventory contents (resources, tools, weapons)
   - Distance to nearest resources (exploration bonus)
   - Day/night cycle position
   - Proximity to dangers (zombies, skeletons)
   
   Example scoring:
   ```python
   score = 0.0
   score += len(achievements) * 100  # Major reward for achievements
   score += health * 5
   score += (9 - hunger) * 2  # Lower hunger is better
   score += (9 - thirst) * 2  # Lower thirst is better
   score += inventory_value()  # Value of items
   score -= day_count * 0.1   # Slight penalty for time
   ```

4. **Terminal State Detection**
   - All achievements unlocked (win)
   - Player health reaches 0 (loss)
   - Maximum steps/days reached (timeout)

5. **Rollout Policy**
   - Random actions work poorly in Crafter due to survival mechanics
   - Consider biased rollouts:
     * Prioritize "do" action near resources
     * Avoid moving into danger zones
     * Seek food/water when low
   - Alternatively, use a simple policy network

6. **MCTS Algorithm Structure**
   ```python
   async def crafter_mcts_plan(tree, root_id, rollouts_per_action=10, max_depth=20):
       plan = []
       node_id = root_id
       
       for depth in range(max_depth):
           # Load environment from node
           env = deserialize_crafter_env(tree.load_snapshot_blob(node_id))
           
           if is_terminal(env):
               break
               
           # Evaluate each action
           q_values = {}
           for action in CRAFTER_ACTIONS:
               # Expand node if needed
               child_id = expand_if_needed(tree, node_id, action, env)
               
               # Run rollouts from child state
               scores = []
               for _ in range(rollouts_per_action):
                   score = await rollout(child_env, max_steps=50)
                   scores.append(score)
               
               q_values[action] = np.mean(scores)
           
           # Select best action
           best_action = max(q_values, key=q_values.get)
           plan.append(best_action)
           node_id = get_child_for_action(tree, node_id, best_action)
       
       return plan
   ```

7. **Optimizations for Crafter**
   - **Action Filtering**: Not all 17 actions are valid at every state
     * Can't craft without resources
     * Can't place objects without them in inventory
     * Filter invalid actions before expansion
   
   - **Progressive Widening**: Start with core actions (move, do), 
     gradually add crafting actions as resources are collected
   
   - **Domain Knowledge Integration**:
     * Prioritize water/food collection when low
     * Seek shelter at night
     * Maintain tool progression (wood � stone � iron)

8. **Challenges Specific to Crafter**
   - Long-horizon planning required (achievements need multi-step sequences)
   - Day/night cycle adds urgency 
   - Resource management is critical
   - Combat situations need quick responses
   - Much richer state space than Pokemon RED movement

9. **Implementation Steps**
   1. Create serialization methods for CrafterEnvironment
   2. Implement heuristic scoring based on game state
   3. Build MCTS planner with Crafter-specific optimizations
   4. Add action filtering and validity checking
   5. Test on simple achievement sequences first
   6. Gradually increase complexity to full achievement set

The key insight from RED's implementation is that MCTS works well for exploration
and planning in environments with discrete actions and clear progress metrics.
Crafter's achievement system provides natural waypoints for the search tree.
"""

# ============================================================================
# CRAFTER ENVIRONMENT SERIALIZATION TESTS
# ============================================================================

import asyncio
import gzip
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.reproducibility.tree import FilesystemSnapshotStore, TrajectoryTreeStore


@pytest.mark.asyncio
async def test_basic_crafter_serialization():
    """Test basic serialization/deserialization of Crafter environment."""
    # Create task instance
    from synth_ai.environments.tasks.core import Impetus, Intent
    from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstanceMetadata
    from uuid import uuid4
    
    metadata = CrafterTaskInstanceMetadata(
        difficulty="easy", 
        seed=42,
        num_trees_radius=5,
        num_cows_radius=2,
        num_hostiles_radius=0
    )
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test serialization"),
        intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    
    # Create and initialize environment
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Get initial state
    initial_snapshot = await env._serialize_engine()
    initial_pub = env.engine._get_public_state_from_env()
    
    # Take a few actions
    actions = [
        EnvToolCall(tool="interact", args={"action": 5}),  # do
        EnvToolCall(tool="interact", args={"action": 2}),  # right
        EnvToolCall(tool="interact", args={"action": 5}),  # do
    ]
    
    for action in actions:
        await env.step(action)
    
    # Get state after actions
    after_pub = env.engine._get_public_state_from_env()
    after_snapshot = await env._serialize_engine()
    
    # Verify state changed
    assert initial_pub.player_position != after_pub.player_position or initial_pub.inventory != after_pub.inventory
    assert initial_snapshot.total_reward_snapshot != after_snapshot.total_reward_snapshot
    
    # Deserialize from initial snapshot
    restored_env = await CrafterClassicEnvironment._deserialize_engine(initial_snapshot, task)
    restored_pub = restored_env.engine._get_public_state_from_env()
    
    # Verify restoration
    assert restored_pub.player_position == initial_pub.player_position
    assert restored_pub.inventory == initial_pub.inventory
    assert restored_pub.achievements_status == initial_pub.achievements_status
    
    print("✓ Basic serialization test passed")


@pytest.mark.asyncio
async def test_crafter_tree_storage():
    """Test storing Crafter states in a trajectory tree (for MCTS)."""
    from synth_ai.environments.tasks.core import Impetus, Intent
    from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstanceMetadata
    from uuid import uuid4
    
    metadata = CrafterTaskInstanceMetadata(
        difficulty="medium", 
        seed=123,
        num_trees_radius=3,
        num_cows_radius=1,
        num_hostiles_radius=1
    )
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test tree storage"),
        intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Set up tree storage
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_store_path = Path(tmpdir) / "crafter_tree"
        tree = TrajectoryTreeStore(FilesystemSnapshotStore(snap_store_path))
        
        # Add root snapshot
        root_snapshot = await env._serialize_engine()
        root_blob = gzip.compress(pickle.dumps(root_snapshot))
        root_id = tree.add_root(root_blob)
        
        # Expand tree with different actions
        action_rewards = {}
        for action_idx in [0, 1, 2, 3, 4, 5]:  # noop, move directions, do
            # Restore from root
            root_env_snapshot = pickle.loads(gzip.decompress(tree.load_snapshot_blob(root_id)))
            env = await CrafterClassicEnvironment._deserialize_engine(root_env_snapshot, task)
            
            # Take action
            call = EnvToolCall(tool="interact", args={"action": action_idx})
            obs = await env.step(call)
            
            # Store child
            child_snapshot = await env._serialize_engine()
            child_blob = gzip.compress(pickle.dumps(child_snapshot))
            child_id = tree.add_child(
                root_id,
                child_blob,
                action=action_idx,
                reward=obs.get("reward_last_step", 0.0),
                terminated=obs.get("terminated", False),
                info={"total_reward": obs.get("total_reward", 0.0)}
            )
            
            action_rewards[action_idx] = obs.get("reward_last_step", 0.0)
        
        # Verify tree structure
        children = tree.get_children(root_id)
        assert len(children) == 6
        
        # Verify we can load any child state
        for child_id in children:
            child_snapshot = pickle.loads(gzip.decompress(tree.load_snapshot_blob(child_id)))
            child_env = await CrafterClassicEnvironment._deserialize_engine(child_snapshot, task)
            # Should be able to continue from this state
            await child_env.step(EnvToolCall(tool="interact", args={"action": 0}))
        
        print(f"✓ Tree storage test passed - stored {len(children)} child states")
        print(f"  Action rewards: {action_rewards}")


@pytest.mark.asyncio
async def test_crafter_state_consistency():
    """Test that serialization preserves all important state components."""
    from synth_ai.environments.tasks.core import Impetus, Intent
    from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstanceMetadata
    from uuid import uuid4
    
    metadata = CrafterTaskInstanceMetadata(
        difficulty="medium", 
        seed=777,
        num_trees_radius=4,
        num_cows_radius=2,
        num_hostiles_radius=1
    )
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test consistency"),
        intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Perform various actions to create a complex state
    action_sequence = [
        5,  # do (gather resource)
        2,  # right
        5,  # do
        1,  # up
        5,  # do
        10, # make_wood_pickaxe (if resources available)
    ]
    
    for action_idx in action_sequence:
        try:
            await env.step(EnvToolCall(tool="interact", args={"action": action_idx}))
        except:
            pass  # Some actions may fail if resources not available
    
    # Get current state details
    original_pub = env.engine._get_public_state_from_env()
    original_priv = env.engine._get_private_state_from_env(0, False, False)
    original_total_reward = env.engine._total_reward
    
    # Serialize
    snapshot = await env._serialize_engine()
    
    # Create new environment and deserialize
    new_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    restored_pub = new_env.engine._get_public_state_from_env()
    restored_priv = new_env.engine._get_private_state_from_env(0, False, False)
    
    # Check public state consistency
    assert restored_pub.player_position == original_pub.player_position
    assert restored_pub.inventory == original_pub.inventory
    assert restored_pub.achievements_status == original_pub.achievements_status
    assert restored_pub.num_steps_taken == original_pub.num_steps_taken
    
    # Check private state consistency
    assert restored_priv.player_internal_stats == original_priv.player_internal_stats
    assert new_env.engine._total_reward == original_total_reward
    
    # Verify we can continue playing from restored state
    before_step = restored_pub.num_steps_taken
    await new_env.step(EnvToolCall(tool="interact", args={"action": 0}))
    after_pub = new_env.engine._get_public_state_from_env()
    assert after_pub.num_steps_taken == before_step + 1
    
    print("✓ State consistency test passed")


@pytest.mark.asyncio
async def test_crafter_mcts_ready_serialization():
    """Test serialization patterns needed for MCTS implementation."""
    from synth_ai.environments.tasks.core import Impetus, Intent
    from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstanceMetadata
    from uuid import uuid4
    
    metadata = CrafterTaskInstanceMetadata(
        difficulty="easy", 
        seed=999,
        num_trees_radius=6,
        num_cows_radius=3,
        num_hostiles_radius=0
    )
    task = CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test MCTS"),
        intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    
    # Helper function to create heuristic score
    def heuristic_score(env: CrafterClassicEnvironment) -> float:
        pub = env.engine._get_public_state_from_env()
        priv = env.engine._get_private_state_from_env(0, False, False)
        
        score = 10.0  # Base score
        
        # Achievement bonus
        achievements_unlocked = sum(1 for v in pub.achievements_status.values() if v)
        score += achievements_unlocked * 100.0
        
        # Survival metrics
        health = priv.player_internal_stats.get("health", 0)
        hunger = priv.player_internal_stats.get("_hunger", 0)
        thirst = priv.player_internal_stats.get("_thirst", 0)
        
        score += health * 5.0
        score += (9 - hunger) * 2.0  # Lower is better
        score += (9 - thirst) * 2.0   # Lower is better
        
        # Inventory value
        for item, count in pub.inventory.items():
            if "pickaxe" in item:
                score += count * 20.0
            elif "sword" in item:
                score += count * 15.0
            else:
                score += count * 2.0
        
        return score
    
    # Initialize environment
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    initial_score = heuristic_score(env)
    
    # Simulate MCTS-style exploration
    with tempfile.TemporaryDirectory() as tmpdir:
        snap_store = FilesystemSnapshotStore(Path(tmpdir) / "mcts_test")
        tree = TrajectoryTreeStore(snap_store)
        
        # Create root
        root_snapshot = await env._serialize_engine()
        root_blob = gzip.compress(pickle.dumps(root_snapshot))
        root_id = tree.add_root(root_blob)
        
        # Explore a few actions and track scores
        action_scores: Dict[int, float] = {}
        
        for action_idx in [0, 1, 2, 3, 4, 5]:
            # Restore from root for fair comparison
            root_snapshot = pickle.loads(gzip.decompress(tree.load_snapshot_blob(root_id)))
            test_env = await CrafterClassicEnvironment._deserialize_engine(root_snapshot, task)
            
            # Take action
            await test_env.step(EnvToolCall(tool="interact", args={"action": action_idx}))
            
            # Calculate score after action
            action_scores[action_idx] = heuristic_score(test_env)
        
        # Find best action
        best_action = max(action_scores, key=action_scores.get)
        
        print("✓ MCTS-ready serialization test passed")
        print(f"  Initial score: {initial_score:.2f}")
        print(f"  Action scores: {action_scores}")
        print(f"  Best action: {best_action} (score: {action_scores[best_action]:.2f})")


# Run all tests
if __name__ == "__main__":
    async def main():
        print("Running Crafter Serialization Tests...")
        print("=" * 50)
        
        await test_basic_crafter_serialization()
        print()
        
        await test_crafter_tree_storage()
        print()
        
        await test_crafter_state_consistency()
        print()
        
        await test_crafter_mcts_ready_serialization()
        print()
        
        print("=" * 50)
        print("🎉 All tests passed!")
    
    asyncio.run(main())