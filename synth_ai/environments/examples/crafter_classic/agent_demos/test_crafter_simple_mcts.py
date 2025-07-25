#!/usr/bin/env python3
"""
Simplified Random MCTS for Crafter
==================================
A simpler version that avoids serialization issues during rollouts.
"""

import asyncio
import random
import time
import logging
from typing import Dict, List, Tuple
from uuid import uuid4

from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("simple-mcts")

# Crafter action space
CRAFTER_ACTIONS = list(range(17))
ACTION_NAMES = [
    "noop", "left", "right", "up", "down", "do",
    "place_stone", "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword", "rest"
]


async def simple_mcts_search(seed: int = 42, max_actions: int = 100, num_rollouts: int = 20):
    """
    Simple MCTS-like search without full tree structure.
    Just tries different action sequences and tracks which ones lead to achievements.
    """
    
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
        impetus=Impetus(instructions="Find achievements"),
        intent=Intent(rubric={"goal": "Unlock achievements"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None
    )
    
    LOG.info(f"🎮 Starting Simple MCTS - Seed: {seed}")
    LOG.info("=" * 60)
    
    best_sequence = []
    best_achievements = 0
    achievements_found = set()
    
    # Try multiple random rollouts
    for rollout_num in range(num_rollouts):
        # Create fresh environment for each rollout
        env = CrafterClassicEnvironment(task)
        await env.initialize()
        
        action_sequence = []
        
        # Execute random actions
        for step in range(max_actions):
            # Choose action with bias
            if random.random() < 0.4:  # 40% chance for useful actions
                # Bias towards movement and 'do'
                if random.random() < 0.7:
                    action = random.choice([1, 2, 3, 4, 5, 5])  # Move or do
                else:
                    # Try crafting actions if we have resources
                    pub = env.engine._get_public_state_from_env()
                    if pub.inventory.get("wood", 0) >= 2:
                        action = random.choice([10, 11, 12, 13, 14, 15])  # Crafting
                    else:
                        action = 5  # Do
            else:
                action = random.choice(CRAFTER_ACTIONS)
            
            action_sequence.append(action)
            
            try:
                call = EnvToolCall(tool="interact", args={"action": action})
                obs = await env.step(call)
                
                # Check if we're dead
                if obs.get("terminated", False):
                    break
                
                # Check achievements periodically
                if step % 10 == 0:
                    pub = env.engine._get_public_state_from_env()
                    current_achievements = sum(1 for v in pub.achievements_status.values() if v)
                    
                    if current_achievements > 0:
                        # Log new achievements
                        for ach, status in pub.achievements_status.items():
                            if status and ach not in achievements_found:
                                achievements_found.add(ach)
                                LOG.info(f"  🎯 Rollout {rollout_num}: Found '{ach}' at step {step}")
                        
                        if current_achievements > best_achievements:
                            best_achievements = current_achievements
                            best_sequence = action_sequence.copy()
                
            except Exception as e:
                LOG.debug(f"Step failed: {e}")
                break
        
        # Final check
        try:
            pub = env.engine._get_public_state_from_env()
            final_achievements = sum(1 for v in pub.achievements_status.values() if v)
            
            if final_achievements > best_achievements:
                best_achievements = final_achievements
                best_sequence = action_sequence.copy()
                
            if rollout_num % 5 == 0:
                LOG.info(f"Rollout {rollout_num}/{num_rollouts}: {final_achievements} achievements")
                
        except Exception:
            pass
    
    LOG.info("\n" + "=" * 60)
    LOG.info(f"📊 Results after {num_rollouts} rollouts:")
    LOG.info(f"Best achievement count: {best_achievements}")
    LOG.info(f"Unique achievements found: {len(achievements_found)}")
    
    if achievements_found:
        LOG.info("\n🏆 Achievements discovered:")
        for ach in sorted(achievements_found):
            LOG.info(f"  ✓ {ach}")
    
    # Replay best sequence
    if best_sequence and best_achievements > 0:
        LOG.info(f"\n🔄 Replaying best sequence ({len(best_sequence)} actions)...")
        
        env = CrafterClassicEnvironment(task)
        await env.initialize()
        
        for i, action in enumerate(best_sequence[:50]):  # First 50 actions
            try:
                call = EnvToolCall(tool="interact", args={"action": action})
                await env.step(call)
                
                if i % 10 == 0:
                    pub = env.engine._get_public_state_from_env()
                    ach_count = sum(1 for v in pub.achievements_status.values() if v)
                    LOG.info(f"  Step {i}: {ACTION_NAMES[action]}, Achievements: {ach_count}")
                    
            except Exception:
                break
        
        # Final state
        try:
            pub = env.engine._get_public_state_from_env()
            final_pub_achievements = sum(1 for v in pub.achievements_status.values() if v)
            LOG.info(f"\nFinal state: {final_pub_achievements} achievements")
            
            # Show inventory
            inv_items = [(k, v) for k, v in pub.inventory.items() if v > 0 and k not in ['health', 'food', 'drink', 'energy']]
            if inv_items:
                LOG.info(f"Inventory: {dict(inv_items)}")
                
        except Exception:
            pass
    
    return best_achievements > 0


async def test_multiple_seeds():
    """Test MCTS on multiple random seeds."""
    
    LOG.info("🎲 Testing Simple MCTS on multiple seeds...")
    LOG.info("Can random exploration discover achievements?")
    LOG.info("=" * 60)
    
    successes = 0
    seeds_to_test = 5
    
    for i in range(seeds_to_test):
        seed = random.randint(0, 10000)
        LOG.info(f"\n📍 Test {i+1}/{seeds_to_test} - Seed {seed}:")
        
        success = await simple_mcts_search(
            seed=seed,
            max_actions=100,
            num_rollouts=10
        )
        
        if success:
            successes += 1
            LOG.info("✅ Success! Found achievements.")
        else:
            LOG.info("❌ No achievements found.")
        
        LOG.info("-" * 60)
    
    LOG.info(f"\n📈 Overall Results:")
    LOG.info(f"Success rate: {successes}/{seeds_to_test} ({100*successes/seeds_to_test:.0f}%)")
    LOG.info("🎉 Experiment complete!")


if __name__ == "__main__":
    asyncio.run(test_multiple_seeds())