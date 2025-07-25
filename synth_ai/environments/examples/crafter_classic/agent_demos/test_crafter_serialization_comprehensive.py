#!/usr/bin/env python3
"""
Comprehensive Serialization Tests for Crafter
=============================================
Tests EVERY aspect of game state to ensure perfect serialization/deserialization.
"""

import asyncio
import numpy as np
import pickle
import gzip
import random
from uuid import uuid4
from typing import Any, Dict, List
import pytest

import crafter
from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent


def create_test_task(seed: int = 42) -> CrafterTaskInstance:
    """Create a task instance for testing."""
    metadata = CrafterTaskInstanceMetadata(
        difficulty="easy",
        seed=seed,
        num_trees_radius=5,
        num_cows_radius=2,
        num_hostiles_radius=0
    )
    return CrafterTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test serialization"),
        intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None
    )


def deep_compare(obj1: Any, obj2: Any, path: str = "") -> List[str]:
    """Deep comparison of two objects, returning list of differences."""
    differences = []
    
    if type(obj1) != type(obj2):
        differences.append(f"{path}: Type mismatch - {type(obj1)} vs {type(obj2)}")
        return differences
    
    if isinstance(obj1, np.ndarray):
        if not np.array_equal(obj1, obj2):
            differences.append(f"{path}: Array mismatch - shapes {obj1.shape} vs {obj2.shape}, values differ")
    elif isinstance(obj1, dict):
        keys1, keys2 = set(obj1.keys()), set(obj2.keys())
        if keys1 != keys2:
            differences.append(f"{path}: Dict keys mismatch - {keys1 ^ keys2}")
        for key in keys1 & keys2:
            differences.extend(deep_compare(obj1[key], obj2[key], f"{path}.{key}"))
    elif isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            differences.append(f"{path}: Length mismatch - {len(obj1)} vs {len(obj2)}")
        else:
            for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                differences.extend(deep_compare(item1, item2, f"{path}[{i}]"))
    elif hasattr(obj1, '__dict__'):
        # Compare object attributes
        differences.extend(deep_compare(obj1.__dict__, obj2.__dict__, f"{path}.__dict__"))
    else:
        if obj1 != obj2:
            differences.append(f"{path}: Value mismatch - {obj1} vs {obj2}")
    
    return differences


@pytest.mark.asyncio
async def test_basic_env_state_preservation():
    """Test that basic environment state is preserved."""
    task = create_test_task(seed=100)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Take some actions to create interesting state
    actions = [5, 2, 5, 1, 5, 3, 5]  # do, right, do, up, do, down, do
    for action in actions:
        await env.step(EnvToolCall(tool="interact", args={"action": action}))
    
    # Get state before serialization
    before_pub = env.engine._get_public_state_from_env()
    before_priv = env.engine._get_private_state_from_env(0, False, False)
    before_step = env.engine.env._step
    before_seed = env.engine.env._seed
    before_length = env.engine.env._length
    before_episode = env.engine.env._episode
    
    # Serialize and deserialize
    snapshot = await env._serialize_engine()
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Get state after deserialization
    after_pub = restored_env.engine._get_public_state_from_env()
    after_priv = restored_env.engine._get_private_state_from_env(0, False, False)
    after_step = restored_env.engine.env._step
    after_seed = restored_env.engine.env._seed
    after_length = restored_env.engine.env._length
    after_episode = restored_env.engine.env._episode
    
    # Assert everything matches
    assert before_step == after_step, f"Step count mismatch: {before_step} vs {after_step}"
    assert before_seed == after_seed, f"Seed mismatch: {before_seed} vs {after_seed}"
    assert before_length == after_length, f"Length mismatch: {before_length} vs {after_length}"
    assert before_episode == after_episode, f"Episode mismatch: {before_episode} vs {after_episode}"
    
    # Public state
    assert before_pub.player_position == after_pub.player_position
    assert before_pub.inventory == after_pub.inventory
    assert before_pub.achievements_status == after_pub.achievements_status
    assert before_pub.num_steps_taken == after_pub.num_steps_taken
    
    # Private state
    assert before_priv.player_internal_stats == after_priv.player_internal_stats
    assert before_priv.total_reward_episode == after_priv.total_reward_episode
    
    print("✓ Basic environment state preservation test passed")


@pytest.mark.asyncio
async def test_player_state_exact():
    """Test that all player attributes are preserved exactly."""
    task = create_test_task(seed=200)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Modify player state in various ways
    player = env.engine.env._player
    
    # Move player
    await env.step(EnvToolCall(tool="interact", args={"action": 1}))  # left
    await env.step(EnvToolCall(tool="interact", args={"action": 3}))  # up
    
    # Try to collect resources
    for _ in range(10):
        await env.step(EnvToolCall(tool="interact", args={"action": 5}))  # do
    
    # Record ALL player attributes before serialization
    before_player = {
        'pos': player.pos.copy(),
        'facing': player.facing if hasattr(player, 'facing') else None,
        'health': player.health,
        'inventory': dict(player.inventory),
        'achievements': dict(player.achievements),
        '_hunger': getattr(player, '_hunger', None),
        '_thirst': getattr(player, '_thirst', None),
        '_fatigue': getattr(player, '_fatigue', None),
        '_recover': getattr(player, '_recover', None),
        'action': getattr(player, 'action', None),
        'sleeping': getattr(player, 'sleeping', None),
        '_last_health': getattr(player, '_last_health', None),
    }
    
    # Serialize and deserialize
    snapshot = await env._serialize_engine()
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    restored_player = restored_env.engine.env._player
    
    # Check EVERY attribute
    assert np.array_equal(before_player['pos'], restored_player.pos), "Player position mismatch"
    if before_player['facing'] is not None:
        assert before_player['facing'] == restored_player.facing, "Player facing mismatch"
    assert before_player['health'] == restored_player.health, "Player health mismatch"
    assert before_player['inventory'] == dict(restored_player.inventory), "Player inventory mismatch"
    assert before_player['achievements'] == dict(restored_player.achievements), "Player achievements mismatch"
    assert before_player['_hunger'] == getattr(restored_player, '_hunger', None), "Player hunger mismatch"
    assert before_player['_thirst'] == getattr(restored_player, '_thirst', None), "Player thirst mismatch"
    assert before_player['_fatigue'] == getattr(restored_player, '_fatigue', None), "Player fatigue mismatch"
    assert before_player['_recover'] == getattr(restored_player, '_recover', None), "Player recover mismatch"
    assert before_player['action'] == getattr(restored_player, 'action', None), "Player action mismatch"
    assert before_player['sleeping'] == getattr(restored_player, 'sleeping', None), "Player sleeping mismatch"
    assert before_player['_last_health'] == getattr(restored_player, '_last_health', None), "Player last health mismatch"
    
    print("✓ Player state exact preservation test passed")


@pytest.mark.asyncio
async def test_world_objects_preservation():
    """Test that all world objects are preserved exactly."""
    task = create_test_task(seed=300)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Take actions to potentially modify world
    for _ in range(20):
        action = random.choice([0, 1, 2, 3, 4, 5])
        await env.step(EnvToolCall(tool="interact", args={"action": action}))
    
    # Get world state before
    world_before = env.engine.env._world
    objects_before = []
    for obj in world_before._objects:
        if obj is not None:
            obj_info = {
                'type': type(obj).__name__,
                'pos': obj.pos.copy() if hasattr(obj, 'pos') else None,
                'health': getattr(obj, 'health', None),
                'inventory': dict(getattr(obj, 'inventory', {})),
                'attributes': {}
            }
            # Capture additional attributes
            for attr in ['grown', 'kind', 'facing', 'cooldown', 'reload']:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    if isinstance(val, np.ndarray):
                        obj_info['attributes'][attr] = val.copy()
                    else:
                        obj_info['attributes'][attr] = val
            objects_before.append(obj_info)
        else:
            objects_before.append(None)
    
    # Serialize and deserialize
    snapshot = await env._serialize_engine()
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Get world state after
    world_after = restored_env.engine.env._world
    
    # Compare object counts
    assert len(world_before._objects) == len(world_after._objects), \
        f"Object count mismatch: {len(world_before._objects)} vs {len(world_after._objects)}"
    
    # Compare each object
    for i, (obj_before, obj_after) in enumerate(zip(world_before._objects, world_after._objects)):
        if obj_before is None:
            assert obj_after is None, f"Object {i} should be None"
        else:
            assert obj_after is not None, f"Object {i} should not be None"
            assert type(obj_before).__name__ == type(obj_after).__name__, \
                f"Object {i} type mismatch: {type(obj_before).__name__} vs {type(obj_after).__name__}"
            
            if hasattr(obj_before, 'pos'):
                assert np.array_equal(obj_before.pos, obj_after.pos), f"Object {i} position mismatch"
            if hasattr(obj_before, 'health'):
                assert obj_before.health == obj_after.health, f"Object {i} health mismatch"
    
    print("✓ World objects preservation test passed")


@pytest.mark.asyncio
async def test_world_maps_preservation():
    """Test that world maps (material and object maps) are preserved."""
    task = create_test_task(seed=400)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Take various actions
    for _ in range(15):
        action = random.choice(range(17))
        try:
            await env.step(EnvToolCall(tool="interact", args={"action": action}))
        except:
            pass  # Some actions might fail
    
    # Get maps before
    world = env.engine.env._world
    mat_map_before = world._mat_map.copy()
    obj_map_before = world._obj_map.copy()
    area_before = world.area
    daylight_before = world.daylight
    
    # Serialize and deserialize
    snapshot = await env._serialize_engine()
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Get maps after
    restored_world = restored_env.engine.env._world
    mat_map_after = restored_world._mat_map
    obj_map_after = restored_world._obj_map
    area_after = restored_world.area
    daylight_after = restored_world.daylight
    
    # Compare
    assert np.array_equal(mat_map_before, mat_map_after), "Material map mismatch"
    assert np.array_equal(obj_map_before, obj_map_after), "Object map mismatch"
    assert area_before == area_after, f"Area mismatch: {area_before} vs {area_after}"
    assert daylight_before == daylight_after, f"Daylight mismatch: {daylight_before} vs {daylight_after}"
    
    print("✓ World maps preservation test passed")


@pytest.mark.asyncio
async def test_random_state_preservation():
    """Test that random state is preserved for deterministic behavior."""
    task = create_test_task(seed=500)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Take some actions
    for _ in range(5):
        await env.step(EnvToolCall(tool="interact", args={"action": random.choice([0, 1, 2, 3, 4, 5])}))
    
    # Save state
    snapshot = await env._serialize_engine()
    
    # Generate some random numbers from original
    world = env.engine.env._world
    original_randoms = [world.random.randint(0, 1000) for _ in range(10)]
    
    # Restore state
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    restored_world = restored_env.engine.env._world
    
    # Generate same random numbers from restored
    restored_randoms = [restored_world.random.randint(0, 1000) for _ in range(10)]
    
    # Should be identical
    assert original_randoms == restored_randoms, f"Random sequences differ: {original_randoms} vs {restored_randoms}"
    
    print("✓ Random state preservation test passed")


@pytest.mark.asyncio
async def test_deterministic_gameplay_after_restore():
    """Test that gameplay is completely deterministic after restore."""
    task = create_test_task(seed=600)
    env1 = CrafterClassicEnvironment(task)
    await env1.initialize()
    
    # Take initial actions
    initial_actions = [1, 2, 5, 3, 5, 0, 2, 5]
    for action in initial_actions[:4]:
        await env1.step(EnvToolCall(tool="interact", args={"action": action}))
    
    # Save state
    snapshot = await env1._serialize_engine()
    
    # Continue in env1
    results1 = []
    for action in initial_actions[4:]:
        obs = await env1.step(EnvToolCall(tool="interact", args={"action": action}))
        pub = env1.engine._get_public_state_from_env()
        results1.append({
            'position': pub.player_position,
            'inventory': dict(pub.inventory),
            'achievements': sum(1 for v in pub.achievements_status.values() if v),
            'reward': obs.get('reward_last_step', 0),
            'observation': obs.get('observation_image', np.array([])).shape,
        })
    
    # Create new env and restore
    env2 = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Take same actions
    results2 = []
    for action in initial_actions[4:]:
        obs = await env2.step(EnvToolCall(tool="interact", args={"action": action}))
        pub = env2.engine._get_public_state_from_env()
        results2.append({
            'position': pub.player_position,
            'inventory': dict(pub.inventory),
            'achievements': sum(1 for v in pub.achievements_status.values() if v),
            'reward': obs.get('reward_last_step', 0),
            'observation': obs.get('observation_image', np.array([])).shape,
        })
    
    # Compare results
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        assert r1['position'] == r2['position'], f"Step {i}: Position mismatch"
        assert r1['inventory'] == r2['inventory'], f"Step {i}: Inventory mismatch"
        assert r1['achievements'] == r2['achievements'], f"Step {i}: Achievements mismatch"
        # Allow small floating point differences in rewards
        assert abs(r1['reward'] - r2['reward']) < 0.01, f"Step {i}: Reward mismatch {r1['reward']} vs {r2['reward']}"
        assert r1['observation'] == r2['observation'], f"Step {i}: Observation shape mismatch"
    
    print("✓ Deterministic gameplay test passed")


@pytest.mark.asyncio
async def test_inventory_and_achievements_exact():
    """Test inventory and achievements preservation with complex state."""
    task = create_test_task(seed=700)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Try to get some achievements and items
    action_sequence = [
        5, 5, 5,  # do actions to collect resources
        1, 5, 5,  # move and collect
        2, 5, 5,  # move and collect
        3, 5, 5,  # move and collect
        4, 5, 5,  # move and collect
        10,       # try to craft
    ]
    
    for action in action_sequence:
        try:
            await env.step(EnvToolCall(tool="interact", args={"action": action}))
        except:
            pass
    
    # Get exact state before
    pub_before = env.engine._get_public_state_from_env()
    inventory_before = dict(pub_before.inventory)
    achievements_before = dict(pub_before.achievements_status)
    
    # Serialize and deserialize
    snapshot = await env._serialize_engine()
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Get exact state after
    pub_after = restored_env.engine._get_public_state_from_env()
    inventory_after = dict(pub_after.inventory)
    achievements_after = dict(pub_after.achievements_status)
    
    # Compare item by item
    assert set(inventory_before.keys()) == set(inventory_after.keys()), "Inventory keys mismatch"
    for item, count in inventory_before.items():
        assert inventory_after[item] == count, f"Inventory {item} count mismatch: {count} vs {inventory_after[item]}"
    
    # Compare achievement by achievement
    assert set(achievements_before.keys()) == set(achievements_after.keys()), "Achievement keys mismatch"
    for ach, status in achievements_before.items():
        assert achievements_after[ach] == status, f"Achievement {ach} status mismatch: {status} vs {achievements_after[ach]}"
    
    print("✓ Inventory and achievements exact test passed")


@pytest.mark.asyncio
async def test_edge_cases():
    """Test edge cases like dead player, night time, etc."""
    task = create_test_task(seed=800)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Test 1: Empty inventory at start
    snapshot1 = await env._serialize_engine()
    restored1 = await CrafterClassicEnvironment._deserialize_engine(snapshot1, task)
    pub1 = restored1.engine._get_public_state_from_env()
    assert all(v == 0 or k in ['health', 'food', 'drink', 'energy'] for k, v in pub1.inventory.items())
    
    # Test 2: After many steps (day/night cycle)
    for _ in range(100):
        await env.step(EnvToolCall(tool="interact", args={"action": 0}))  # noop
    
    snapshot2 = await env._serialize_engine()
    world_before = env.engine.env._world
    daylight_before = world_before.daylight
    
    restored2 = await CrafterClassicEnvironment._deserialize_engine(snapshot2, task)
    world_after = restored2.engine.env._world
    daylight_after = world_after.daylight
    
    assert daylight_before == daylight_after, f"Daylight preservation failed: {daylight_before} vs {daylight_after}"
    
    print("✓ Edge cases test passed")


@pytest.mark.asyncio
async def test_multiple_serialize_deserialize_cycles():
    """Test that multiple serialization cycles don't degrade state."""
    task = create_test_task(seed=900)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    # Take some actions
    for _ in range(10):
        await env.step(EnvToolCall(tool="interact", args={"action": random.choice(range(6))}))
    
    # Get initial state snapshot
    initial_pub = env.engine._get_public_state_from_env()
    initial_pos = initial_pub.player_position
    initial_inv = dict(initial_pub.inventory)
    initial_achievements = dict(initial_pub.achievements_status)
    
    # Do multiple serialize/deserialize cycles
    current_env = env
    for cycle in range(5):
        snapshot = await current_env._serialize_engine()
        current_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
        
        # Verify state hasn't changed
        pub = current_env.engine._get_public_state_from_env()
        assert pub.player_position == initial_pos, f"Cycle {cycle}: Position changed"
        assert dict(pub.inventory) == initial_inv, f"Cycle {cycle}: Inventory changed"
        assert dict(pub.achievements_status) == initial_achievements, f"Cycle {cycle}: Achievements changed"
    
    print("✓ Multiple serialization cycles test passed")


@pytest.mark.asyncio
async def test_compressed_size_reasonable():
    """Test that serialized states have reasonable compressed sizes."""
    task = create_test_task(seed=1000)
    env = CrafterClassicEnvironment(task)
    await env.initialize()
    
    sizes = []
    
    # Test various game states
    for i in range(10):
        # Take some random actions
        for _ in range(i * 5):
            await env.step(EnvToolCall(tool="interact", args={"action": random.choice(range(17))}))
        
        # Serialize and check size
        snapshot = await env._serialize_engine()
        compressed = gzip.compress(pickle.dumps(snapshot))
        size_kb = len(compressed) / 1024
        sizes.append(size_kb)
        
        assert size_kb < 500, f"State {i} too large: {size_kb:.1f} KB"
    
    print(f"✓ Compressed sizes reasonable: {min(sizes):.1f} - {max(sizes):.1f} KB")


# Run all tests
if __name__ == "__main__":
    async def run_all_tests():
        print("Running Comprehensive Crafter Serialization Tests")
        print("=" * 60)
        
        tests = [
            test_basic_env_state_preservation,
            test_player_state_exact,
            test_world_objects_preservation,
            test_world_maps_preservation,
            test_random_state_preservation,
            test_deterministic_gameplay_after_restore,
            test_inventory_and_achievements_exact,
            test_edge_cases,
            test_multiple_serialize_deserialize_cycles,
            test_compressed_size_reasonable,
        ]
        
        failed = 0
        for test in tests:
            try:
                await test()
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                failed += 1
        
        print("\n" + "=" * 60)
        if failed == 0:
            print(f"🎉 All {len(tests)} tests passed!")
        else:
            print(f"❌ {failed}/{len(tests)} tests failed")
        
        return failed == 0
    
    import sys
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)