"""
Test suite for Crafter serialization patch.
These tests will fail until we implement save/load methods for crafter.Env

The tests show what needs to be serialized for MCTS to work properly.
"""

import asyncio
import pickle
import gzip
import tempfile
from pathlib import Path
import numpy as np

import crafter
from synth_ai.environments.examples.crafter_classic.environment import CrafterClassicEnvironment
from synth_ai.environments.examples.crafter_classic.taskset import CrafterTaskInstance, CrafterTaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.tasks.core import Impetus, Intent
from uuid import uuid4


def test_crafter_env_has_save_load():
    """Test that crafter.Env has save/load methods (will fail until patched)."""
    env = crafter.Env()
    env.reset()
    
    # These methods need to exist
    assert hasattr(env, 'save'), "crafter.Env needs a save() method"
    assert hasattr(env, 'load'), "crafter.Env needs a load(state) method"
    
    print("✓ Crafter has save/load methods")


def test_crafter_save_load_basic():
    """Test basic save/load functionality."""
    env = crafter.Env(seed=42)
    env.reset()
    
    # Take some actions
    for _ in range(5):
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            break
    
    # Save state
    saved_state = env.save()
    
    # Record current state info
    original_step = env._step
    original_pos = env._player.pos.copy()
    original_health = env._player.health
    original_inventory = env._player.inventory.copy()
    
    # Take more actions to change state
    for _ in range(5):
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            break
    
    # State should be different now
    assert env._step != original_step or not np.array_equal(env._player.pos, original_pos)
    
    # Load saved state
    env.load(saved_state)
    
    # Verify restoration
    assert env._step == original_step
    assert np.array_equal(env._player.pos, original_pos)
    assert env._player.health == original_health
    assert env._player.inventory == original_inventory
    
    print("✓ Basic save/load works correctly")


def test_crafter_save_includes_all_state():
    """Test that save captures all necessary state components."""
    env = crafter.Env(seed=123)
    env.reset()
    
    # Perform various actions to create complex state
    env.step(5)  # do action
    env.step(2)  # move right
    
    saved_state = env.save()
    
    # Check that saved state includes all necessary components
    assert 'step' in saved_state, "Must save step count"
    assert 'seed' in saved_state, "Must save random seed"
    assert 'length' in saved_state, "Must save episode length"
    assert 'player' in saved_state, "Must save player state"
    assert 'world' in saved_state, "Must save world state"
    assert 'episode' in saved_state, "Must save episode number"
    
    # Check player state components
    player_state = saved_state['player']
    assert 'pos' in player_state
    assert 'health' in player_state
    assert 'inventory' in player_state
    assert 'achievements' in player_state
    assert 'facing' in player_state
    assert '_hunger' in player_state
    assert '_thirst' in player_state
    assert '_fatigue' in player_state
    
    # Check world state components
    world_state = saved_state['world']
    assert 'objects' in world_state
    assert 'chunks' in world_state
    assert 'step' in world_state
    assert 'random_state' in world_state
    
    print("✓ Save includes all necessary state components")


def test_crafter_deterministic_after_load():
    """Test that gameplay is deterministic after loading."""
    env1 = crafter.Env(seed=456)
    env1.reset()
    
    # Take some actions
    actions = [1, 2, 5, 3, 5, 0, 2]
    for action in actions[:3]:
        env1.step(action)
    
    # Save state
    saved_state = env1.save()
    
    # Continue from saved state in env1
    results1 = []
    for action in actions[3:]:
        obs, reward, done, info = env1.step(action)
        results1.append((obs.copy(), reward, done, env1._player.pos.copy()))
    
    # Create new env and load state
    env2 = crafter.Env(seed=789)  # Different seed shouldn't matter after load
    env2.reset()
    env2.load(saved_state)
    
    # Take same actions from loaded state
    results2 = []
    for action in actions[3:]:
        obs, reward, done, info = env2.step(action)
        results2.append((obs.copy(), reward, done, env2._player.pos.copy()))
    
    # Results should be identical
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        obs1, rew1, done1, pos1 = r1
        obs2, rew2, done2, pos2 = r2
        assert np.array_equal(obs1, obs2), f"Observations differ at step {i}"
        assert rew1 == rew2, f"Rewards differ at step {i}"
        assert done1 == done2, f"Done flags differ at step {i}"
        assert np.array_equal(pos1, pos2), f"Positions differ at step {i}"
    
    print("✓ Gameplay is deterministic after loading")


async def test_engine_serialization_with_patched_crafter():
    """Test that CrafterEngine serialization works with patched crafter."""
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
    
    # Take some actions
    actions = [
        EnvToolCall(tool="interact", args={"action": 5}),  # do
        EnvToolCall(tool="interact", args={"action": 2}),  # right
        EnvToolCall(tool="interact", args={"action": 5}),  # do
    ]
    
    for action in actions:
        await env.step(action)
    
    # Serialize
    snapshot = await env._serialize_engine()
    
    # Deserialize
    restored_env = await CrafterClassicEnvironment._deserialize_engine(snapshot, task)
    
    # Verify state matches
    original_pub = env.engine._get_public_state_from_env()
    restored_pub = restored_env.engine._get_public_state_from_env()
    
    assert original_pub.position == restored_pub.position
    assert original_pub.inventory == restored_pub.inventory
    assert original_pub.achievements_status == restored_pub.achievements_status
    assert original_pub.step_count == restored_pub.step_count
    
    print("✓ Engine serialization works with patched crafter")


def propose_crafter_patch():
    """
    Propose implementation for crafter save/load methods.
    This shows what needs to be added to crafter.Env
    """
    print("\n" + "="*60)
    print("PROPOSED CRAFTER PATCH")
    print("="*60)
    print("""
Add these methods to crafter.Env:

def save(self):
    '''Save complete environment state.'''
    # Save world objects
    objects_data = []
    for obj in self._world._objects:
        if obj is None:
            objects_data.append(None)
        else:
            obj_data = {
                'type': obj.__class__.__name__,
                'pos': obj.pos.tolist(),
                'health': getattr(obj, 'health', None),
                'inventory': getattr(obj, 'inventory', None),
                # Add other attributes based on object type
            }
            objects_data.append(obj_data)
    
    # Save world chunks
    chunks_data = {}
    for key, chunk in self._world._chunks.items():
        chunks_data[key] = {
            'objects': [id(obj) for obj in chunk]  # Reference by id
        }
    
    return {
        'step': self._step,
        'seed': self._seed,
        'length': self._length,
        'episode': self._episode,
        'player': {
            'pos': self._player.pos.tolist(),
            'facing': self._player.facing.tolist(),
            'health': self._player.health,
            'inventory': dict(self._player.inventory),
            'achievements': dict(self._player.achievements),
            '_hunger': self._player._hunger,
            '_thirst': self._player._thirst,
            '_fatigue': self._player._fatigue,
            '_recover': self._player._recover,
            'action': self._player.action,
            'sleeping': self._player.sleeping,
            '_last_health': self._player._last_health,
        },
        'world': {
            'objects': objects_data,
            'chunks': chunks_data,
            'step': self._world._step,
            'random_state': self._world.random.getstate(),
        }
    }

def load(self, state):
    '''Load environment state from saved data.'''
    # Restore basic attributes
    self._step = state['step']
    self._seed = state['seed']
    self._length = state['length']
    self._episode = state['episode']
    
    # Restore player
    player_data = state['player']
    self._player.pos = np.array(player_data['pos'])
    self._player.facing = np.array(player_data['facing'])
    self._player.health = player_data['health']
    self._player.inventory = collections.Counter(player_data['inventory'])
    self._player.achievements = dict(player_data['achievements'])
    self._player._hunger = player_data['_hunger']
    self._player._thirst = player_data['_thirst']
    self._player._fatigue = player_data['_fatigue']
    self._player._recover = player_data['_recover']
    self._player.action = player_data['action']
    self._player.sleeping = player_data['sleeping']
    self._player._last_health = player_data['_last_health']
    
    # Restore world objects
    # This is complex - need to recreate objects with correct types
    # and restore their states...
    
    # Restore random state
    self._world.random.setstate(state['world']['random_state'])
""")
    print("="*60)


if __name__ == "__main__":
    print("Running Crafter Serialization Patch Tests...")
    print("These tests show what needs to be implemented.")
    print()
    
    try:
        test_crafter_env_has_save_load()
    except AssertionError as e:
        print(f"❌ Expected failure: {e}")
    
    try:
        test_crafter_save_load_basic()
    except (AttributeError, AssertionError) as e:
        print(f"❌ Expected failure: {e}")
    
    try:
        test_crafter_save_includes_all_state()
    except (AttributeError, AssertionError) as e:
        print(f"❌ Expected failure: {e}")
    
    try:
        test_crafter_deterministic_after_load()
    except (AttributeError, AssertionError) as e:
        print(f"❌ Expected failure: {e}")
    
    try:
        asyncio.run(test_engine_serialization_with_patched_crafter())
    except (AttributeError, AssertionError) as e:
        print(f"❌ Expected failure: {e}")
    
    # Show proposed patch
    propose_crafter_patch()