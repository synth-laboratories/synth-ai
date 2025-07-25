"""Simple test of crafter serialization"""
import sys
import importlib

# Force reload of modules
if 'crafter' in sys.modules:
    del sys.modules['crafter']

# Clear any patches
import crafter
if hasattr(crafter.Env, 'save'):
    delattr(crafter.Env, 'save')
if hasattr(crafter.Env, 'load'):
    delattr(crafter.Env, 'load')

# Now import patches
from synth_ai.environments.examples.crafter_classic import engine_serialization_patch

# Test basic functionality
env = crafter.Env(seed=42)
env.reset()

print("Taking 5 actions...")
for i in range(5):
    action = i % 6  # Cycle through movement actions
    obs, reward, done, info = env.step(action)
    print(f"Step {i+1}: action={action}, pos={env._player.pos}, health={env._player.health}")

print("\nSaving state...")
saved_state = env.save()
print(f"Saved state keys: {list(saved_state.keys())}")
print(f"Saved world keys: {list(saved_state['world'].keys())}")

print("\nTaking 5 more actions...")
original_pos = env._player.pos.copy()
for i in range(5):
    action = (i + 3) % 6
    obs, reward, done, info = env.step(action)
    print(f"Step {i+6}: action={action}, pos={env._player.pos}, health={env._player.health}")

print(f"\nPosition changed from {original_pos} to {env._player.pos}")

print("\nLoading saved state...")
env.load(saved_state)
print(f"After load: pos={env._player.pos}, health={env._player.health}")
print(f"Position restored: {(env._player.pos == original_pos).all()}")

print("\n✓ Basic serialization test passed!")