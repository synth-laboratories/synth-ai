"""
Detailed analysis of why diamonds aren't spawning despite high probability.
"""

import numpy as np
import opensimplex
from crafter.config import WorldGenConfig
from crafter.worldgen import _simplex

def trace_single_position(x, y, config, simplex):
    """Trace through the exact logic for a single position."""
    print(f"\nTracing position ({x}, {y}):")
    
    # Terrain generation
    water = _simplex(simplex, x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
    mountain = _simplex(simplex, x, y, 0, {15: 1, 5: 0.3})
    mountain -= 0.3 * water
    
    print(f"  Water value: {water:.3f}")
    print(f"  Mountain value: {mountain:.3f}")
    print(f"  Mountain threshold: {config.mountain_threshold}")
    
    if mountain <= config.mountain_threshold:
        print(f"  ❌ Not in mountain area (mountain={mountain:.3f} <= threshold={config.mountain_threshold})")
        return False
    
    print(f"  ✓ In mountain area")
    
    # Check each condition in order
    # Cave check
    cave1 = _simplex(simplex, x, y, 6, 7)
    cave2 = _simplex(simplex, x, y, 6, 5)
    print(f"  Cave check 1: {cave1:.3f} > 0.15 and mountain > 0.3? {cave1 > 0.15 and mountain > 0.3}")
    print(f"  Cave check 2: {cave2:.3f} > {config.cave_threshold}? {cave2 > config.cave_threshold}")
    
    if (cave1 > 0.15 and mountain > 0.3) or cave2 > config.cave_threshold:
        print(f"  ❌ Blocked by cave")
        return False
    
    # Tunnel checks
    h_tunnel = _simplex(simplex, 2 * x, y / 5, 7, 3)
    v_tunnel = _simplex(simplex, x / 5, 2 * y, 7, 3)
    print(f"  Horizontal tunnel: {h_tunnel:.3f} > 0.4? {h_tunnel > 0.4}")
    print(f"  Vertical tunnel: {v_tunnel:.3f} > 0.4? {v_tunnel > 0.4}")
    
    if h_tunnel > 0.4 or v_tunnel > 0.4:
        print(f"  ❌ Blocked by tunnel")
        return False
    
    # Coal check
    coal_noise = _simplex(simplex, x, y, 1, 8)
    coal_random = np.random.uniform()
    print(f"  Coal noise: {coal_noise:.3f} > {config.coal_threshold}? {coal_noise > config.coal_threshold}")
    print(f"  Coal random: {coal_random:.3f} > {1 - config.coal_probability:.3f}? {coal_random > (1 - config.coal_probability)}")
    
    if coal_noise > config.coal_threshold and coal_random > (1 - config.coal_probability):
        print(f"  ❌ Blocked by coal")
        return False
    
    # Iron check
    iron_noise = _simplex(simplex, x, y, 2, 6)
    iron_random = np.random.uniform()
    print(f"  Iron noise: {iron_noise:.3f} > {config.iron_threshold}? {iron_noise > config.iron_threshold}")
    print(f"  Iron random: {iron_random:.3f} > {1 - config.iron_probability:.3f}? {iron_random > (1 - config.iron_probability)}")
    
    if iron_noise > config.iron_threshold and iron_random > (1 - config.iron_probability):
        print(f"  ❌ Blocked by iron")
        return False
    
    # Diamond check
    print(f"  Diamond threshold check: {mountain:.3f} > {config.diamond_threshold}? {mountain > config.diamond_threshold}")
    
    if mountain > config.diamond_threshold:
        diamond_random = np.random.uniform()
        print(f"  ✓ Diamond threshold met!")
        print(f"  Diamond random: {diamond_random:.3f} > {1 - config.diamond_probability:.3f}? {diamond_random > (1 - config.diamond_probability)}")
        
        if diamond_random > (1 - config.diamond_probability):
            print(f"  ✅ DIAMOND SPAWNED!")
            return True
        else:
            print(f"  ❌ Diamond probability check failed")
    else:
        print(f"  ❌ Mountain value too low for diamonds")
    
    return False

def find_diamond_candidates(config, num_positions=1000, seed=42):
    """Find positions that meet the diamond threshold."""
    np.random.seed(seed)
    simplex = opensimplex.OpenSimplex(seed=seed)
    
    candidates = []
    
    for i in range(num_positions):
        x = np.random.randint(10, 200)
        y = np.random.randint(10, 200)
        
        water = _simplex(simplex, x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
        mountain = _simplex(simplex, x, y, 0, {15: 1, 5: 0.3})
        mountain -= 0.3 * water
        
        if mountain > config.diamond_threshold:
            # Check if it would pass all other checks
            cave1 = _simplex(simplex, x, y, 6, 7) > 0.15 and mountain > 0.3
            cave2 = _simplex(simplex, x, y, 6, 5) > config.cave_threshold
            h_tunnel = _simplex(simplex, 2 * x, y / 5, 7, 3) > 0.4
            v_tunnel = _simplex(simplex, x / 5, 2 * y, 7, 3) > 0.4
            
            if not (cave1 or cave2 or h_tunnel or v_tunnel):
                # This position could potentially spawn a diamond
                candidates.append((x, y, mountain))
    
    return candidates

# The key issue: the probability condition
print("THE KEY ISSUE WITH DIAMOND SPAWNING:")
print("=" * 50)
print("\nIn the worldgen.py code, line 55:")
print("  elif mountain > config.diamond_threshold and uniform() > (1 - config.diamond_probability):")
print()
print("With default diamond_probability = 0.006:")
print(f"  The condition uniform() > (1 - 0.006) means uniform() > 0.994")
print(f"  This means we need to roll > 0.994 to spawn a diamond")
print(f"  That's only a 0.6% chance!")
print()
print("The comment says 'high probability' but 0.006 is actually VERY LOW!")
print()
print("To have 'high probability', diamond_probability should be something like:")
print("  - 0.5 for 50% chance")
print("  - 0.8 for 80% chance")
print("  - 0.95 for 95% chance")
print()

# Demonstrate with actual positions
print("\nDemonstration with a few random positions:")
print("=" * 50)

config = WorldGenConfig(diamond_probability=0.006)  # Default low probability
np.random.seed(42)
simplex = opensimplex.OpenSimplex(seed=42)

# Find some candidate positions
candidates = find_diamond_candidates(config, num_positions=5000)
print(f"\nFound {len(candidates)} positions that could spawn diamonds")

if candidates:
    # Trace a few
    for i in range(min(3, len(candidates))):
        x, y, mountain = candidates[i]
        trace_single_position(x, y, config, simplex)

print("\n" + "=" * 50)
print("\nNow testing with HIGH probability (0.95):")
config_high = WorldGenConfig(diamond_probability=0.95)

if candidates:
    x, y, mountain = candidates[0]
    trace_single_position(x, y, config_high, simplex)