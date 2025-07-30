"""
Analyze diamond spawning in Crafter world generation.
This script helps understand why diamonds aren't appearing with high probability.
"""

import numpy as np
import opensimplex
from crafter.config import WorldGenConfig
from crafter.worldgen import _simplex

def analyze_diamond_conditions(config=None, num_samples=10000, seed=42):
    """Analyze how often diamond spawning conditions are met."""
    if config is None:
        config = WorldGenConfig()
    
    np.random.seed(seed)
    simplex = opensimplex.OpenSimplex(seed=seed)
    
    # Track statistics
    stats = {
        'total_positions': num_samples,
        'mountain_positions': 0,
        'diamond_threshold_met': 0,
        'diamonds_spawned': 0,
        'blocked_by_coal': 0,
        'blocked_by_iron': 0,
        'blocked_by_caves': 0,
        'blocked_by_tunnels': 0,
        'blocked_by_lava': 0,
        'mountain_values': [],
        'positions_checked': []
    }
    
    # Simulate world generation at random positions
    for i in range(num_samples):
        x = np.random.randint(10, 200)  # Avoid spawn area
        y = np.random.randint(10, 200)
        
        # Simulate terrain generation (simplified - no player spawn adjustment)
        water = _simplex(simplex, x, y, 3, {15: 1, 5: 0.15}, False) + 0.1
        mountain = _simplex(simplex, x, y, 0, {15: 1, 5: 0.3})
        mountain -= 0.3 * water  # Simplified without spawn adjustment
        
        stats['mountain_values'].append(mountain)
        
        # Check if in mountain area
        if mountain > config.mountain_threshold:
            stats['mountain_positions'] += 1
            
            # Check cave conditions
            cave_check1 = _simplex(simplex, x, y, 6, 7) > 0.15 and mountain > 0.3
            cave_check2 = _simplex(simplex, x, y, 6, 5) > config.cave_threshold
            if cave_check1 or cave_check2:
                stats['blocked_by_caves'] += 1
                continue
                
            # Check tunnel conditions
            h_tunnel = _simplex(simplex, 2 * x, y / 5, 7, 3) > 0.4
            v_tunnel = _simplex(simplex, x / 5, 2 * y, 7, 3) > 0.4
            if h_tunnel or v_tunnel:
                stats['blocked_by_tunnels'] += 1
                continue
                
            # Check coal condition
            coal_noise = _simplex(simplex, x, y, 1, 8) > config.coal_threshold
            coal_prob = np.random.uniform() > (1 - config.coal_probability)
            if coal_noise and coal_prob:
                stats['blocked_by_coal'] += 1
                continue
                
            # Check iron condition
            iron_noise = _simplex(simplex, x, y, 2, 6) > config.iron_threshold
            iron_prob = np.random.uniform() > (1 - config.iron_probability)
            if iron_noise and iron_prob:
                stats['blocked_by_iron'] += 1
                continue
                
            # Check diamond threshold
            if mountain > config.diamond_threshold:
                stats['diamond_threshold_met'] += 1
                
                # Check if diamond probability passes
                if np.random.uniform() > (1 - config.diamond_probability):
                    # Check lava condition (which comes after diamond)
                    lava_check = mountain > 0.3 and _simplex(simplex, x, y, 6, 5) > config.lava_threshold
                    if not lava_check:
                        stats['diamonds_spawned'] += 1
                        stats['positions_checked'].append((x, y, mountain))
                    else:
                        stats['blocked_by_lava'] += 1
    
    return stats

def print_analysis(stats, config):
    """Print analysis results."""
    print("Diamond Spawning Analysis")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Mountain threshold: {config.mountain_threshold}")
    print(f"  Diamond threshold: {config.diamond_threshold}")
    print(f"  Diamond probability: {config.diamond_probability}")
    print(f"  Coal threshold: {config.coal_threshold}")
    print(f"  Coal probability: {config.coal_probability}")
    print(f"  Iron threshold: {config.iron_threshold}")
    print(f"  Iron probability: {config.iron_probability}")
    print()
    
    print(f"Results from {stats['total_positions']} positions:")
    print(f"  Mountain positions: {stats['mountain_positions']} ({100 * stats['mountain_positions'] / stats['total_positions']:.2f}%)")
    
    if stats['mountain_positions'] > 0:
        print(f"  Within mountain areas:")
        print(f"    Blocked by caves: {stats['blocked_by_caves']} ({100 * stats['blocked_by_caves'] / stats['mountain_positions']:.2f}%)")
        print(f"    Blocked by tunnels: {stats['blocked_by_tunnels']} ({100 * stats['blocked_by_tunnels'] / stats['mountain_positions']:.2f}%)")
        print(f"    Blocked by coal: {stats['blocked_by_coal']} ({100 * stats['blocked_by_coal'] / stats['mountain_positions']:.2f}%)")
        print(f"    Blocked by iron: {stats['blocked_by_iron']} ({100 * stats['blocked_by_iron'] / stats['mountain_positions']:.2f}%)")
        print(f"    Diamond threshold met: {stats['diamond_threshold_met']} ({100 * stats['diamond_threshold_met'] / stats['mountain_positions']:.2f}%)")
        
        if stats['diamond_threshold_met'] > 0:
            print(f"    Of positions meeting diamond threshold:")
            print(f"      Diamonds spawned: {stats['diamonds_spawned']} ({100 * stats['diamonds_spawned'] / stats['diamond_threshold_met']:.2f}%)")
            print(f"      Blocked by lava: {stats['blocked_by_lava']} ({100 * stats['blocked_by_lava'] / stats['diamond_threshold_met']:.2f}%)")
    
    print(f"\n  Overall diamond spawn rate: {stats['diamonds_spawned']} ({100 * stats['diamonds_spawned'] / stats['total_positions']:.4f}%)")
    
    # Mountain value statistics
    mountain_values = np.array(stats['mountain_values'])
    print(f"\nMountain value statistics:")
    print(f"  Min: {mountain_values.min():.3f}")
    print(f"  Max: {mountain_values.max():.3f}")
    print(f"  Mean: {mountain_values.mean():.3f}")
    print(f"  Std: {mountain_values.std():.3f}")
    print(f"  Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"    {p}th: {np.percentile(mountain_values, p):.3f}")

if __name__ == "__main__":
    # Test with default config
    print("Testing with default configuration:")
    config = WorldGenConfig()
    stats = analyze_diamond_conditions(config)
    print_analysis(stats, config)
    
    print("\n" + "=" * 50 + "\n")
    
    # Test with modified config
    print("Testing with increased diamond probability (0.5):")
    config_high_prob = WorldGenConfig(diamond_probability=0.5)
    stats_high = analyze_diamond_conditions(config_high_prob)
    print_analysis(stats_high, config_high_prob)
    
    print("\n" + "=" * 50 + "\n")
    
    # Test with lower diamond threshold
    print("Testing with lower diamond threshold (0.16):")
    config_low_threshold = WorldGenConfig(diamond_threshold=0.16)
    stats_low = analyze_diamond_conditions(config_low_threshold)
    print_analysis(stats_low, config_low_threshold)