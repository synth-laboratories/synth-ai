"""
Summary of the diamond spawning issue in Crafter.
"""

def explain_probability_condition():
    """Explain how the probability condition works."""
    print("DIAMOND SPAWNING PROBABILITY EXPLANATION")
    print("=" * 60)
    print()
    print("The code uses this condition to spawn diamonds:")
    print("  uniform() > (1 - config.diamond_probability)")
    print()
    print("This means:")
    print("  - uniform() generates a random number between 0 and 1")
    print("  - We spawn a diamond if this random number is GREATER than (1 - diamond_probability)")
    print()
    print("Examples:")
    print("-" * 60)
    
    probabilities = [0.006, 0.1, 0.5, 0.8, 0.95]
    
    for prob in probabilities:
        threshold = 1 - prob
        actual_chance = 1 - threshold
        print(f"\nWith diamond_probability = {prob}:")
        print(f"  Condition: uniform() > {threshold:.3f}")
        print(f"  Actual spawn chance: {actual_chance:.1%}")
        print(f"  Description: {'VERY LOW' if prob < 0.1 else 'LOW' if prob < 0.3 else 'MODERATE' if prob < 0.7 else 'HIGH' if prob < 0.9 else 'VERY HIGH'}")

def show_cascading_conditions():
    """Show how conditions cascade in the world generation."""
    print("\n\nCASCADING CONDITIONS IN WORLD GENERATION")
    print("=" * 60)
    print()
    print("For a position in a mountain area, materials are checked in this order:")
    print()
    print("1. Cave check (two conditions)")
    print("   └─ If true → place 'path'")
    print()
    print("2. Horizontal tunnel check") 
    print("   └─ If true → place 'path'")
    print()
    print("3. Vertical tunnel check")
    print("   └─ If true → place 'path'")
    print()
    print("4. Coal check (noise threshold AND probability)")
    print("   └─ If true → place 'coal'")
    print()
    print("5. Iron check (noise threshold AND probability)")
    print("   └─ If true → place 'iron'")
    print()
    print("6. Diamond check (mountain threshold AND probability)")
    print("   └─ If true → place 'diamond'")
    print()
    print("7. Lava check")
    print("   └─ If true → place 'lava'")
    print()
    print("8. Default: place 'stone'")
    print()
    print("IMPORTANT: Once any condition is met, no further checks are done!")
    print("This means diamonds can be blocked by caves, tunnels, coal, or iron.")

def calculate_effective_spawn_rate():
    """Calculate the effective diamond spawn rate."""
    print("\n\nEFFECTIVE DIAMOND SPAWN RATE")
    print("=" * 60)
    print()
    print("Based on the analysis of 10,000 positions:")
    print()
    
    # Data from our analysis
    total_positions = 10000
    mountain_positions = 2402
    blocked_caves = 431
    blocked_tunnels = 445
    blocked_coal = 104
    blocked_iron = 41
    diamond_threshold_met = 1185
    diamond_probability = 0.006
    
    print(f"Total positions: {total_positions}")
    print(f"Mountain positions: {mountain_positions} ({100*mountain_positions/total_positions:.1f}%)")
    print(f"  Blocked by caves: {blocked_caves}")
    print(f"  Blocked by tunnels: {blocked_tunnels}")
    print(f"  Blocked by coal: {blocked_coal}")
    print(f"  Blocked by iron: {blocked_iron}")
    print(f"  Reached diamond check: {diamond_threshold_met}")
    print()
    print(f"Diamond spawn probability: {diamond_probability} ({diamond_probability*100:.1f}%)")
    print()
    
    expected_diamonds = diamond_threshold_met * diamond_probability
    overall_rate = expected_diamonds / total_positions
    
    print(f"Expected diamonds: {expected_diamonds:.1f}")
    print(f"Overall spawn rate: {overall_rate:.4%}")
    print()
    print("This explains why diamonds are so rare!")

if __name__ == "__main__":
    explain_probability_condition()
    show_cascading_conditions()
    calculate_effective_spawn_rate()
    
    print("\n\nRECOMMENDATION")
    print("=" * 60)
    print()
    print("To make diamonds appear with 'high probability', you should:")
    print()
    print("1. Increase diamond_probability to at least 0.5 (50% chance)")
    print("   or even 0.8-0.9 for truly 'high' probability")
    print()
    print("2. Consider adjusting other thresholds to reduce blocking:")
    print("   - Increase coal_threshold (currently 0.0)")
    print("   - Increase iron_threshold (currently 0.4)")
    print("   - Decrease coal_probability and iron_probability")
    print()
    print("3. Optionally lower diamond_threshold slightly (currently 0.18)")
    print("   to increase the number of positions that can spawn diamonds")