#!/usr/bin/env python3
"""
Summary analysis of seed performance comparison between GPT-4.1-NANO and GPT-4.1-MINI.
"""

def print_summary():
    print("🎯 KEY FINDINGS: SEED PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print("\n📊 OVERALL PERFORMANCE:")
    print("  • GPT-4.1-MINI: 12 total achievements across 9 instances")
    print("  • GPT-4.1-NANO:  2 total achievements across 2 instances")
    print("  • MINI wins: 9 out of 10 instances (90% win rate)")
    print("  • NANO wins: 0 out of 10 instances")
    print("  • Ties: 1 instance (instance 9)")
    
    print("\n🏆 INSTANCE-BY-INSTANCE BREAKDOWN:")
    print("  • Instance 1 (Seed 43): MINI wins (collect_wood vs 0)")
    print("  • Instance 2 (Seed 44): MINI wins (collect_wood vs 0)")
    print("  • Instance 3 (Seed 45): MINI wins (collect_sapling vs 0)")
    print("  • Instance 4 (Seed 46): MINI wins (collect_wood vs 0)")
    print("  • Instance 5 (Seed 47): MINI wins (collect_wood vs 0)")
    print("  • Instance 6 (Seed 48): MINI wins (collect_sapling + eat_cow vs collect_sapling)")
    print("  • Instance 7 (Seed 49): MINI wins (collect_sapling + collect_wood vs 0)")
    print("  • Instance 8 (Seed 50): MINI wins (collect_wood vs 0)")
    print("  • Instance 9 (Seed 51): TIE (0 vs 0)")
    print("  • Instance 10 (Seed 52): MINI wins (collect_sapling + collect_wood vs collect_wood)")
    
    print("\n🎯 ACHIEVEMENT TYPE ANALYSIS:")
    print("  • collect_wood: MINI 7, NANO 1 (MINI dominates)")
    print("  • collect_sapling: MINI 4, NANO 1 (MINI dominates)")
    print("  • eat_cow: MINI 1, NANO 0 (MINI only)")
    print("  • All other achievements: 0 for both models")
    
    print("\n🔍 PATTERNS OBSERVED:")
    print("  1. MINI consistently outperforms NANO across almost all seeds")
    print("  2. MINI achieves more complex combinations (e.g., collect_sapling + eat_cow)")
    print("  3. NANO struggles with basic achievements (only 2 total vs MINI's 12)")
    print("  4. Both models struggle with advanced achievements (iron, diamond, etc.)")
    print("  5. MINI shows better exploration and resource gathering capabilities")
    
    print("\n📈 IMPLICATIONS:")
    print("  • MINI demonstrates significantly better reasoning and planning")
    print("  • MINI's larger context window may enable better multi-step planning")
    print("  • NANO may be hitting context limits or reasoning limitations")
    print("  • Both models struggle with complex crafting and combat achievements")
    print("  • The performance gap is consistent across different environment seeds")
    
    print("\n🎲 RANDOMNESS ANALYSIS:")
    print("  • Seeds 43-52 were tested (10 different environments)")
    print("  • MINI wins 9/10 = 90% win rate")
    print("  • This suggests the performance difference is robust, not random")
    print("  • Only instance 9 was a tie, suggesting MINI's advantage is consistent")

if __name__ == "__main__":
    print_summary() 