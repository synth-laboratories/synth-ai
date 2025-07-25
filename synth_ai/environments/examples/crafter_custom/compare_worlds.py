#!/usr/bin/env python3
"""
Compare agent performance across different world configurations.
"""

import subprocess
import json
from pathlib import Path
import sys

def run_evaluation(world_config, episodes=2, max_turns=20):
    """Run evaluation with a specific world configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {world_config.upper()} world configuration")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "-m",
        "synth_ai.environments.examples.crafter_custom.agent_demos.test_crafter_custom_agent",
        "--model", "gpt-4.1-nano",
        "--world-config", world_config,
        "--episodes", str(episodes),
        "--max-turns", str(max_turns),
        "--evaluate-traces"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract key metrics from output
    lines = result.stdout.split('\n')
    metrics = {}
    
    for line in lines:
        if "Mean Score" in line:
            metrics["mean_score"] = float(line.split()[-1])
        elif "Avg Achievements/Episode" in line:
            metrics["avg_achievements"] = float(line.split()[-1])
        elif "Average Score:" in line and "📊" not in line:
            metrics["trace_score"] = float(line.split()[-1])
    
    return metrics

def main():
    print("🎮 Crafter World Configuration Comparison")
    print("=" * 60)
    
    configs = ["peaceful", "easy", "normal", "hard"]
    results = {}
    
    for config in configs:
        results[config] = run_evaluation(config, episodes=2, max_turns=15)
        print(f"\nResults for {config}: {results[config]}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Config':<10} {'Mean Score':<12} {'Avg Achievements':<18} {'Trace Score':<12}")
    print("-"*60)
    
    for config in configs:
        r = results[config]
        print(f"{config:<10} {r.get('mean_score', 0):<12.2f} {r.get('avg_achievements', 0):<18.2f} {r.get('trace_score', 0):<12.2f}")
    
    print("\n💡 Analysis:")
    print("- Peaceful worlds should have highest scores (no enemies)")
    print("- Hard worlds should have lowest scores (many enemies, few resources)")
    print("- Resource availability directly impacts achievement unlocking")

if __name__ == "__main__":
    main()