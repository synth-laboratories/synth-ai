#!/usr/bin/env python3
"""
Run Full Continual Learning Comparison on Banking77

This script runs both approaches and generates a comparison:
1. Classic GEPA (cold start and warm start)
2. MIPRO Continual Learning

Usage:
    uv run python run_comparison.py
    uv run python run_comparison.py --rollouts-per-split 100 --model gpt-4.1-nano
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

from data_splits import get_split_size


def run_classic_gepa(
    rollouts: int,
    train_size: int,
    model: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Run classic GEPA experiment."""
    print("\n" + "="*70)
    print("RUNNING CLASSIC GEPA (Cold Start + Warm Start)")
    print("="*70)
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_classic_gepa.py"),
        "--rollouts", str(rollouts),
        "--train-size", str(train_size),
        "--model", model,
        "--output", str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Classic GEPA failed with return code {result.returncode}")
        return {}
    
    # Load results
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return {}


def run_mipro_continual(
    rollouts_per_split: int,
    train_size: int,
    model: str,
    output_path: Path,
) -> Dict[str, Any]:
    """Run MIPRO continual learning experiment."""
    print("\n" + "="*70)
    print("RUNNING MIPRO CONTINUAL LEARNING")
    print("="*70)
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_mipro_continual.py"),
        "--rollouts-per-split", str(rollouts_per_split),
        "--train-size", str(train_size),
        "--model", model,
        "--output", str(output_path),
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"MIPRO Continual failed with return code {result.returncode}")
        return {}
    
    # Load results
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return {}


def format_accuracy(value: float | None) -> str:
    """Format accuracy as percentage string."""
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def generate_comparison_table(
    classic_results: Dict[str, Any],
    continual_results: Dict[str, Any],
) -> str:
    """Generate a markdown comparison table."""
    lines = []
    lines.append("\n## Comparison Results\n")
    lines.append("| Split | Intents | Cold Start | Warm Start | MIPRO Continual | Best Method |")
    lines.append("|-------|---------|------------|------------|-----------------|-------------|")
    
    for split_num in [1, 2, 3, 4]:
        intents = get_split_size(split_num)
        
        # Get Classic GEPA results
        classic_split = classic_results.get("splits", {}).get(str(split_num), {})
        cold_result = classic_split.get("cold_start", {})
        warm_result = classic_split.get("warm_start", {})
        cold_acc = cold_result.get("best_reward")
        warm_acc = warm_result.get("best_reward") if warm_result else None
        
        # Get MIPRO Continual results
        continual_split = continual_results.get("split_results", {}).get(str(split_num), {})
        continual_acc = continual_split.get("accuracy")
        
        # Determine best method
        values = [
            ("Cold Start", cold_acc),
            ("Warm Start", warm_acc),
            ("MIPRO Continual", continual_acc),
        ]
        valid_values = [(name, v) for name, v in values if v is not None]
        if valid_values:
            best_name, best_val = max(valid_values, key=lambda x: x[1])
        else:
            best_name = "N/A"
        
        # Format row
        cold_str = format_accuracy(cold_acc)
        warm_str = format_accuracy(warm_acc) if split_num > 1 else "-"
        continual_str = format_accuracy(continual_acc)
        
        lines.append(f"| Split {split_num} | {intents} | {cold_str} | {warm_str} | {continual_str} | {best_name} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_summary(
    classic_results: Dict[str, Any],
    continual_results: Dict[str, Any],
) -> str:
    """Generate a text summary of the comparison."""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("EXPERIMENT SUMMARY")
    lines.append("="*70)
    
    # Hypothesis evaluation
    lines.append("\n### Hypothesis Evaluation")
    lines.append("Hypothesis: Continual learning fares better in later splits")
    lines.append("")
    
    # Check if continual learning beats cold start on later splits
    wins_vs_cold = 0
    wins_vs_warm = 0
    
    for split_num in [2, 3, 4]:  # Exclude split 1 where warm start isn't applicable
        classic_split = classic_results.get("splits", {}).get(str(split_num), {})
        continual_split = continual_results.get("split_results", {}).get(str(split_num), {})
        
        cold_acc = classic_split.get("cold_start", {}).get("best_reward")
        warm_acc = classic_split.get("warm_start", {}).get("best_reward")
        continual_acc = continual_split.get("accuracy")
        
        if continual_acc is not None and cold_acc is not None:
            if continual_acc > cold_acc:
                wins_vs_cold += 1
        if continual_acc is not None and warm_acc is not None:
            if continual_acc > warm_acc:
                wins_vs_warm += 1
    
    lines.append(f"MIPRO Continual beats Cold Start on {wins_vs_cold}/3 later splits")
    lines.append(f"MIPRO Continual beats Warm Start on {wins_vs_warm}/3 later splits")
    
    if wins_vs_cold >= 2:
        lines.append("\n✓ HYPOTHESIS SUPPORTED: Continual learning shows advantage on later splits")
    else:
        lines.append("\n✗ HYPOTHESIS NOT SUPPORTED: Continual learning did not consistently outperform")
    
    # Timing comparison
    lines.append("\n### Timing Comparison")
    classic_total = 0
    for split_data in classic_results.get("splits", {}).values():
        for variant in ["cold_start", "warm_start"]:
            if variant in split_data and split_data[variant]:
                classic_total += split_data[variant].get("elapsed_seconds", 0)
    
    continual_total = continual_results.get("total_elapsed_seconds", 0)
    
    lines.append(f"Classic GEPA total time: {classic_total:.1f}s")
    lines.append(f"MIPRO Continual total time: {continual_total:.1f}s")
    
    lines.append("\n" + "="*70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run Full Continual Learning Comparison")
    parser.add_argument("--rollouts-per-split", type=int, default=100, help="Rollouts per split")
    parser.add_argument("--train-size", type=int, default=30, help="Training seeds count")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--skip-classic", action="store_true", help="Skip classic GEPA (use existing results)")
    parser.add_argument("--skip-continual", action="store_true", help="Skip MIPRO continual (use existing results)")
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print("="*70)
    print("CONTINUAL LEARNING COMPARISON: Banking77")
    print("="*70)
    print(f"  Rollouts per split: {args.rollouts_per_split}")
    print(f"  Train size: {args.train_size}")
    print(f"  Model: {args.model}")
    print(f"  Output directory: {output_dir}")
    print("="*70)
    
    start_time = time.time()
    
    # Run Classic GEPA
    classic_output = output_dir / f"classic_gepa_{timestamp}.json"
    if args.skip_classic:
        # Try to load existing results
        existing = list(output_dir.glob("classic_gepa_*.json"))
        if existing:
            classic_output = max(existing, key=lambda p: p.stat().st_mtime)
            print(f"\nUsing existing Classic GEPA results: {classic_output}")
            with open(classic_output) as f:
                classic_results = json.load(f)
        else:
            print("\nNo existing Classic GEPA results found. Running...")
            classic_results = run_classic_gepa(
                rollouts=args.rollouts_per_split,
                train_size=args.train_size,
                model=args.model,
                output_path=classic_output,
            )
    else:
        classic_results = run_classic_gepa(
            rollouts=args.rollouts_per_split,
            train_size=args.train_size,
            model=args.model,
            output_path=classic_output,
        )
    
    # Run MIPRO Continual
    continual_output = output_dir / f"mipro_continual_{timestamp}.json"
    if args.skip_continual:
        # Try to load existing results
        existing = list(output_dir.glob("mipro_continual_*.json"))
        if existing:
            continual_output = max(existing, key=lambda p: p.stat().st_mtime)
            print(f"\nUsing existing MIPRO Continual results: {continual_output}")
            with open(continual_output) as f:
                continual_results = json.load(f)
        else:
            print("\nNo existing MIPRO Continual results found. Running...")
            continual_results = run_mipro_continual(
                rollouts_per_split=args.rollouts_per_split,
                train_size=args.train_size,
                model=args.model,
                output_path=continual_output,
            )
    else:
        continual_results = run_mipro_continual(
            rollouts_per_split=args.rollouts_per_split,
            train_size=args.train_size,
            model=args.model,
            output_path=continual_output,
        )
    
    total_elapsed = time.time() - start_time
    
    # Generate comparison table and summary
    table = generate_comparison_table(classic_results, continual_results)
    summary = generate_summary(classic_results, continual_results)
    
    print(table)
    print(summary)
    
    # Save combined results
    combined_output = output_dir / f"comparison_{timestamp}.json"
    combined_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_elapsed,
        "config": {
            "rollouts_per_split": args.rollouts_per_split,
            "train_size": args.train_size,
            "model": args.model,
        },
        "classic_gepa": classic_results,
        "mipro_continual": continual_results,
    }
    
    with open(combined_output, "w") as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nCombined results saved to: {combined_output}")
    
    # Update README with results
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path) as f:
            readme_content = f.read()
        
        # Replace the results section
        marker_start = "## Results\n"
        marker_end = "\n## Usage"
        
        if marker_start in readme_content and marker_end in readme_content:
            before = readme_content.split(marker_start)[0]
            after = readme_content.split(marker_end)[1]
            
            new_readme = before + marker_start + table + summary + "\n" + marker_end + after
            
            with open(readme_path, "w") as f:
                f.write(new_readme)
            print(f"README.md updated with results")
    
    print(f"\nTotal comparison time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
