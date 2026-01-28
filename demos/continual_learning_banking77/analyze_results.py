#!/usr/bin/env python3
"""
Analyze Continual Learning Comparison Results

This script analyzes existing results and generates comparison tables/reports.

Usage:
    uv run python analyze_results.py
    uv run python analyze_results.py --results-dir ./results
    uv run python analyze_results.py --classic results/classic_gepa.json --continual results/mipro_continual.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

from data_splits import get_split_size


def load_latest_results(results_dir: Path, prefix: str) -> Optional[Dict[str, Any]]:
    """Load the most recent results file with the given prefix."""
    files = list(results_dir.glob(f"{prefix}*.json"))
    if not files:
        return None
    latest = max(files, key=lambda p: p.stat().st_mtime)
    print(f"Loading {prefix} results from: {latest}")
    with open(latest) as f:
        return json.load(f)


def format_accuracy(value: float | None) -> str:
    """Format accuracy as percentage string."""
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def analyze_classic_gepa(results: Dict[str, Any]) -> None:
    """Analyze Classic GEPA results."""
    print("\n" + "="*70)
    print("CLASSIC GEPA ANALYSIS")
    print("="*70)
    
    splits = results.get("splits", {})
    
    print(f"\n{'Split':<12} {'Intents':<10} {'Cold Start':<15} {'Warm Start':<15} {'Δ (Warm-Cold)':<15}")
    print("-"*70)
    
    for split_num in [1, 2, 3, 4]:
        split_data = splits.get(str(split_num), {})
        intents = get_split_size(split_num)
        
        cold = split_data.get("cold_start", {})
        warm = split_data.get("warm_start", {})
        
        cold_acc = cold.get("best_reward")
        warm_acc = warm.get("best_reward") if warm else None
        
        cold_str = format_accuracy(cold_acc)
        warm_str = format_accuracy(warm_acc) if split_num > 1 else "-"
        
        if cold_acc is not None and warm_acc is not None:
            diff = warm_acc - cold_acc
            diff_str = f"{diff:+.1%}"
        else:
            diff_str = "-"
        
        print(f"Split {split_num:<6} {intents:<10} {cold_str:<15} {warm_str:<15} {diff_str:<15}")
    
    # Summary statistics
    print("\n" + "-"*70)
    warm_wins = 0
    cold_wins = 0
    
    for split_num in [2, 3, 4]:
        split_data = splits.get(str(split_num), {})
        cold_acc = split_data.get("cold_start", {}).get("best_reward")
        warm_acc = split_data.get("warm_start", {}).get("best_reward")
        
        if cold_acc is not None and warm_acc is not None:
            if warm_acc > cold_acc:
                warm_wins += 1
            elif cold_acc > warm_acc:
                cold_wins += 1
    
    print(f"\nWarm Start vs Cold Start (splits 2-4):")
    print(f"  Warm Start wins: {warm_wins}")
    print(f"  Cold Start wins: {cold_wins}")
    print(f"  Ties: {3 - warm_wins - cold_wins}")


def analyze_mipro_continual(results: Dict[str, Any]) -> None:
    """Analyze MIPRO Continual results."""
    print("\n" + "="*70)
    print("MIPRO CONTINUAL LEARNING ANALYSIS")
    print("="*70)
    
    split_results = results.get("split_results", {})
    checkpoints = results.get("checkpoints", [])
    
    print(f"\n{'Split':<12} {'Intents':<10} {'Accuracy':<15} {'Candidates':<12} {'Proposals':<12}")
    print("-"*70)
    
    for checkpoint in checkpoints:
        split_num = checkpoint.get("split", 0)
        intents = checkpoint.get("num_intents", 0)
        accuracy = checkpoint.get("split_accuracy")
        ontology = checkpoint.get("ontology", {})
        num_candidates = ontology.get("num_candidates", 0)
        num_proposals = ontology.get("proposal_seq", 0)
        
        acc_str = format_accuracy(accuracy)
        
        print(f"Split {split_num:<6} {intents:<10} {acc_str:<15} {num_candidates:<12} {num_proposals:<12}")
    
    # Ontology growth analysis
    print("\n" + "-"*70)
    print("Ontology Growth:")
    
    if checkpoints:
        first = checkpoints[0].get("ontology", {})
        last = checkpoints[-1].get("ontology", {})
        
        print(f"  Initial candidates: {first.get('num_candidates', 0)}")
        print(f"  Final candidates: {last.get('num_candidates', 0)}")
        print(f"  Total proposals: {last.get('proposal_seq', 0)}")
    
    # Best candidate evolution
    print("\nBest Candidate Evolution:")
    for checkpoint in checkpoints:
        split_num = checkpoint.get("split", 0)
        best_id = checkpoint.get("best_candidate_id", "N/A")
        best_text = checkpoint.get("best_candidate_text", "")
        preview = best_text[:100] + "..." if best_text and len(best_text) > 100 else best_text
        print(f"  Split {split_num}: {best_id}")
        if preview:
            print(f"    {preview}")


def compare_methods(classic: Dict[str, Any], continual: Dict[str, Any]) -> None:
    """Compare Classic GEPA vs MIPRO Continual."""
    print("\n" + "="*70)
    print("COMPARISON: CLASSIC GEPA VS MIPRO CONTINUAL")
    print("="*70)
    
    classic_splits = classic.get("splits", {})
    continual_splits = continual.get("split_results", {})
    
    print(f"\n{'Split':<12} {'Intents':<8} {'Cold':<10} {'Warm':<10} {'Continual':<10} {'Best':<15}")
    print("-"*70)
    
    method_wins = {"cold": 0, "warm": 0, "continual": 0}
    
    for split_num in [1, 2, 3, 4]:
        intents = get_split_size(split_num)
        
        # Get accuracies
        classic_split = classic_splits.get(str(split_num), {})
        cold_acc = classic_split.get("cold_start", {}).get("best_reward")
        warm_acc = classic_split.get("warm_start", {}).get("best_reward") if classic_split.get("warm_start") else None
        
        continual_split = continual_splits.get(str(split_num), {})
        continual_acc = continual_split.get("accuracy")
        
        # Determine best
        values = [
            ("Cold", cold_acc),
            ("Warm", warm_acc),
            ("Continual", continual_acc),
        ]
        valid_values = [(name, v) for name, v in values if v is not None]
        if valid_values:
            best_name, best_val = max(valid_values, key=lambda x: x[1])
            if best_name == "Cold":
                method_wins["cold"] += 1
            elif best_name == "Warm":
                method_wins["warm"] += 1
            else:
                method_wins["continual"] += 1
        else:
            best_name = "N/A"
        
        # Format strings
        cold_str = format_accuracy(cold_acc)
        warm_str = format_accuracy(warm_acc) if split_num > 1 else "-"
        cont_str = format_accuracy(continual_acc)
        
        print(f"Split {split_num:<6} {intents:<8} {cold_str:<10} {warm_str:<10} {cont_str:<10} {best_name:<15}")
    
    # Summary
    print("\n" + "-"*70)
    print("Method Wins (per split):")
    print(f"  Cold Start: {method_wins['cold']}")
    print(f"  Warm Start: {method_wins['warm']}")
    print(f"  MIPRO Continual: {method_wins['continual']}")
    
    # Hypothesis evaluation
    print("\n" + "-"*70)
    print("HYPOTHESIS EVALUATION")
    print("-"*70)
    print("Hypothesis: Continual learning fares better in later splits")
    
    # Check continual vs cold on splits 3-4 (later splits)
    later_continual_wins = 0
    for split_num in [3, 4]:
        classic_split = classic_splits.get(str(split_num), {})
        cold_acc = classic_split.get("cold_start", {}).get("best_reward")
        continual_acc = continual_splits.get(str(split_num), {}).get("accuracy")
        
        if continual_acc is not None and cold_acc is not None:
            if continual_acc > cold_acc:
                later_continual_wins += 1
    
    print(f"\nMIPRO Continual beats Cold Start on {later_continual_wins}/2 later splits (3-4)")
    
    if later_continual_wins >= 1:
        print("✓ HYPOTHESIS PARTIALLY SUPPORTED")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED")


def generate_markdown_table(classic: Dict[str, Any], continual: Dict[str, Any]) -> str:
    """Generate markdown comparison table."""
    lines = []
    lines.append("| Split | Intents | Cold Start | Warm Start | MIPRO Continual | Best Method |")
    lines.append("|-------|---------|------------|------------|-----------------|-------------|")
    
    classic_splits = classic.get("splits", {})
    continual_splits = continual.get("split_results", {})
    
    for split_num in [1, 2, 3, 4]:
        intents = get_split_size(split_num)
        
        classic_split = classic_splits.get(str(split_num), {})
        cold_acc = classic_split.get("cold_start", {}).get("best_reward")
        warm_acc = classic_split.get("warm_start", {}).get("best_reward") if classic_split.get("warm_start") else None
        
        continual_split = continual_splits.get(str(split_num), {})
        continual_acc = continual_split.get("accuracy")
        
        # Determine best
        values = [
            ("Cold Start", cold_acc),
            ("Warm Start", warm_acc),
            ("MIPRO Continual", continual_acc),
        ]
        valid_values = [(name, v) for name, v in values if v is not None]
        best_name = max(valid_values, key=lambda x: x[1])[0] if valid_values else "N/A"
        
        cold_str = format_accuracy(cold_acc)
        warm_str = format_accuracy(warm_acc) if split_num > 1 else "-"
        cont_str = format_accuracy(continual_acc)
        
        lines.append(f"| Split {split_num} | {intents} | {cold_str} | {warm_str} | {cont_str} | {best_name} |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Continual Learning Results")
    parser.add_argument("--results-dir", type=str, default=None, help="Results directory")
    parser.add_argument("--classic", type=str, default=None, help="Classic GEPA results file")
    parser.add_argument("--continual", type=str, default=None, help="MIPRO Continual results file")
    parser.add_argument("--markdown", action="store_true", help="Output markdown table")
    args = parser.parse_args()
    
    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent / "results"
    
    # Load results
    if args.classic:
        with open(args.classic) as f:
            classic_results = json.load(f)
    else:
        classic_results = load_latest_results(results_dir, "classic_gepa")
    
    if args.continual:
        with open(args.continual) as f:
            continual_results = json.load(f)
    else:
        continual_results = load_latest_results(results_dir, "mipro_continual")
    
    # Analyze results
    if classic_results:
        analyze_classic_gepa(classic_results)
    else:
        print("No Classic GEPA results found")
    
    if continual_results:
        analyze_mipro_continual(continual_results)
    else:
        print("No MIPRO Continual results found")
    
    if classic_results and continual_results:
        compare_methods(classic_results, continual_results)
        
        if args.markdown:
            print("\n" + "="*70)
            print("MARKDOWN TABLE")
            print("="*70)
            print(generate_markdown_table(classic_results, continual_results))


if __name__ == "__main__":
    main()
