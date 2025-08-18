#!/usr/bin/env python3
"""
Trace evaluation functions for Crafter episodes.
Scores traces based on achievements and invalid actions.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Scoring weights
WEIGHTS = {
    'easy_achievement': 1.0,      # Easy achievement (e.g., collect_wood)
    'medium_achievement': 2.5,    # Medium achievement (e.g., make_wood_pickaxe)
    'hard_achievement': 5.0,      # Hard achievement (e.g., make_iron_sword)
    'invalid_action': -0.05,      # Invalid action penalty (50 invalid = -1 medium achievement)
}

# Map hook names to scoring categories
HOOK_TO_SCORE_TYPE = {
    'easy_achievement': 'easy_achievement',
    'medium_achievement': 'medium_achievement', 
    'hard_achievement': 'hard_achievement',
    'invalid_action': 'invalid_action'
}


def evaluate_event(event: Dict[str, Any]) -> Tuple[float, str]:
    """
    Evaluate a single event based on its hooks.
    Returns: (score, symbol) where symbol is '+', '-', or '0'
    """
    score = 0.0
    symbol = '0'
    
    # Check if event has metadata from hooks
    event_metadata = event.get('event_metadata', [])
    
    for metadata in event_metadata:
        hook_name = metadata.get('hook_name', '')
        
        if hook_name in HOOK_TO_SCORE_TYPE:
            score_type = HOOK_TO_SCORE_TYPE[hook_name]
            weight = WEIGHTS[score_type]
            score += weight
            
            # Determine symbol
            if weight > 0:
                symbol = '+'
            elif weight < 0:
                symbol = '-'
    
    return score, symbol


def evaluate_trace(trace_path: Path) -> Dict[str, Any]:
    """
    Evaluate an entire trace file.
    Returns detailed scoring breakdown and trajectory visualization.
    """
    with open(trace_path, 'r') as f:
        trace_data = json.load(f)
    
    # Track counts
    counts = defaultdict(int)
    total_score = 0.0
    trajectory_symbols = []
    
    # Process event history
    event_history = trace_data.get('event_history', [])
    
    for event in event_history:
        event_score, symbol = evaluate_event(event)
        total_score += event_score
        
        # Only add symbol if score is non-zero
        if event_score != 0:
            trajectory_symbols.append(symbol)
        
        # Count hook types
        for metadata in event.get('event_metadata', []):
            hook_name = metadata.get('hook_name', '')
            if hook_name in HOOK_TO_SCORE_TYPE:
                score_type = HOOK_TO_SCORE_TYPE[hook_name]
                counts[score_type] += 1
    
    # Create trajectory string
    trajectory_str = ''.join(trajectory_symbols) if trajectory_symbols else '(no scored events)'
    
    return {
        'total_score': total_score,
        'counts': dict(counts),
        'trajectory': trajectory_str,
        'num_events': len(event_history),
        'trace_file': trace_path.name
    }


def print_trace_evaluation(eval_result: Dict[str, Any]):
    """Print a formatted evaluation result for a single trace."""
    print(f"\n📊 Trace: {eval_result['trace_file']}")
    print(f"   Score: {eval_result['total_score']:.2f}")
    print(f"   Events: {eval_result['num_events']}")
    
    counts = eval_result['counts']
    if counts:
        print("   Breakdown:")
        if 'easy_achievement' in counts:
            print(f"     Easy achievements: {counts['easy_achievement']} × {WEIGHTS['easy_achievement']} = {counts['easy_achievement'] * WEIGHTS['easy_achievement']:.2f}")
        if 'medium_achievement' in counts:
            print(f"     Medium achievements: {counts['medium_achievement']} × {WEIGHTS['medium_achievement']} = {counts['medium_achievement'] * WEIGHTS['medium_achievement']:.2f}")
        if 'hard_achievement' in counts:
            print(f"     Hard achievements: {counts['hard_achievement']} × {WEIGHTS['hard_achievement']} = {counts['hard_achievement'] * WEIGHTS['hard_achievement']:.2f}")
        if 'invalid_action' in counts:
            print(f"     Invalid actions: {counts['invalid_action']} × {WEIGHTS['invalid_action']} = {counts['invalid_action'] * WEIGHTS['invalid_action']:.2f}")
    
    print(f"   Trajectory: {eval_result['trajectory']}")


def evaluate_all_traces(trace_dir: Path, pattern: str = "*.json") -> List[Dict[str, Any]]:
    """
    Evaluate all trace files in a directory.
    Returns list of evaluation results sorted by score.
    """
    trace_files = list(trace_dir.glob(pattern))
    results = []
    
    for trace_file in trace_files:
        try:
            result = evaluate_trace(trace_file)
            results.append(result)
        except Exception as e:
            print(f"⚠️  Error evaluating {trace_file.name}: {e}")
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['total_score'], reverse=True)
    
    return results


def print_evaluation_summary(results: List[Dict[str, Any]]):
    """Print a summary of all trace evaluations."""
    if not results:
        print("No traces to evaluate.")
        return
    
    print("\n" + "=" * 80)
    print("📈 TRACE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Rank':<6} {'Score':<10} {'Trajectory':<50} {'File':<30}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        trajectory = result['trajectory']
        if len(trajectory) > 50:
            trajectory = trajectory[:47] + "..."
        print(f"{i:<6} {result['total_score']:<10.2f} {trajectory:<50} {result['trace_file'][:30]:<30}")
    
    print("-" * 80)
    
    # Summary statistics
    scores = [r['total_score'] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    
    print(f"Average Score: {avg_score:.2f}")
    print(f"Best Score: {max_score:.2f}")
    print(f"Worst Score: {min_score:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Crafter trace files")
    parser.add_argument("trace_path", type=str, help="Directory containing trace files or single trace file")
    parser.add_argument("--pattern", type=str, default="*.json", help="File pattern to match (for directories)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed evaluation for each trace")
    
    args = parser.parse_args()
    
    trace_path = Path(args.trace_path)
    if not trace_path.exists():
        print(f"❌ Path not found: {trace_path}")
        exit(1)
    
    # Check if it's a file or directory
    if trace_path.is_file():
        print(f"🔍 Evaluating single trace: {trace_path}")
        result = evaluate_trace(trace_path)
        print_trace_evaluation(result)
    else:
        print(f"🔍 Evaluating traces in: {trace_path}")
        results = evaluate_all_traces(trace_path, args.pattern)
        
        if args.verbose:
            for result in results:
                print_trace_evaluation(result)
        
        print_evaluation_summary(results)