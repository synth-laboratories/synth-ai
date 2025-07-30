#!/usr/bin/env python3
"""
Evaluate traces grouped by difficulty level.
"""

import json
from pathlib import Path
from collections import defaultdict
from trace_eval import evaluate_trace, WEIGHTS

def get_trace_difficulty(trace_path: Path) -> str:
    """Extract difficulty from trace metadata."""
    try:
        with open(trace_path, 'r') as f:
            data = json.load(f)
        
        # Try to find difficulty in metadata
        metadata = data.get('metadata', {})
        if 'difficulty' in metadata:
            return metadata['difficulty']
        
        # Try to find in task instance metadata
        if 'task_instance' in metadata:
            task_metadata = metadata['task_instance'].get('metadata', {})
            if 'difficulty' in task_metadata:
                return task_metadata['difficulty']
        
        return 'unknown'
    except:
        return 'unknown'

def main():
    traces_dir = Path("traces")
    if not traces_dir.exists():
        print(f"Traces directory not found: {traces_dir}")
        return
    
    # Group traces by difficulty
    traces_by_difficulty = defaultdict(list)
    
    for trace_file in traces_dir.glob("*.json"):
        difficulty = get_trace_difficulty(trace_file)
        result = evaluate_trace(trace_file)
        traces_by_difficulty[difficulty].append(result)
    
    # Sort difficulties
    difficulty_order = ['easy', 'medium', 'hard', 'unknown']
    
    print("=" * 80)
    print("CRAFTER EVALUATION BY DIFFICULTY")
    print("=" * 80)
    
    for difficulty in difficulty_order:
        traces = traces_by_difficulty[difficulty]
        if not traces:
            continue
        
        print(f"\n{difficulty.upper()} ({len(traces)} traces)")
        print("-" * 40)
        
        # Calculate statistics
        scores = [t['total_score'] for t in traces]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        min_score = min(scores) if scores else 0
        
        # Count achievements and invalid actions
        total_easy = sum(t['counts'].get('easy_achievement', 0) for t in traces)
        total_medium = sum(t['counts'].get('medium_achievement', 0) for t in traces)
        total_hard = sum(t['counts'].get('hard_achievement', 0) for t in traces)
        total_invalid = sum(t['counts'].get('invalid_action', 0) for t in traces)
        
        print(f"Average Score: {avg_score:.2f}")
        print(f"Score Range: {min_score:.2f} to {max_score:.2f}")
        print(f"\nAchievements per trace:")
        print(f"  Easy:   {total_easy / len(traces):.2f}")
        print(f"  Medium: {total_medium / len(traces):.2f}")
        print(f"  Hard:   {total_hard / len(traces):.2f}")
        print(f"\nInvalid actions per trace: {total_invalid / len(traces):.2f}")
        
        # Show score distribution
        positive_scores = [s for s in scores if s > 0]
        zero_scores = [s for s in scores if s == 0]
        negative_scores = [s for s in scores if s < 0]
        
        print(f"\nScore distribution:")
        print(f"  Positive: {len(positive_scores)} ({len(positive_scores)/len(scores)*100:.1f}%)")
        print(f"  Zero:     {len(zero_scores)} ({len(zero_scores)/len(scores)*100:.1f}%)")
        print(f"  Negative: {len(negative_scores)} ({len(negative_scores)/len(scores)*100:.1f}%)")
        
        # Show top 3 traces
        traces_sorted = sorted(traces, key=lambda x: x['total_score'], reverse=True)
        print(f"\nTop 3 traces:")
        for i, trace in enumerate(traces_sorted[:3], 1):
            print(f"  {i}. Score: {trace['total_score']:.2f}, Trajectory: {trace['trajectory'][:50]}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    all_traces = []
    for traces in traces_by_difficulty.values():
        all_traces.extend(traces)
    
    if all_traces:
        all_scores = [t['total_score'] for t in all_traces]
        print(f"Total traces evaluated: {len(all_traces)}")
        print(f"Overall average score: {sum(all_scores) / len(all_scores):.2f}")
        
        # Achievement type distribution
        total_achievements = defaultdict(int)
        for trace in all_traces:
            for achievement_type in ['easy_achievement', 'medium_achievement', 'hard_achievement']:
                total_achievements[achievement_type] += trace['counts'].get(achievement_type, 0)
        
        print(f"\nTotal achievements unlocked:")
        print(f"  Easy:   {total_achievements['easy_achievement']} (worth {total_achievements['easy_achievement'] * WEIGHTS['easy_achievement']:.1f} points)")
        print(f"  Medium: {total_achievements['medium_achievement']} (worth {total_achievements['medium_achievement'] * WEIGHTS['medium_achievement']:.1f} points)")  
        print(f"  Hard:   {total_achievements['hard_achievement']} (worth {total_achievements['hard_achievement'] * WEIGHTS['hard_achievement']:.1f} points)")
        
        total_invalid = sum(t['counts'].get('invalid_action', 0) for t in all_traces)
        print(f"\nTotal invalid actions: {total_invalid} (penalty: {total_invalid * WEIGHTS['invalid_action']:.1f} points)")

if __name__ == "__main__":
    main()