#!/usr/bin/env python3
"""
Example of using the trace evaluation system programmatically.
"""

from pathlib import Path
from trace_eval import evaluate_trace, evaluate_all_traces, print_trace_evaluation, print_evaluation_summary

def main():
    # Example 1: Evaluate a single trace
    print("=" * 60)
    print("Example 1: Evaluating a single trace")
    print("=" * 60)
    
    # Pick a high-scoring trace
    trace_path = Path("traces/session_crafter_episode_1_f2cea96d-34b6-46a3-9991-fe74ef263462_20250724_162140.json")
    if trace_path.exists():
        result = evaluate_trace(trace_path)
        print_trace_evaluation(result)
    else:
        print(f"Trace file not found: {trace_path}")
    
    # Example 2: Evaluate all traces and show top 5
    print("\n" + "=" * 60)
    print("Example 2: Top 5 traces by score")
    print("=" * 60)
    
    traces_dir = Path("traces")
    if traces_dir.exists():
        all_results = evaluate_all_traces(traces_dir)
        
        # Show only top 5
        print(f"\nFound {len(all_results)} traces. Showing top 5:\n")
        for i, result in enumerate(all_results[:5], 1):
            print(f"{i}. {result['trace_file']}")
            print(f"   Score: {result['total_score']:.2f}")
            print(f"   Trajectory: {result['trajectory']}")
            if result['counts']:
                print("   Breakdown:")
                for score_type, count in result['counts'].items():
                    weight = {
                        'easy_achievement': 1.0,
                        'medium_achievement': 2.5,
                        'hard_achievement': 5.0,
                        'invalid_action': -0.05
                    }[score_type]
                    print(f"     {score_type}: {count} × {weight} = {count * weight:.2f}")
            print()
    
    # Example 3: Score distribution analysis
    print("=" * 60)
    print("Example 3: Score distribution analysis")
    print("=" * 60)
    
    if traces_dir.exists():
        all_results = evaluate_all_traces(traces_dir)
        scores = [r['total_score'] for r in all_results]
        
        # Group by score ranges
        score_ranges = {
            "Negative (<0)": 0,
            "Low (0-0.5)": 0,
            "Medium (0.5-1.5)": 0,
            "High (1.5-2.5)": 0,
            "Very High (>2.5)": 0
        }
        
        for score in scores:
            if score < 0:
                score_ranges["Negative (<0)"] += 1
            elif score <= 0.5:
                score_ranges["Low (0-0.5)"] += 1
            elif score <= 1.5:
                score_ranges["Medium (0.5-1.5)"] += 1
            elif score <= 2.5:
                score_ranges["High (1.5-2.5)"] += 1
            else:
                score_ranges["Very High (>2.5)"] += 1
        
        print(f"\nScore distribution across {len(scores)} traces:")
        for range_name, count in score_ranges.items():
            percentage = (count / len(scores) * 100) if scores else 0
            bar = "█" * int(percentage / 2)  # Scale to 50 chars max
            print(f"  {range_name:<20} {count:3d} ({percentage:5.1f}%) {bar}")
        
        # Additional statistics
        if scores:
            print(f"\nStatistics:")
            print(f"  Mean score: {sum(scores) / len(scores):.2f}")
            print(f"  Median score: {sorted(scores)[len(scores)//2]:.2f}")
            print(f"  Std deviation: {(sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5:.2f}")

if __name__ == "__main__":
    main()