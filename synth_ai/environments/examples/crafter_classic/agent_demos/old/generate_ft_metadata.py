#!/usr/bin/env python3
"""
Generate metadata for fine-tuning datasets
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent))
from filter_traces_sft import load_trace, extract_trajectory_score, extract_llm_calls, calculate_window_score


def analyze_trajectory_dataset(traces_dir: Path, threshold: float = 2.0):
    """Analyze trajectory-based filtering results."""
    trace_files = sorted(traces_dir.glob("*.json"))
    
    included_traces = []
    excluded_traces = []
    score_distribution = defaultdict(int)
    achievement_counts = defaultdict(int)
    total_llm_calls = 0
    
    for trace_file in trace_files:
        trace = load_trace(trace_file)
        score = extract_trajectory_score(trace)
        score_distribution[int(score)] += 1
        
        # Get achievements
        metadata = trace.get('session_metadata', [])
        if isinstance(metadata, list):
            for item in metadata:
                if isinstance(item, dict) and item.get('metadata_type') == 'episode_results':
                    episode_results = item.get('data', {})
                    achievements = episode_results.get('achievements', {})
                    for ach, unlocked in achievements.items():
                        if unlocked:
                            achievement_counts[ach] += 1
                    break
        
        # Count LLM calls
        llm_calls = extract_llm_calls(trace)
        
        if score >= threshold:
            included_traces.append({
                'trace_file': trace_file.name,
                'score': score,
                'num_llm_calls': len(llm_calls),
                'achievements': [k for k, v in achievements.items() if v] if 'achievements' in locals() else []
            })
            total_llm_calls += len(llm_calls)
        else:
            excluded_traces.append({
                'trace_file': trace_file.name,
                'score': score
            })
    
    return {
        'threshold': threshold,
        'total_traces': len(trace_files),
        'included_traces': len(included_traces),
        'excluded_traces': len(excluded_traces),
        'yield_rate': len(included_traces) / len(trace_files) * 100,
        'total_examples': total_llm_calls,
        'avg_examples_per_trace': total_llm_calls / len(included_traces) if included_traces else 0,
        'score_distribution': dict(score_distribution),
        'achievement_distribution': dict(achievement_counts),
        'included_trace_details': included_traces
    }


def analyze_window_dataset(traces_dir: Path, window_size: int = 5, threshold: float = 1.0):
    """Analyze window-based filtering results with greedy extraction."""
    trace_files = sorted(traces_dir.glob("*.json"))
    
    window_scores = defaultdict(int)
    traces_with_windows = 0
    total_windows = 0
    total_examples = 0
    window_details = []
    
    for trace_file in trace_files:
        trace = load_trace(trace_file)
        llm_calls = extract_llm_calls(trace)
        
        if not llm_calls:
            continue
            
        # Get max turn
        max_turn = max(turn for turn, _ in llm_calls)
        trace_has_window = False
        used_turns = set()
        
        # Greedy extraction - same as in filter_traces_sft.py
        for start in range(0, max_turn - window_size + 2):
            end = start + window_size - 1
            
            # Skip if any turn in window already used
            if any(t in used_turns for t in range(start, end + 1)):
                continue
            
            score = calculate_window_score(trace, start, end)
            
            if score >= threshold:
                window_scores[int(score)] += 1
                total_windows += 1
                trace_has_window = True
                
                # Mark turns as used
                for t in range(start, end + 1):
                    used_turns.add(t)
                
                # Count examples in window
                window_llm_calls = [llm for turn, llm in llm_calls if start <= turn <= end]
                total_examples += len(window_llm_calls)
                
                window_details.append({
                    'trace_file': trace_file.name,
                    'window': f"[{start}-{end}]",
                    'score': score,
                    'num_examples': len(window_llm_calls)
                })
        
        if trace_has_window:
            traces_with_windows += 1
    
    return {
        'window_size': window_size,
        'threshold': threshold,
        'total_traces': len(trace_files),
        'traces_with_qualifying_windows': traces_with_windows,
        'total_windows_extracted': total_windows,
        'total_examples': total_examples,
        'avg_examples_per_window': total_examples / total_windows if total_windows else 0,
        'window_score_distribution': dict(window_scores),
        'window_details': window_details[:20]  # First 20 for brevity
    }


def main():
    traces_dir = Path("traces")
    ft_dir = Path("ft_dataset")
    
    # Analyze trajectory dataset
    print("Analyzing trajectory-based dataset...")
    traj_metadata = analyze_trajectory_dataset(traces_dir, threshold=2.0)
    
    # Analyze window dataset
    print("Analyzing window-based dataset...")
    window_metadata = analyze_window_dataset(traces_dir, window_size=5, threshold=1.0)
    
    # Create combined metadata
    combined_metadata = {
        'dataset_creation': {
            'source_traces_dir': str(traces_dir),
            'num_source_traces': traj_metadata['total_traces'],
            'filtering_methods': ['trajectory_score', 'window_score']
        },
        'trajectory_filtering': traj_metadata,
        'window_filtering': window_metadata,
        'comparison': {
            'trajectory_examples': traj_metadata['total_examples'],
            'window_examples': window_metadata['total_examples'],
            'trajectory_yield_rate': f"{traj_metadata['yield_rate']:.1f}%",
            'window_trace_coverage': f"{window_metadata['traces_with_qualifying_windows'] / window_metadata['total_traces'] * 100:.1f}%"
        }
    }
    
    # Save metadata
    with open(ft_dir / "metadata.json", 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    # Save trajectory-specific metadata
    with open(ft_dir / "trajectory_score_metadata.json", 'w') as f:
        json.dump(traj_metadata, f, indent=2)
    
    # Save window-specific metadata
    with open(ft_dir / "window_score_metadata.json", 'w') as f:
        json.dump(window_metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("FINE-TUNING DATASET SUMMARY")
    print("="*60)
    print(f"Source traces: {traj_metadata['total_traces']}")
    print(f"\nTrajectory-based filtering (score >= 2.0):")
    print(f"  - Included traces: {traj_metadata['included_traces']} ({traj_metadata['yield_rate']:.1f}%)")
    print(f"  - Total examples: {traj_metadata['total_examples']}")
    print(f"  - Avg examples/trace: {traj_metadata['avg_examples_per_trace']:.1f}")
    
    print(f"\nWindow-based filtering (window_size=5, score >= 1.0):")
    print(f"  - Traces with windows: {window_metadata['traces_with_qualifying_windows']} ({window_metadata['traces_with_qualifying_windows'] / window_metadata['total_traces'] * 100:.1f}%)")
    print(f"  - Total windows: {window_metadata['total_windows_extracted']}")
    print(f"  - Total examples: {window_metadata['total_examples']}")
    print(f"  - Avg examples/window: {window_metadata['avg_examples_per_window']:.1f}")
    
    print(f"\nWhy so many examples?")
    print(f"  - Each trace has multiple LLM calls (turns)")
    print(f"  - Trajectory method: {traj_metadata['included_traces']} traces × {traj_metadata['avg_examples_per_trace']:.1f} turns = {traj_metadata['total_examples']} examples")
    print(f"  - Window method: {window_metadata['total_windows_extracted']} windows × {window_metadata['avg_examples_per_window']:.1f} turns = {window_metadata['total_examples']} examples")


if __name__ == "__main__":
    main()