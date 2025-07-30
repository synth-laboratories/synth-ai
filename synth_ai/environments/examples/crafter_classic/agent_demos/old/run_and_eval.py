#!/usr/bin/env python3
"""
Run Crafter agent evaluation and automatically evaluate traces.
"""

import subprocess
import sys
import time
from pathlib import Path
from trace_eval import evaluate_all_traces, print_evaluation_summary, print_trace_evaluation

def main():
    # Run the agent evaluation
    print("🎮 Running Crafter Agent Evaluation...")
    print("=" * 60)
    
    # Pass all arguments to the test script
    cmd = [sys.executable, "test_crafter_react_agent_openai.py"] + sys.argv[1:]
    
    # Record start time
    start_time = time.time()
    
    # Run the evaluation
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Agent evaluation failed with return code {result.returncode}")
        return
    
    # Wait a moment for files to be written
    time.sleep(1)
    
    # Find recent trace files
    print("\n" + "=" * 80)
    print("📊 TRACE EVALUATION")
    print("=" * 80)
    
    trace_dir = Path("traces")
    if not trace_dir.exists():
        print("❌ No traces directory found")
        return
    
    # Find traces created since we started
    recent_traces = []
    for trace_file in trace_dir.glob("*.json"):
        if trace_file.stat().st_mtime >= start_time:
            recent_traces.append(trace_file)
    
    if not recent_traces:
        print("❌ No new trace files found")
        return
    
    print(f"Found {len(recent_traces)} new trace files")
    
    # Evaluate all recent traces
    results = []
    for trace_file in recent_traces:
        from trace_eval import evaluate_trace
        result = evaluate_trace(trace_file)
        results.append(result)
    
    # Sort by score
    results.sort(key=lambda x: x['total_score'], reverse=True)
    
    # Show individual evaluations if not too many
    if len(results) <= 5:
        for result in results:
            print_trace_evaluation(result)
    
    # Always show summary
    print_evaluation_summary(results)
    
    # Show achievement distribution
    print("\n" + "=" * 80)
    print("📊 ACHIEVEMENT DISTRIBUTION")
    print("=" * 80)
    
    total_easy = sum(r['counts'].get('easy_achievement', 0) for r in results)
    total_medium = sum(r['counts'].get('medium_achievement', 0) for r in results)
    total_hard = sum(r['counts'].get('hard_achievement', 0) for r in results)
    total_invalid = sum(r['counts'].get('invalid_action', 0) for r in results)
    
    print(f"Easy achievements:    {total_easy} total ({total_easy/len(results):.1f} per episode)")
    print(f"Medium achievements:  {total_medium} total ({total_medium/len(results):.1f} per episode)")
    print(f"Hard achievements:    {total_hard} total ({total_hard/len(results):.1f} per episode)")
    print(f"Invalid actions:      {total_invalid} total ({total_invalid/len(results):.1f} per episode)")
    
    # Score interpretation
    avg_score = sum(r['total_score'] for r in results) / len(results)
    print(f"\nAverage Score: {avg_score:.2f}")
    
    if avg_score >= 2.0:
        print("🎉 Excellent performance!")
    elif avg_score >= 1.0:
        print("✅ Good performance")
    elif avg_score >= 0.0:
        print("📈 Room for improvement")
    else:
        print("⚠️  Many invalid actions detected")

if __name__ == "__main__":
    main()