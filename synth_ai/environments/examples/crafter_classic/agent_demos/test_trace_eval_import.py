#!/usr/bin/env python3
"""Test trace evaluation import and functionality."""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from trace_eval import evaluate_trace, print_trace_evaluation
    print("✅ Successfully imported trace_eval module")
    
    # Test on a recent trace
    trace_dir = current_dir / "traces"
    if trace_dir.exists():
        recent_traces = sorted(trace_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if recent_traces:
            print(f"\nTesting on trace: {recent_traces[0].name}")
            result = evaluate_trace(recent_traces[0])
            print_trace_evaluation(result)
        else:
            print("No trace files found")
    else:
        print("Traces directory not found")
        
except ImportError as e:
    print(f"❌ Failed to import trace_eval: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()