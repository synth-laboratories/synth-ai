#!/usr/bin/env python3
"""
DEPRECATED: This script evaluates individual JSON trace files which are no longer generated.

The trace evaluation functionality should be updated to work with DuckDB instead.
Individual JSON trace files have been replaced with centralized DuckDB storage.

To evaluate traces from DuckDB:
1. Query traces using DuckDBTraceManager
2. Extract achievement and action data from the events table
3. Calculate scores based on the same scoring logic

This file is kept for reference only and will be removed in a future version.
"""

import sys
print("=" * 80)
print("DEPRECATED: trace_eval.py is no longer supported.")
print("Trace evaluation should be updated to work with DuckDB.")
print("=" * 80)
print("\nJSON trace files are no longer generated. All traces are stored in DuckDB.")
print("Update your evaluation scripts to query data from crafter_traces.duckdb")
sys.exit(1)