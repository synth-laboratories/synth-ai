#!/usr/bin/env python3
"""
DEPRECATED: This script reads individual JSON trace files which are no longer generated.
Please use filter_traces_sft_duckdb.py instead, which reads from DuckDB.

Original functionality:
- Filter traces to create OpenAI SFT-ready .jsonl files
- Supports trajectory-level and window-based filtering

Migration guide:
1. Run your agent with DuckDB enabled (this happens automatically now)
2. Use filter_traces_sft_duckdb.py to filter and extract training data
3. Example: python filter_traces_sft_duckdb.py -d crafter_traces.duckdb -o training.jsonl

This file is kept for reference only and will be removed in a future version.
"""

import sys
print("=" * 80)
print("DEPRECATED: This script is no longer supported.")
print("Please use filter_traces_sft_duckdb.py instead.")
print("=" * 80)
print("\nExample usage:")
print("  python filter_traces_sft_duckdb.py -d crafter_traces.duckdb -o training.jsonl")
print("\nSee DUCKDB_FILTERING_GUIDE.md for more information.")
sys.exit(1)