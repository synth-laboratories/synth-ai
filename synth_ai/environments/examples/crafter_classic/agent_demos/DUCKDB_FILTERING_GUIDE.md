# DuckDB Filtering Guide for Fine-Tuning Data

This guide explains how to use DuckDB to filter and extract fine-tuning data from Crafter traces.

## Overview

The new DuckDB-based filtering system provides:
- Efficient storage and querying of trace data
- SQL-based filtering capabilities
- Automatic extraction of training conversations
- Rich analytics and statistics

## Prerequisites

1. Run Crafter episodes with DuckDB enabled:
```bash
python test_crafter_react_agent_openai.py --episodes 10
```

This will create `crafter_traces.duckdb` with all trace data.

## Basic Usage

### 1. Test DuckDB Connection

First, verify your traces are in DuckDB:

```bash
python test_duckdb_filter.py
```

This will show:
- Database statistics
- Sample sessions and conversations
- Test extraction functionality

### 2. Filter Traces for Fine-Tuning

Extract training data using the filter script:

```bash
# Basic usage with defaults
python filter_traces_sft_duckdb.py -d crafter_traces.duckdb -o training_data.jsonl

# With custom config
python filter_traces_sft_duckdb.py -d crafter_traces.duckdb -c duckdb_filter_config.toml

# Override specific parameters
python filter_traces_sft_duckdb.py -d crafter_traces.duckdb --min-reward 5.0 --max-cost 0.1

# Dry run to see statistics without writing output
python filter_traces_sft_duckdb.py -d crafter_traces.duckdb --dry-run
```

### 3. Configuration Options

The `duckdb_filter_config.toml` file supports:

```toml
mode = "trajectory"  # or "window" for windowed extraction

[filters]
min_total_reward = 1.0      # Minimum session reward
min_achievements = 0        # Minimum items collected
max_cost = 1.0             # Maximum cost in dollars
max_tokens = 50000         # Maximum tokens per session
```

## Output Format

The script generates OpenAI-compatible JSONL files:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Current state: ..."},
    {"role": "assistant", "content": "I'll collect wood..."}
  ],
  "metadata": {
    "session_id": "crafter_episode_0_abc123",
    "source": "duckdb_traces"
  }
}
```

## Advanced Queries

You can also query DuckDB directly for custom filtering:

```python
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager

with DuckDBTraceManager("crafter_traces.duckdb") as db:
    # Get high-reward sessions
    high_reward = db.query_traces("""
        SELECT s.session_id, SUM(e.reward) as total_reward
        FROM session_traces s
        JOIN events e ON s.session_id = e.session_id
        WHERE e.event_type = 'environment'
        GROUP BY s.session_id
        HAVING SUM(e.reward) > 10
        ORDER BY total_reward DESC
    """)
    print(high_reward)
```

## Statistics and Visualization

The filter script provides rich statistics:
- Reward distribution histogram
- Token usage distribution
- Model usage breakdown
- Filter pass rates

Example output:
```
FILTERING STATISTICS
================================================================================

Total sessions in database: 100
Sessions after filtering: 42
Training examples generated: 42
Filter pass rate: 42.0%

                    Reward Distribution
==============================================================
Count
   12 │██████████████████████
    9 │████████████████
    6 │███████████
    3 │█████
    0 └──────────────────────
      0.0    5.2    10.4    15.6    20.8
                    Total Reward
```

## Troubleshooting

1. **No data in database**: Make sure you've run episodes with DuckDB enabled
2. **Import errors**: Ensure you're in the correct directory with synth_ai in path
3. **Empty results**: Try lowering filter thresholds (e.g., `--min-reward 0`)

## Next Steps

After filtering:
1. Validate the training data format
2. Upload to OpenAI for fine-tuning
3. Test the fine-tuned model on new Crafter episodes