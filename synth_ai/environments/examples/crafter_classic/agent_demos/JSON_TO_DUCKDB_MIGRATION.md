# JSON to DuckDB Migration Guide

## Overview

Individual JSON trace files have been **deprecated** in favor of centralized DuckDB storage. This change provides:
- Better performance for large-scale trace collection
- SQL-based analytics and filtering
- Reduced file system clutter
- Easier data management

## What Changed

### Deprecated Features

1. **Individual JSON trace files** - No longer saved to `traces/` directory
2. **filter_traces_sft.py** - Replaced with `filter_traces_sft_duckdb.py`
3. **trace_eval.py** - JSON-based trace evaluation deprecated
4. **Manual trace file management** - All traces now in single DuckDB file

### New Features

1. **Automatic DuckDB storage** - Traces saved to `crafter_traces.duckdb`
2. **SQL-based filtering** - Use `filter_traces_sft_duckdb.py`
3. **Rich analytics** - Built-in model usage stats, cost tracking, etc.
4. **Efficient querying** - No need to load multiple JSON files

## Migration Steps

### For Running Episodes

No changes needed! Just run as before:
```bash
python test_crafter_react_agent_openai.py --episodes 10 --model gpt-4o-mini
```

Traces are automatically saved to DuckDB.

### For Filtering Training Data

**Old way (deprecated):**
```bash
python filter_traces_sft.py --traces-dir traces/ --output training.jsonl
```

**New way:**
```bash
python filter_traces_sft_duckdb.py -d crafter_traces.duckdb -o training.jsonl
```

### For Custom Analysis

**Old way:**
```python
# Load individual JSON files
import json
for trace_file in Path("traces").glob("*.json"):
    with open(trace_file) as f:
        trace = json.load(f)
        # analyze trace...
```

**New way:**
```python
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager

with DuckDBTraceManager("crafter_traces.duckdb") as db:
    # Query with SQL
    df = db.query_traces("""
        SELECT * FROM events 
        WHERE event_type = 'environment' 
        AND reward > 0
    """)
    # analyze data...
```

## Benefits

1. **Performance**: Single database file instead of thousands of JSON files
2. **Analytics**: SQL queries, aggregations, joins
3. **Storage**: More efficient storage format
4. **Management**: Single file to backup/transfer
5. **Scalability**: Handles millions of events efficiently

## Deprecated Scripts

The following scripts now show deprecation warnings:
- `filter_traces_sft.py` → Use `filter_traces_sft_duckdb.py`
- `trace_eval.py` → Update to query DuckDB instead

## FAQ

**Q: Can I still access old JSON traces?**
A: Yes, but you'll need to use the old scripts (renamed with `_OLD` suffix)

**Q: What if I need JSON format?**
A: The filter script still outputs JSONL for fine-tuning. You can also export from DuckDB.

**Q: Is the trace data format the same?**
A: Yes, the same data is stored, just in a database instead of JSON files.

**Q: Can I convert old JSON traces to DuckDB?**
A: Yes, you can write a script to load JSON files and insert into DuckDB using the same APIs.

## Need Help?

See `DUCKDB_FILTERING_GUIDE.md` for detailed usage of the new filtering system.