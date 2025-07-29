---
title: 'Trace Manager'
description: 'Trace storage and management'
---

# Trace Manager

The trace manager provides persistent storage for tracing data.

## Classes

### TraceManager

Manages trace storage and retrieval.

**Signature:** `TraceManager(db_path: str)`

**Methods:**
- `insert_session_trace(session)` - Insert a session trace
- `query_traces(query: str)` - Execute a query on trace data
- `create_experiment(experiment_id: str, name: str, description: str)` - Create an experiment

**Example:**
```python
from synth_ai.tracing.manager import TraceManager

with TraceManager("traces.db") as db:
    # Insert session trace
    db.insert_session_trace(session)
    
    # Query traces
    results = db.query_traces("SELECT * FROM session_traces")
```

