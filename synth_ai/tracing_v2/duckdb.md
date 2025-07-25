# DuckDB Integration for Tracing v2

## Overview

This document outlines the design and implementation plan for integrating DuckDB as a storage backend for the tracing_v2 system. DuckDB provides efficient analytical queries over trace data while maintaining the flexibility to handle complex nested structures.

## Motivation

- **Analytical Queries**: Enable SQL-based analysis of traces (e.g., token usage analytics, latency patterns)
- **Scalability**: Handle large volumes of trace data efficiently
- **Local Storage**: Keep data local while providing database features
- **Flexible Schema**: Support evolving trace structures with JSON columns
- **Fast Aggregations**: Compute statistics across sessions, models, and providers

## Proposed Schema Design

### Core Tables

```sql
-- Main session trace table
CREATE TABLE session_traces (
    session_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    num_timesteps INTEGER,
    num_events INTEGER,
    num_messages INTEGER,
    metadata JSON,  -- Stores session_metadata as JSON array
    UNIQUE(session_id)
);

-- Timesteps within sessions
CREATE TABLE session_timesteps (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    step_id VARCHAR NOT NULL,
    step_index INTEGER NOT NULL,  -- Order within session
    timestamp TIMESTAMP NOT NULL,
    num_events INTEGER,
    num_messages INTEGER,
    step_metadata JSON,
    FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
    UNIQUE(session_id, step_id)
);

-- Events (polymorphic table for all event types)
CREATE TABLE events (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    timestep_id INTEGER,
    event_type VARCHAR NOT NULL,  -- 'cais', 'environment', 'runtime'
    system_instance_id VARCHAR NOT NULL,
    event_time TIMESTAMP,
    message_time INTEGER,  -- Logical time (turn number)
    
    -- CAIS-specific fields (NULL for other types)
    span_id VARCHAR,
    trace_id VARCHAR,
    model_name VARCHAR,
    provider VARCHAR,  -- 'openai', 'anthropic', 'azure_openai', etc.
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost DOUBLE,
    latency_ms DOUBLE,
    
    -- Environment-specific fields (NULL for other types)
    reward DOUBLE,
    terminated BOOLEAN,
    
    -- Common fields
    system_state_before JSON,
    system_state_after JSON,
    metadata JSON,
    event_metadata JSON,  -- Hook results
    
    FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
    FOREIGN KEY (timestep_id) REFERENCES session_timesteps(id)
);

-- Messages
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    timestep_id INTEGER,
    message_type VARCHAR NOT NULL,  -- 'llm_prompt', 'llm_completion', etc.
    content JSON NOT NULL,  -- Store full content as JSON
    timestamp TIMESTAMP,
    event_time TIMESTAMP,
    message_time INTEGER,  -- Logical time
    
    FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
    FOREIGN KEY (timestep_id) REFERENCES session_timesteps(id)
);

-- Indexes for performance
CREATE INDEX idx_events_session_time ON events(session_id, event_time);
CREATE INDEX idx_events_system ON events(system_instance_id, event_type);
CREATE INDEX idx_events_model ON events(model_name, provider);
CREATE INDEX idx_messages_session_time ON messages(session_id, timestamp);
CREATE INDEX idx_messages_type ON messages(message_type);
```

### Materialized Views for Analytics

```sql
-- Model usage statistics
CREATE VIEW model_usage_stats AS
SELECT 
    model_name,
    provider,
    COUNT(*) as call_count,
    SUM(prompt_tokens) as total_prompt_tokens,
    SUM(completion_tokens) as total_completion_tokens,
    SUM(total_tokens) as total_tokens,
    AVG(latency_ms) as avg_latency_ms,
    SUM(cost) as total_cost,
    MIN(event_time) as first_call,
    MAX(event_time) as last_call
FROM events
WHERE event_type = 'cais' AND model_name IS NOT NULL
GROUP BY model_name, provider;

-- Session summary
CREATE VIEW session_summary AS
SELECT 
    s.session_id,
    s.created_at,
    s.num_timesteps,
    s.num_events,
    s.num_messages,
    COUNT(DISTINCT e.model_name) as models_used,
    COUNT(DISTINCT e.provider) as providers_used,
    SUM(e.total_tokens) as total_tokens_used,
    SUM(e.cost) as total_cost
FROM session_traces s
LEFT JOIN events e ON s.session_id = e.session_id AND e.event_type = 'cais'
GROUP BY s.session_id, s.created_at, s.num_timesteps, s.num_events, s.num_messages;
```

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Create DuckDB Manager** (`duckdb_manager.py`)
   ```python
   class DuckDBTraceManager:
       def __init__(self, db_path: str = "traces.duckdb"):
           self.db_path = db_path
           self.init_schema()
       
       def init_schema(self):
           # Create tables if not exist
           pass
       
       def insert_session_trace(self, trace: SessionTrace):
           # Convert trace to relational format
           # Insert into appropriate tables
           pass
       
       def query_traces(self, query: str) -> pd.DataFrame:
           # Execute analytical queries
           pass
   ```

2. **Add Serialization Methods**
   - Extend dataclasses with `to_dict()` methods for DuckDB compatibility
   - Handle JSON serialization for complex fields

3. **Batch Upload Support**
   ```python
   def upload_trace_batch(self, traces: List[SessionTrace]):
       # Efficient batch insertion
       pass
   ```

### Phase 2: Query Interface

1. **Common Query Patterns**
   ```python
   class TraceQueries:
       @staticmethod
       def get_model_usage(start_date: datetime, end_date: datetime):
           return """
           SELECT * FROM model_usage_stats
           WHERE first_call >= ? AND last_call <= ?
           """
       
       @staticmethod
       def get_expensive_calls(cost_threshold: float):
           return """
           SELECT session_id, model_name, prompt_tokens, 
                  completion_tokens, cost, latency_ms
           FROM events
           WHERE event_type = 'cais' AND cost > ?
           ORDER BY cost DESC
           """
   ```

2. **Export Utilities**
   ```python
   def export_to_parquet(self, output_path: str):
       # Export traces to Parquet for external analysis
       pass
   ```

### Phase 3: Integration with SessionTracer

1. **Auto-upload Option**
   ```python
   class SessionTracer:
       def __init__(self, traces_dir: str = "traces", 
                    hooks: Optional[List[TraceHook]] = None,
                    duckdb_path: Optional[str] = None):
           # ... existing init ...
           self.db_manager = DuckDBTraceManager(duckdb_path) if duckdb_path else None
       
       def end_session(self, save: bool = True, upload_to_db: bool = True):
           # ... existing code ...
           if self.db_manager and upload_to_db:
               self.db_manager.insert_session_trace(self.current_session)
   ```

2. **Streaming Upload**
   - Option to stream events to DuckDB as they occur
   - Useful for long-running sessions

### Phase 4: Analytics Dashboard

1. **Query Examples**
   ```sql
   -- Token usage by provider over time
   SELECT 
       DATE_TRUNC('hour', event_time) as hour,
       provider,
       SUM(total_tokens) as tokens_used
   FROM events
   WHERE event_type = 'cais'
   GROUP BY hour, provider
   ORDER BY hour;
   
   -- Find sessions with errors
   SELECT DISTINCT s.session_id, s.created_at
   FROM session_traces s
   JOIN events e ON s.session_id = e.session_id
   WHERE e.metadata->>'error' IS NOT NULL;
   
   -- Message flow analysis
   SELECT 
       m.message_type,
       COUNT(*) as count,
       AVG(LENGTH(m.content::text)) as avg_content_length
   FROM messages m
   GROUP BY m.message_type;
   ```

2. **Performance Monitoring**
   - Track latency trends
   - Identify slow operations
   - Cost optimization insights

## Migration Strategy

1. **Backward Compatibility**
   - Keep JSON file storage as default
   - DuckDB as optional add-on
   - Batch migration tool for existing traces

2. **Migration Script**
   ```python
   def migrate_json_traces_to_duckdb(json_dir: str, db_path: str):
       db = DuckDBTraceManager(db_path)
       for trace_file in Path(json_dir).glob("*.json"):
           trace = load_trace_from_json(trace_file)
           db.insert_session_trace(trace)
   ```

## Configuration

Add to `config.py`:
```python
DUCKDB_CONFIG = {
    "enabled": False,  # Set to True to enable DuckDB storage
    "db_path": "traces.duckdb",
    "batch_size": 1000,  # For batch uploads
    "auto_upload": True,  # Upload on session end
    "retention_days": 30,  # Data retention policy
}
```

## Benefits

1. **SQL Analytics**: Query traces with familiar SQL syntax
2. **Performance**: DuckDB's columnar storage for fast analytics
3. **Local-first**: No external dependencies, runs embedded
4. **Schema Evolution**: JSON columns handle schema changes gracefully
5. **Export Options**: Easy export to Parquet, CSV for external tools

## Next Steps

1. Implement Phase 1 core infrastructure
2. Create unit tests for DuckDB operations
3. Add example notebooks for trace analysis
4. Document query patterns and best practices
5. Performance benchmarking with large trace volumes