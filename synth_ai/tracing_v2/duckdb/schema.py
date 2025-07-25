"""
DuckDB schema definitions for tracing_v2.
"""

SCHEMA_SQL = """
-- Main session trace table
CREATE TABLE IF NOT EXISTS session_traces (
    session_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    num_timesteps INTEGER,
    num_events INTEGER,
    num_messages INTEGER,
    metadata JSON
);

-- Timesteps within sessions
CREATE TABLE IF NOT EXISTS session_timesteps (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    step_id VARCHAR NOT NULL,
    step_index INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    num_events INTEGER,
    num_messages INTEGER,
    step_metadata JSON,
    FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
    UNIQUE(session_id, step_id)
);

-- Events (polymorphic table for all event types)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    timestep_id INTEGER,
    event_type VARCHAR NOT NULL,
    system_instance_id VARCHAR NOT NULL,
    event_time TIMESTAMP,
    message_time INTEGER,
    
    -- CAIS-specific fields
    span_id VARCHAR,
    trace_id VARCHAR,
    model_name VARCHAR,
    provider VARCHAR,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost DOUBLE,
    latency_ms DOUBLE,
    
    -- Environment-specific fields
    reward DOUBLE,
    terminated BOOLEAN,
    
    -- Common fields
    system_state_before JSON,
    system_state_after JSON,
    metadata JSON,
    event_metadata JSON,
    
    FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
    FOREIGN KEY (timestep_id) REFERENCES session_timesteps(id)
);

-- Messages
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    timestep_id INTEGER,
    message_type VARCHAR NOT NULL,
    content JSON NOT NULL,
    timestamp TIMESTAMP,
    event_time TIMESTAMP,
    message_time INTEGER,
    
    FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
    FOREIGN KEY (timestep_id) REFERENCES session_timesteps(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id, event_time);
CREATE INDEX IF NOT EXISTS idx_events_system ON events(system_instance_id, event_type);
CREATE INDEX IF NOT EXISTS idx_events_model ON events(model_name, provider);
CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type);
"""

ANALYTICS_VIEWS_SQL = """
-- Model usage statistics
CREATE OR REPLACE VIEW model_usage_stats AS
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
CREATE OR REPLACE VIEW session_summary AS
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

-- Error analysis view
CREATE OR REPLACE VIEW error_events AS
SELECT 
    session_id,
    event_type,
    system_instance_id,
    event_time,
    metadata
FROM events
WHERE metadata LIKE '%error%' OR metadata LIKE '%exception%';

-- Performance metrics view
CREATE OR REPLACE VIEW performance_metrics AS
SELECT 
    DATE_TRUNC('hour', event_time) as hour,
    model_name,
    COUNT(*) as calls,
    AVG(latency_ms) as avg_latency,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency
FROM events
WHERE event_type = 'cais' AND latency_ms IS NOT NULL
GROUP BY hour, model_name;
"""