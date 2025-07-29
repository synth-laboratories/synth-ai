"""SQL schema and query management for DuckDB storage."""

# Table creation statements
TABLES = {
    "session_traces": """
        CREATE TABLE IF NOT EXISTS session_traces (
            session_id VARCHAR PRIMARY KEY,
            created_at TIMESTAMP,
            num_timesteps INTEGER,
            num_events INTEGER,
            num_messages INTEGER,
            metadata JSON,
            experiment_id VARCHAR
        )
    """,
    
    "session_timesteps": """
        CREATE TABLE IF NOT EXISTS session_timesteps (
            id INTEGER PRIMARY KEY DEFAULT nextval('timestep_id_seq'),
            session_id VARCHAR NOT NULL,
            step_id VARCHAR NOT NULL,
            step_index INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            num_events INTEGER,
            num_messages INTEGER,
            step_metadata JSON,
            UNIQUE(session_id, step_id)
        )
    """,
    
    "events": """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY DEFAULT nextval('event_id_seq'),
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
            event_metadata JSON
        )
    """,
    
    "messages": """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY DEFAULT nextval('message_id_seq'),
            session_id VARCHAR NOT NULL,
            timestep_id INTEGER,
            message_type VARCHAR NOT NULL,
            content JSON NOT NULL,
            timestamp TIMESTAMP,
            event_time TIMESTAMP,
            message_time INTEGER
        )
    """,
    
    "systems": """
        CREATE TABLE IF NOT EXISTS systems (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            description TEXT
        )
    """,
    
    "system_versions": """
        CREATE TABLE IF NOT EXISTS system_versions (
            id VARCHAR PRIMARY KEY,
            system_id VARCHAR NOT NULL,
            branch VARCHAR,
            commit VARCHAR,
            created_at TIMESTAMP NOT NULL,
            description TEXT,
            FOREIGN KEY (system_id) REFERENCES systems(id)
        )
    """,
    
    "experiments": """
        CREATE TABLE IF NOT EXISTS experiments (
            id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            description TEXT,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """,
    
    "experimental_systems": """
        CREATE TABLE IF NOT EXISTS experimental_systems (
            experiment_id VARCHAR NOT NULL,
            system_id VARCHAR NOT NULL,
            system_version_id VARCHAR NOT NULL,
            PRIMARY KEY (experiment_id, system_id),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id),
            FOREIGN KEY (system_id) REFERENCES systems(id),
            FOREIGN KEY (system_version_id) REFERENCES system_versions(id)
        )
    """
}

# Sequences
SEQUENCES = {
    "timestep_id_seq": "CREATE SEQUENCE IF NOT EXISTS timestep_id_seq",
    "event_id_seq": "CREATE SEQUENCE IF NOT EXISTS event_id_seq",
    "message_id_seq": "CREATE SEQUENCE IF NOT EXISTS message_id_seq"
}

# Indexes
INDEXES = {
    "idx_events_session_time": "CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id, event_time)",
    "idx_events_system": "CREATE INDEX IF NOT EXISTS idx_events_system ON events(system_instance_id, event_type)",
    "idx_events_model": "CREATE INDEX IF NOT EXISTS idx_events_model ON events(model_name, provider)",
    "idx_messages_session_time": "CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, timestamp)",
    "idx_messages_type": "CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type)",
    "idx_system_versions_system": "CREATE INDEX IF NOT EXISTS idx_system_versions_system ON system_versions(system_id)",
    "idx_experimental_systems": "CREATE INDEX IF NOT EXISTS idx_experimental_systems ON experimental_systems(experiment_id)",
    "idx_session_experiment": "CREATE INDEX IF NOT EXISTS idx_session_experiment ON session_traces(experiment_id)"
}

# Views
VIEWS = {
    "model_usage_stats": """
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
        GROUP BY model_name, provider
    """,
    
    "session_summary": """
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
        GROUP BY s.session_id, s.created_at, s.num_timesteps, s.num_events, s.num_messages
    """,
    
    "experiment_overview": """
        CREATE OR REPLACE VIEW experiment_overview AS
        SELECT 
            e.id as experiment_id,
            e.name as experiment_name,
            COUNT(DISTINCT st.session_id) as session_count,
            COUNT(DISTINCT es.system_id) as system_count,
            MIN(st.created_at) as first_session,
            MAX(st.created_at) as last_session
        FROM experiments e
        LEFT JOIN session_traces st ON e.id = st.experiment_id
        LEFT JOIN experimental_systems es ON e.id = es.experiment_id
        GROUP BY e.id, e.name
    """
}

# Insert statements
INSERT = {
    "session_trace": """
        INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
    
    "session_timestep": """
        INSERT INTO session_timesteps (session_id, step_id, step_index, timestamp, num_events, num_messages, step_metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        RETURNING id
    """,
    
    "message": """
        INSERT INTO messages (session_id, timestep_id, message_type, content, timestamp, event_time, message_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    
    "system": """
        INSERT INTO systems (id, name, description)
        VALUES (?, ?, ?)
    """,
    
    "system_version": """
        INSERT INTO system_versions (id, system_id, branch, commit, created_at, description)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
    
    "experiment": """
        INSERT INTO experiments (id, name, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """,
    
    "experimental_system": """
        INSERT INTO experimental_systems (experiment_id, system_id, system_version_id)
        VALUES (?, ?, ?)
    """
}

# Select statements
SELECT = {
    "session_exists": """
        SELECT session_id FROM session_traces WHERE session_id = ?
    """,
    
    "experiment_sessions": """
        SELECT 
            st.*,
            COUNT(DISTINCT e.id) as total_events,
            COUNT(DISTINCT m.id) as total_messages,
            SUM(CASE WHEN e.event_type = 'cais' THEN e.cost ELSE 0 END) as total_cost,
            SUM(CASE WHEN e.event_type = 'cais' THEN e.total_tokens ELSE 0 END) as total_tokens
        FROM session_traces st
        LEFT JOIN events e ON st.session_id = e.session_id
        LEFT JOIN messages m ON st.session_id = m.session_id
        WHERE st.experiment_id = ?
        GROUP BY st.session_id
    """,
    
    "experiments_by_system_version": """
        SELECT 
            e.*,
            es.system_id,
            es.system_version_id,
            s.name as system_name,
            sv.branch,
            sv.commit
        FROM experiments e
        JOIN experimental_systems es ON e.id = es.experiment_id
        JOIN systems s ON es.system_id = s.id
        JOIN system_versions sv ON es.system_version_id = sv.id
        WHERE es.system_version_id = ?
        ORDER BY e.created_at DESC
    """
}

# Delete statements
DELETE = {
    "session_trace": "DELETE FROM session_traces WHERE session_id = ?"
}

# Update statements (workaround for DuckDB bug - use DELETE + INSERT instead)
UPDATE = {
    "link_session_to_experiment": """
        -- This is a placeholder. Use DELETE + INSERT pattern instead due to DuckDB bug
        UPDATE session_traces SET experiment_id = ? WHERE session_id = ?
    """
}

# Bulk insert statements
BULK_INSERT = {
    "events": """
        INSERT INTO events (
            session_id, timestep_id, event_type, system_instance_id, event_time, message_time,
            span_id, trace_id, model_name, provider, prompt_tokens, completion_tokens,
            total_tokens, cost, latency_ms, reward, terminated,
            system_state_before, system_state_after, metadata, event_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    
    "messages": """
        INSERT INTO messages (
            session_id, timestep_id, message_type, content, timestamp, event_time, message_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    
    "timesteps": """
        INSERT INTO session_timesteps (
            session_id, step_id, step_index, timestamp, num_events, num_messages, step_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        RETURNING id
    """
}