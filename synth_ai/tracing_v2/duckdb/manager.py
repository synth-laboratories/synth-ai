"""
DuckDB integration for tracing_v2 system.
Provides efficient storage and analytics for trace data.
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import duckdb
import pandas as pd

from ..session_tracer import SessionTrace, SessionTimeStep, CAISEvent, EnvironmentEvent, RuntimeEvent, SessionEventMessage
from ..abstractions import SessionMetadum


class DuckDBTraceManager:
    """Manages DuckDB storage for trace data."""
    
    def __init__(self, db_path: str = "traces.duckdb"):
        """Initialize DuckDB manager with database path."""
        self.db_path = db_path
        self.conn = None
        self._connect()
        self.init_schema()
    
    def _connect(self):
        """Establish connection to DuckDB."""
        self.conn = duckdb.connect(self.db_path)
    
    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def init_schema(self):
        """Initialize database schema."""
        # Main session trace table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_traces (
                session_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP,
                num_timesteps INTEGER,
                num_events INTEGER,
                num_messages INTEGER,
                metadata JSON
            )
        """)
        
        # Timesteps within sessions
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS timestep_id_seq
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS session_timesteps (
                id INTEGER PRIMARY KEY DEFAULT nextval('timestep_id_seq'),
                session_id VARCHAR NOT NULL,
                step_id VARCHAR NOT NULL,
                step_index INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                num_events INTEGER,
                num_messages INTEGER,
                step_metadata JSON,
                FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
                UNIQUE(session_id, step_id)
            )
        """)
        
        # Events (polymorphic table for all event types)
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS event_id_seq
        """)
        
        self.conn.execute("""
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
                event_metadata JSON,
                
                FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
                FOREIGN KEY (timestep_id) REFERENCES session_timesteps(id)
            )
        """)
        
        # Messages
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS message_id_seq
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY DEFAULT nextval('message_id_seq'),
                session_id VARCHAR NOT NULL,
                timestep_id INTEGER,
                message_type VARCHAR NOT NULL,
                content JSON NOT NULL,
                timestamp TIMESTAMP,
                event_time TIMESTAMP,
                message_time INTEGER,
                
                FOREIGN KEY (session_id) REFERENCES session_traces(session_id),
                FOREIGN KEY (timestep_id) REFERENCES session_timesteps(id)
            )
        """)
        
        # Systems table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS systems (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description TEXT
            )
        """)
        
        # System versions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS system_versions (
                id VARCHAR PRIMARY KEY,
                system_id VARCHAR NOT NULL,
                branch VARCHAR,
                commit VARCHAR,
                created_at TIMESTAMP NOT NULL,
                description TEXT,
                FOREIGN KEY (system_id) REFERENCES systems(id)
            )
        """)
        
        # Experiments table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                description TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)
        
        # Experimental systems (many-to-many relationship)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experimental_systems (
                experiment_id VARCHAR NOT NULL,
                system_id VARCHAR NOT NULL,
                system_version_id VARCHAR NOT NULL,
                PRIMARY KEY (experiment_id, system_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                FOREIGN KEY (system_id) REFERENCES systems(id),
                FOREIGN KEY (system_version_id) REFERENCES system_versions(id)
            )
        """)
        
        # Link sessions to experiments
        self.conn.execute("""
            ALTER TABLE session_traces ADD COLUMN IF NOT EXISTS experiment_id VARCHAR;
        """)
        
        # Add foreign key constraint for experiment_id
        try:
            self.conn.execute("""
                ALTER TABLE session_traces ADD CONSTRAINT fk_experiment 
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            """)
        except:
            # Constraint might already exist
            pass
        
        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id, event_time)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_system ON events(system_instance_id, event_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_model ON events(model_name, provider)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, timestamp)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_system_versions_system ON system_versions(system_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_experimental_systems ON experimental_systems(experiment_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_session_experiment ON session_traces(experiment_id)")
        
        # Create analytics views
        self._create_analytics_views()
    
    def _create_analytics_views(self):
        """Create materialized views for analytics."""
        # Model usage statistics
        self.conn.execute("""
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
        """)
        
        # Session summary
        self.conn.execute("""
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
        """)
    
    def insert_session_trace(self, trace: SessionTrace):
        """Insert a complete session trace into DuckDB."""
        session_id = trace.session_metadata[0].data.get("session_id") if trace.session_metadata else None
        if not session_id:
            session_id = f"session_{datetime.now().isoformat()}"
        
        # Get timesteps from session_time_steps
        timesteps = trace.session_time_steps
        
        # Insert main session record
        self.conn.execute("""
            INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            session_id,
            datetime.now(),
            len(timesteps),
            len(trace.event_history),
            len(trace.message_history),
            json.dumps([self._serialize_metadata(m) for m in trace.session_metadata])
        ])
        
        # Insert timesteps
        timestep_id_map = {}
        for idx, timestep in enumerate(timesteps):
            timestep_id = self._insert_timestep(session_id, timestep, idx)
            timestep_id_map[timestep.step_id] = timestep_id
        
        # Insert events
        for event in trace.event_history:
            self._insert_event(session_id, event, timestep_id_map)
        
        # Insert messages
        for message in trace.message_history:
            self._insert_message(session_id, message, timestep_id_map)
        
        # Commit transaction
        self.conn.commit()
    
    def _serialize_metadata(self, metadata: SessionMetadum) -> Dict[str, Any]:
        """Serialize session metadata to dictionary."""
        return {
            "metadata_type": metadata.__class__.__name__,
            "data": getattr(metadata, 'data', {})
        }
    
    def _insert_timestep(self, session_id: str, timestep: SessionTimeStep, index: int) -> int:
        """Insert a timestep and return its ID."""
        result = self.conn.execute("""
            INSERT INTO session_timesteps (session_id, step_id, step_index, timestamp, num_events, num_messages, step_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id
        """, [
            session_id,
            timestep.step_id,
            index,
            timestep.timestamp or datetime.now(),
            len(timestep.events),
            len(timestep.step_messages),
            json.dumps(timestep.step_metadata) if timestep.step_metadata else None
        ]).fetchone()
        
        return result[0]
    
    def _insert_event(self, session_id: str, event: Union[CAISEvent, EnvironmentEvent, RuntimeEvent], timestep_id_map: Dict[str, int]):
        """Insert an event into the database."""
        event_type = event.__class__.__name__.replace('Event', '').lower()
        
        # Common fields
        params = {
            'session_id': session_id,
            'timestep_id': None,  # TODO: Map event to timestep
            'event_type': event_type,
            'system_instance_id': getattr(event, 'system_instance_id', ''),
            'event_time': datetime.now(),
            'message_time': getattr(event.time_record, 'message_time', None) if hasattr(event, 'time_record') else None,
        }
        
        # Type-specific fields
        if isinstance(event, CAISEvent):
            # First try to extract from llm_call_records (OTel path)
            if event.llm_call_records:
                record = event.llm_call_records[0]  # Take first record
                if hasattr(record, 'span_id'):
                    params['span_id'] = record.span_id
                if hasattr(record, 'trace_id'):
                    params['trace_id'] = record.trace_id
                if hasattr(record, 'attributes'):
                    attrs = record.attributes
                    params['model_name'] = attrs.get('gen_ai.request.model')
                    params['provider'] = attrs.get('gen_ai.system')
                    params['prompt_tokens'] = attrs.get('gen_ai.usage.prompt_tokens')
                    params['completion_tokens'] = attrs.get('gen_ai.usage.completion_tokens')
                    params['total_tokens'] = attrs.get('llm.usage.total_tokens')
                    if hasattr(record, 'duration_ms'):
                        params['latency_ms'] = record.duration_ms
            
            # If no llm_call_records or missing data, try system_state_before/after (v2 path)
            if hasattr(event, 'system_state_before') and event.system_state_before:
                state = event.system_state_before
                if isinstance(state, dict):
                    # Extract model info
                    if not params.get('model_name'):
                        params['model_name'] = state.get('gen_ai.request.model') or state.get('llm.model_name')
                    
                    # Detect provider if not already set
                    if not params.get('provider') and params.get('model_name'):
                        from ..config import detect_provider
                        params['provider'] = detect_provider(params['model_name'])
            
            # Extract token usage from system_state_after (where response data is stored)
            if hasattr(event, 'system_state_after') and event.system_state_after:
                state_after = event.system_state_after
                if isinstance(state_after, dict):
                    # Extract token usage
                    if not params.get('prompt_tokens'):
                        params['prompt_tokens'] = state_after.get('gen_ai.response.usage.prompt_tokens') or state_after.get('gen_ai.usage.prompt_tokens')
                    if not params.get('completion_tokens'):
                        params['completion_tokens'] = state_after.get('gen_ai.response.usage.completion_tokens') or state_after.get('gen_ai.usage.completion_tokens')
                    if not params.get('total_tokens'):
                        params['total_tokens'] = state_after.get('gen_ai.response.usage.total_tokens') or state_after.get('gen_ai.usage.total_tokens') or state_after.get('llm.usage.total_tokens')
            
            # Extract metadata fields
            if hasattr(event, 'metadata') and event.metadata:
                if not params.get('latency_ms') and 'duration_ms' in event.metadata:
                    params['latency_ms'] = event.metadata['duration_ms']
        
        elif isinstance(event, EnvironmentEvent):
            params['reward'] = event.reward
            params['terminated'] = event.terminated
        
        # Common fields for all events
        params['system_state_before'] = json.dumps(event.system_state_before) if hasattr(event, 'system_state_before') else None
        params['system_state_after'] = json.dumps(event.system_state_after) if hasattr(event, 'system_state_after') else None
        params['metadata'] = json.dumps(event.metadata) if hasattr(event, 'metadata') else None
        # Handle event_metadata serialization
        if hasattr(event, 'event_metadata') and event.event_metadata:
            metadata_list = []
            for em in event.event_metadata:
                if hasattr(em, '__dict__'):
                    metadata_list.append(vars(em))
                elif hasattr(em, 'to_dict'):
                    metadata_list.append(em.to_dict())
                else:
                    metadata_list.append(str(em))
            params['event_metadata'] = json.dumps(metadata_list)
        else:
            params['event_metadata'] = None
        
        # Build INSERT query dynamically
        columns = []
        values = []
        placeholders = []
        
        for col, val in params.items():
            if val is not None:
                columns.append(col)
                values.append(val)
                placeholders.append('?')
        
        query = f"""
            INSERT INTO events ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        
        self.conn.execute(query, values)
    
    def _insert_message(self, session_id: str, message: SessionEventMessage, timestep_id_map: Dict[str, int]):
        """Insert a message into the database."""
        # Convert message to dict for JSON serialization
        message_dict = {
            'content': message.content,
            'timestamp': message.timestamp,
            'message_type': message.message_type
        }
        if message.time_record:
            message_dict['time_record'] = message.time_record.to_dict()
        
        self.conn.execute("""
            INSERT INTO messages (session_id, timestep_id, message_type, content, timestamp, event_time, message_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            session_id,
            None,  # TODO: Map message to timestep
            message.__class__.__name__,
            json.dumps(message_dict),
            datetime.now(),
            getattr(message.time_record, 'event_time', None) if hasattr(message, 'time_record') else None,
            getattr(message.time_record, 'message_time', None) if hasattr(message, 'time_record') else None,
        ])
    
    def query_traces(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute analytical query and return results as DataFrame."""
        if params:
            return self.conn.execute(query, params).df()
        return self.conn.execute(query).df()
    
    def get_model_usage(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get model usage statistics."""
        query = "SELECT * FROM model_usage_stats"
        conditions = []
        params = []
        
        if start_date:
            conditions.append("first_call >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("last_call <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        return self.query_traces(query, params)
    
    def get_expensive_calls(self, cost_threshold: float) -> pd.DataFrame:
        """Get expensive LLM calls above threshold."""
        query = """
            SELECT session_id, model_name, prompt_tokens, 
                   completion_tokens, cost, latency_ms
            FROM events
            WHERE event_type = 'cais' AND cost > ?
            ORDER BY cost DESC
        """
        return self.query_traces(query, [cost_threshold])
    
    def get_session_summary(self, session_id: Optional[str] = None) -> pd.DataFrame:
        """Get session summary information."""
        query = "SELECT * FROM session_summary"
        params = []
        
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        
        return self.query_traces(query, params)
    
    def export_to_parquet(self, output_dir: str):
        """Export all tables to Parquet format."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        tables = ['session_traces', 'session_timesteps', 'events', 'messages']
        
        for table in tables:
            df = self.conn.execute(f"SELECT * FROM {table}").df()
            df.to_parquet(output_path / f"{table}.parquet", index=False)
    
    def batch_upload(self, traces: List[SessionTrace], batch_size: int = 1000):
        """Upload multiple traces efficiently in batches."""
        for trace in traces:
            self.insert_session_trace(trace)
            # DuckDB handles transactions efficiently, no need for manual batching
    
    # Experiment and System Management Methods
    
    def create_system(self, system_id: str, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new system."""
        self.conn.execute("""
            INSERT INTO systems (id, name, description)
            VALUES (?, ?, ?)
        """, [system_id, name, description])
        return {"id": system_id, "name": name, "description": description}
    
    def create_system_version(self, version_id: str, system_id: str, branch: str, 
                            commit: str, description: str = "") -> Dict[str, Any]:
        """Create a new system version."""
        created_at = datetime.utcnow()
        self.conn.execute("""
            INSERT INTO system_versions (id, system_id, branch, commit, created_at, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [version_id, system_id, branch, commit, created_at, description])
        return {
            "id": version_id,
            "system_id": system_id,
            "branch": branch,
            "commit": commit,
            "created_at": created_at.isoformat(),
            "description": description
        }
    
    def create_experiment(self, experiment_id: str, name: str, description: str = "",
                         system_versions: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Create a new experiment with optional system versions."""
        created_at = updated_at = datetime.utcnow()
        
        # Create experiment
        self.conn.execute("""
            INSERT INTO experiments (id, name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, [experiment_id, name, description, created_at, updated_at])
        
        # Link system versions if provided
        if system_versions:
            for sv in system_versions:
                self.conn.execute("""
                    INSERT INTO experimental_systems (experiment_id, system_id, system_version_id)
                    VALUES (?, ?, ?)
                """, [experiment_id, sv["system_id"], sv["system_version_id"]])
        
        return {
            "id": experiment_id,
            "name": name,
            "description": description,
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
            "system_versions": system_versions or []
        }
    
    def link_session_to_experiment(self, session_id: str, experiment_id: str):
        """Link a session to an experiment."""
        self.conn.execute("""
            UPDATE session_traces
            SET experiment_id = ?
            WHERE session_id = ?
        """, [experiment_id, session_id])
    
    def get_experiment_sessions(self, experiment_id: str) -> pd.DataFrame:
        """Get all sessions for an experiment."""
        return self.conn.execute("""
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
        """, [experiment_id]).df()
    
    def get_experiments_by_system_version(self, system_version_id: str) -> pd.DataFrame:
        """Get all experiments using a specific system version."""
        return self.conn.execute("""
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
        """, [system_version_id]).df()