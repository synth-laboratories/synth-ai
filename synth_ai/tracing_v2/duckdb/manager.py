"""
DuckDB integration for tracing_v2 system.
Provides efficient storage and analytics for trace data.
"""
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import duckdb
import pandas as pd
from tqdm import tqdm

from ..session_tracer import SessionTrace, SessionTimeStep, LMCAISEvent, EnvironmentEvent, RuntimeEvent, SessionEventMessage
from ..abstractions import CAISEvent
from ..abstractions import SessionMetadum
from ..storage.base import TraceStorage
from ..storage.types import (
    EventType, MessageType, Provider,
    SessionRecord, TimestepRecord, EventRecord, MessageRecord,
    ExperimentRecord, SystemRecord, SystemVersionRecord
)
from ..storage.utils import detect_provider
from ..storage.exceptions import (
    SessionNotFoundError, SessionAlreadyExistsError,
    DatabaseConnectionError, SchemaInitializationError,
    QueryExecutionError, DataValidationError, TraceStorageError
)
from ..storage.config import get_config, DuckDBConfig
from . import schema

logger = logging.getLogger(__name__)


def convert_datetime_for_duckdb(dt: Optional[Union[datetime, str, float]]) -> Optional[str]:
    """Convert datetime to DuckDB-compatible timestamp string."""
    if dt is None:
        return None
    
    if isinstance(dt, datetime):
        return dt.isoformat()
    elif isinstance(dt, str):
        # Already a string, return as is
        return dt
    elif isinstance(dt, (int, float)):
        # Unix timestamp, convert to datetime then to string
        try:
            return datetime.fromtimestamp(dt).isoformat()
        except (ValueError, OSError):
            return None
    else:
        return str(dt)


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize objects to JSON, handling datetime objects."""
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    try:
        return json.dumps(obj, cls=DateTimeEncoder)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize object to JSON: {e}")
        return json.dumps(str(obj))


class DuckDBTraceManager(TraceStorage):
    """Manages DuckDB storage for trace data."""
    
    def __init__(self, db_path: Optional[str] = None, config: Optional[DuckDBConfig] = None, 
                 skip_schema_init: bool = False):
        """Initialize DuckDB manager with database path or config.
        
        Args:
            db_path: Path to database file (overrides config)
            config: DuckDB configuration object
            skip_schema_init: Skip schema initialization (for concurrent connections)
        """
        if config is None:
            config = get_config().duckdb
        self.config = config
        
        # Override db_path if provided
        self.db_path = db_path or config.db_path
        
        # Configure logging
        logging.getLogger(__name__).setLevel(config.log_level)
        
        self.conn = None
        self._connect()
        if not skip_schema_init:
            self.init_schema()
    
    def _connect(self):
        """Establish connection to DuckDB."""
        try:
            self.conn = duckdb.connect(self.db_path)
            
            # Apply performance settings if configured
            if self.config.memory_limit:
                self.conn.execute(f"SET memory_limit='{self.config.memory_limit}'")
            if self.config.threads:
                self.conn.execute(f"SET threads={self.config.threads}")
            
            # Additional performance optimizations for bulk operations
            self.conn.execute("SET preserve_insertion_order=false")  # Faster inserts
            self.conn.execute("SET checkpoint_threshold='1GB'")  # Less frequent checkpoints
            self.conn.execute("SET wal_autocheckpoint='1GB'")  # Larger WAL before checkpoint
            
            #logger.info(f"Connected to DuckDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise DatabaseConnectionError(f"Failed to connect to DuckDB at {self.db_path}: {e}")
    
    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
                logger.info("DuckDB connection closed")
            except Exception as e:
                logger.error(f"Error closing DuckDB connection: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def init_schema(self):
        """Initialize database schema."""
        try:
            # Create sequences
            for seq_name, seq_sql in schema.SEQUENCES.items():
                try:
                    self.conn.execute(seq_sql)
                    logger.debug(f"Created sequence {seq_name}")
                except Exception as e:
                    logger.warning(f"Failed to create sequence {seq_name}: {e}")
            
            # Create tables
            for table_name, table_sql in schema.TABLES.items():
                try:
                    self.conn.execute(table_sql)
                    logger.debug(f"Created table {table_name}")
                except Exception as e:
                    logger.error(f"Failed to create table {table_name}: {e}")
                    raise SchemaInitializationError(f"Failed to create table {table_name}: {e}")
            
            # Add experiment_id column if not exists (for backwards compatibility)
            self.conn.execute("""
                ALTER TABLE session_traces ADD COLUMN IF NOT EXISTS experiment_id VARCHAR;
            """)
            
            # Add foreign key constraint for experiment_id
            try:
                self.conn.execute("""
                    ALTER TABLE session_traces ADD CONSTRAINT fk_experiment 
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                """)
            except Exception as e:
                # Constraint might already exist
                logger.debug(f"Foreign key constraint might already exist: {e}")
            
            # Create indexes
            for idx_name, idx_sql in schema.INDEXES.items():
                try:
                    self.conn.execute(idx_sql)
                    logger.debug(f"Created index {idx_name}")
                except Exception as e:
                    logger.warning(f"Failed to create index {idx_name}: {e}")
            
            # Create analytics views if enabled
            if self.config.enable_analytics_views:
                self._create_analytics_views()
            #logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise SchemaInitializationError(f"Failed to initialize schema: {e}")
    
    def _create_analytics_views(self):
        """Create materialized views for analytics."""
        for view_name, view_sql in schema.VIEWS.items():
            self.conn.execute(view_sql)
    
    def insert_session_trace(self, trace):
        """Insert a complete session trace into DuckDB."""
        try:
            # Handle both abstractions.SessionTrace and session_tracer.SessionTrace
            session_id = getattr(trace, 'session_id', None)
            if not session_id and hasattr(trace, 'session_metadata'):
                session_id = trace.session_metadata[0].data.get("session_id") if trace.session_metadata else None
            if not session_id:
                session_id = f"session_{datetime.now().isoformat()}"
            
            # Check if session already exists
            existing = self.conn.execute(schema.SELECT["session_exists"], [session_id]).fetchone()
            
            if existing:
                logger.warning(f"Session {session_id} already exists, skipping insertion")
                return
            
            # Use bulk insert if available and beneficial
            if len(trace.session_time_steps) > 10 or len(getattr(trace, 'event_history', [])) > 50:
                self._insert_session_trace_bulk(trace)
                self.conn.commit()
                return
            
            #logger.info(f"Inserting session {session_id}")
            
            # Get timesteps from session_time_steps
            timesteps = trace.session_time_steps
            
            # Insert main session record
            created_at = getattr(trace, 'created_at', None)
            if isinstance(created_at, str):
                # Parse ISO format string to datetime
                from dateutil.parser import parse
                created_at = parse(created_at)
            elif not created_at:
                created_at = datetime.now()
                
            metadata = []
            if hasattr(trace, 'session_metadata'):
                metadata = [self._serialize_metadata(m) for m in trace.session_metadata]
            
            self.conn.execute(schema.INSERT["session_trace"], [
                session_id,
                created_at,
                len(timesteps),
                len(getattr(trace, 'event_history', [])),
                len(getattr(trace, 'message_history', [])),
                safe_json_serialize(metadata)
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
            #logger.info(f"Successfully inserted session {session_id}")
        except Exception as e:
            logger.error(f"Failed to insert session trace: {e}")
            try:
                self.conn.rollback()
            except Exception as rollback_error:
                logger.warning(f"Failed to rollback transaction: {rollback_error}")
            raise
    
    def _serialize_metadata(self, metadata: SessionMetadum) -> Dict[str, Any]:
        """Serialize session metadata to dictionary."""
        timestamp = getattr(metadata, 'timestamp', None)
        if timestamp and isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        
        return {
            "metadata_type": getattr(metadata, 'metadata_type', metadata.__class__.__name__),
            "data": getattr(metadata, 'data', {}),
            "timestamp": timestamp
        }
    
    def _insert_timestep(self, session_id: str, timestep: SessionTimeStep, index: int) -> int:
        """Insert a timestep and return its ID."""
        result = self.conn.execute(schema.INSERT["session_timestep"], [
            session_id,
            timestep.step_id,
            index,
            convert_datetime_for_duckdb(timestep.timestamp or datetime.now()),
            len(timestep.events),
            len(timestep.step_messages),
            safe_json_serialize(timestep.step_metadata) if timestep.step_metadata else None
        ]).fetchone()
        
        return result[0]
    
    def _insert_event(self, session_id: str, event: Union[CAISEvent, LMCAISEvent, EnvironmentEvent, RuntimeEvent], timestep_id_map: Dict[str, int]):
        """Insert an event into the database."""
        # Special handling for LMCAISEvent to keep event_type as 'cais'
        if isinstance(event, LMCAISEvent):
            event_type = EventType.CAIS
        elif isinstance(event, EnvironmentEvent):
            event_type = EventType.ENVIRONMENT
        elif isinstance(event, RuntimeEvent):
            event_type = EventType.RUNTIME
        else:
            event_type = EventType.CAIS  # Default for CAISEvent
        
        # Common fields
        params = {
            'session_id': session_id,
            'timestep_id': None,  # TODO: Map event to timestep
            'event_type': event_type.value,
            'system_instance_id': getattr(event, 'system_instance_id', ''),
            'event_time': convert_datetime_for_duckdb(datetime.now()),
            'message_time': getattr(event.time_record, 'message_time', None) if hasattr(event, 'time_record') else None,
        }
        
        # Type-specific fields
        if isinstance(event, CAISEvent):
            # Check if it's the extended LMCAISEvent with direct fields
            if isinstance(event, LMCAISEvent):
                # Direct fields from LMCAISEvent
                if event.model_name:
                    params['model_name'] = event.model_name
                    # Detect provider
                    params['provider'] = detect_provider(event.model_name).value
                if event.span_id:
                    params['span_id'] = event.span_id
                if event.trace_id:
                    params['trace_id'] = event.trace_id
                if event.prompt_tokens is not None:
                    params['prompt_tokens'] = event.prompt_tokens
                if event.completion_tokens is not None:
                    params['completion_tokens'] = event.completion_tokens
                if event.total_tokens is not None:
                    params['total_tokens'] = event.total_tokens
                if event.cost is not None:
                    params['cost'] = event.cost
                if event.latency_ms is not None:
                    params['latency_ms'] = event.latency_ms
                
            # Then try to extract from llm_call_records (OTel path)
            if not params.get('model_name') and event.llm_call_records:
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
                # Handle JSON string
                if isinstance(state, str):
                    try:
                        state = json.loads(state)
                    except Exception as e:
                        print(f"Failed to parse system_state_before: {e}")
                        state = {}
                if isinstance(state, dict):
                    # Extract model info
                    if not params.get('model_name'):
                        model = state.get('gen_ai.request.model') or state.get('llm.model_name')
                        if model:
                            params['model_name'] = model
                    
                    # Detect provider if not already set
                    if not params.get('provider') and params.get('model_name'):
                        params['provider'] = detect_provider(params['model_name']).value
            
            # Extract token usage from system_state_after (where response data is stored)
            if hasattr(event, 'system_state_after') and event.system_state_after:
                state_after = event.system_state_after
                # Handle JSON string
                if isinstance(state_after, str):
                    try:
                        state_after = json.loads(state_after)
                    except:
                        state_after = {}
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
        # Handle system states - they might already be JSON strings
        if hasattr(event, 'system_state_before'):
            state_before = event.system_state_before
            if isinstance(state_before, str):
                params['system_state_before'] = state_before  # Already JSON
            else:
                params['system_state_before'] = safe_json_serialize(state_before)
        else:
            params['system_state_before'] = None
            
        if hasattr(event, 'system_state_after'):
            state_after = event.system_state_after
            if isinstance(state_after, str):
                params['system_state_after'] = state_after  # Already JSON
            else:
                params['system_state_after'] = safe_json_serialize(state_after)
        else:
            params['system_state_after'] = None
            
        if hasattr(event, 'metadata'):
            metadata = event.metadata
            if isinstance(metadata, str):
                params['metadata'] = metadata  # Already JSON
            else:
                params['metadata'] = safe_json_serialize(metadata)
        else:
            params['metadata'] = None
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
            params['event_metadata'] = safe_json_serialize(metadata_list)
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
        
        self.conn.execute(schema.INSERT["message"], [
            session_id,
            None,  # TODO: Map message to timestep
            message.message_type,
            safe_json_serialize(message.content),
            convert_datetime_for_duckdb(datetime.now()),
            convert_datetime_for_duckdb(getattr(message.time_record, 'event_time', None) if hasattr(message, 'time_record') else None),
            getattr(message.time_record, 'message_time', None) if hasattr(message, 'time_record') else None,
        ])
    
    def query_traces(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute analytical query and return results as DataFrame."""
        try:
            if self.config.log_queries:
                logger.debug(f"Executing query: {query}")
            else:
                logger.debug(f"Executing query: {query[:100]}...")
            
            if params:
                result = self.conn.execute(query, params).df()
            else:
                result = self.conn.execute(query).df()
            
            logger.debug(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise QueryExecutionError(query, e)
    
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
    
    def batch_upload(self, traces: List[SessionTrace], batch_size: Optional[int] = None):
        """Upload multiple traces efficiently in batches."""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        uploaded = 0
        failed = 0
        
        # Use transaction for better performance
        self.conn.begin()
        
        try:
            # Use tqdm progress bar for batch processing
            with tqdm(total=len(traces), desc="Uploading traces to DuckDB", unit="traces") as pbar:
                for i, trace in enumerate(traces):
                    try:
                        self._insert_session_trace_bulk(trace)
                        uploaded += 1
                        pbar.update(1)
                        
                        # Commit every batch_size traces
                        if (i + 1) % batch_size == 0:
                            self.conn.commit()
                            self.conn.begin()
                            # Silent commit, progress shown in tqdm bar
                    except Exception as e:
                        logger.error(f"Failed to upload trace {i}: {e}")
                        failed += 1
                        pbar.update(1)
            
            # Commit any remaining traces
            self.conn.commit()
            
            logger.info(f"Batch upload complete: {uploaded} successful, {failed} failed out of {len(traces)} total")
            if failed > 0:
                raise TraceStorageError(f"Batch upload partially failed: {failed} traces could not be inserted")
        except Exception as e:
            self.conn.rollback()
            raise
    
    def _insert_session_trace_bulk(self, trace):
        """Insert a session trace optimized for bulk operations."""
        # Similar to insert_session_trace but without individual commits
        session_id = getattr(trace, 'session_id', None)
        if not session_id and hasattr(trace, 'session_metadata'):
            session_id = trace.session_metadata[0].data.get("session_id") if trace.session_metadata else None
        if not session_id:
            session_id = f"session_{datetime.now().isoformat()}"
        
        # Check if session already exists
        existing = self.conn.execute(schema.SELECT["session_exists"], [session_id]).fetchone()
        if existing:
            logger.debug(f"Session {session_id} already exists, skipping")
            return
        
        timesteps = trace.session_time_steps
        
        # Insert main session record
        created_at = getattr(trace, 'created_at', None)
        if isinstance(created_at, str):
            from dateutil.parser import parse
            created_at = parse(created_at)
        elif not created_at:
            created_at = datetime.now()
            
        metadata = []
        if hasattr(trace, 'session_metadata'):
            metadata = [self._serialize_metadata(m) for m in trace.session_metadata]
        
        self.conn.execute(schema.INSERT["session_trace"], [
            session_id,
            created_at,
            len(timesteps),
            len(getattr(trace, 'event_history', [])),
            len(getattr(trace, 'message_history', [])),
            safe_json_serialize(metadata)
        ])
        
        # Prepare bulk data for timesteps, events, and messages
        timestep_id_map = {}
        
        # Bulk insert timesteps using executemany
        if timesteps:
            timestep_data = []
            for idx, timestep in enumerate(timesteps):
                timestep_data.append((
                    session_id,
                    timestep.step_id,
                    idx,
                    convert_datetime_for_duckdb(timestep.timestamp or datetime.now()),
                    len(timestep.events),
                    len(timestep.step_messages),
                    safe_json_serialize(timestep.step_metadata) if timestep.step_metadata else None
                ))
            
            # Use executemany for bulk insert - much faster!
            self.conn.executemany(
                "INSERT INTO session_timesteps (session_id, step_id, step_index, timestamp, num_events, num_messages, step_metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                timestep_data
            )
            
            # Now get the IDs that were created - order by step_index to maintain order
            results = self.conn.execute(
                "SELECT id, step_id FROM session_timesteps WHERE session_id = ? ORDER BY step_index",
                [session_id]
            ).fetchall()
            
            # Build the mapping more efficiently
            timestep_id_map = {row[1]: row[0] for row in results}
        
        # Bulk insert events using executemany
        if hasattr(trace, 'event_history') and trace.event_history:
            event_data = []
            for event in trace.event_history:
                event_row = self._prepare_event_data(session_id, event, timestep_id_map)
                if event_row:
                    event_data.append(tuple(event_row))
            
            # Use executemany for bulk insert
            if event_data:
                self.conn.executemany(
                    """INSERT INTO events (
                        session_id, timestep_id, event_type, system_instance_id, event_time, message_time,
                        span_id, trace_id, model_name, provider, prompt_tokens, completion_tokens,
                        total_tokens, cost, latency_ms, reward, terminated,
                        system_state_before, system_state_after, metadata, event_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    event_data
                )
        
        # Bulk insert messages using executemany
        if hasattr(trace, 'message_history') and trace.message_history:
            message_data = []
            for message in trace.message_history:
                message_data.append((
                    session_id,
                    None,  # timestep_id - TODO: map to timestep
                    message.message_type,
                    safe_json_serialize(message.content),
                    convert_datetime_for_duckdb(datetime.now()),
                    getattr(message.time_record, 'event_time', None) if hasattr(message, 'time_record') else None,
                    getattr(message.time_record, 'message_time', None) if hasattr(message, 'time_record') else None,
                ))
            
            # Use executemany for bulk insert
            if message_data:
                self.conn.executemany(
                    "INSERT INTO messages (session_id, timestep_id, message_type, content, timestamp, event_time, message_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    message_data
                )
    
    def _prepare_event_data(self, session_id: str, event: Union[CAISEvent, LMCAISEvent, EnvironmentEvent, RuntimeEvent], 
                           timestep_id_map: Dict[str, int]) -> Optional[List[Any]]:
        """Prepare event data for bulk insert."""
        # Determine event type
        if isinstance(event, LMCAISEvent):
            event_type = EventType.CAIS
        elif isinstance(event, EnvironmentEvent):
            event_type = EventType.ENVIRONMENT
        elif isinstance(event, RuntimeEvent):
            event_type = EventType.RUNTIME
        else:
            event_type = EventType.CAIS
        
        # Initialize all fields with None
        row = [None] * 21  # 21 fields in the bulk insert query
        
        # Common fields
        row[0] = session_id
        row[1] = None  # timestep_id - TODO: map to timestep
        row[2] = event_type.value
        row[3] = getattr(event, 'system_instance_id', '')
        row[4] = datetime.now()  # event_time
        row[5] = getattr(event.time_record, 'message_time', None) if hasattr(event, 'time_record') else None
        
        # Type-specific fields
        if isinstance(event, (CAISEvent, LMCAISEvent)):
            # CAIS event fields
            if isinstance(event, LMCAISEvent):
                row[6] = event.span_id
                row[7] = event.trace_id
                row[8] = event.model_name
                row[9] = detect_provider(event.model_name).value if event.model_name else None
                row[10] = event.prompt_tokens
                row[11] = event.completion_tokens
                row[12] = event.total_tokens
                row[13] = event.cost
                row[14] = event.latency_ms
            
            # Try to extract from other sources if still missing
            if not row[8] and hasattr(event, 'system_state_before'):
                state = event.system_state_before
                if isinstance(state, str):
                    try:
                        state = json.loads(state)
                    except:
                        state = {}
                if isinstance(state, dict):
                    row[8] = state.get('gen_ai.request.model') or state.get('llm.model_name')
                    if row[8] and not row[9]:
                        row[9] = detect_provider(row[8]).value
        
        elif isinstance(event, EnvironmentEvent):
            row[15] = event.reward  # reward
            row[16] = event.terminated  # terminated
        
        # Common fields - system states and metadata
        if hasattr(event, 'system_state_before'):
            state_before = event.system_state_before
            row[17] = state_before if isinstance(state_before, str) else safe_json_serialize(state_before)
        
        if hasattr(event, 'system_state_after'):
            state_after = event.system_state_after
            row[18] = state_after if isinstance(state_after, str) else safe_json_serialize(state_after)
        
        if hasattr(event, 'metadata'):
            metadata = event.metadata
            row[19] = metadata if isinstance(metadata, str) else safe_json_serialize(metadata)
        
        if hasattr(event, 'event_metadata') and event.event_metadata:
            metadata_list = []
            for em in event.event_metadata:
                if hasattr(em, '__dict__'):
                    metadata_list.append(vars(em))
                elif hasattr(em, 'to_dict'):
                    metadata_list.append(em.to_dict())
                else:
                    metadata_list.append(str(em))
            row[20] = safe_json_serialize(metadata_list)
        
        return row
    
    # Experiment and System Management Methods
    
    def create_system(self, system_id: str, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new system."""
        self.conn.execute(schema.INSERT["system"], [system_id, name, description])
        return {"id": system_id, "name": name, "description": description}
    
    def create_system_version(self, version_id: str, system_id: str, branch: str, 
                            commit: str, description: str = "") -> Dict[str, Any]:
        """Create a new system version."""
        created_at = datetime.utcnow()
        self.conn.execute(schema.INSERT["system_version"], 
                         [version_id, system_id, branch, commit, created_at, description])
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
        
        # Create and validate experiment record
        experiment = ExperimentRecord(
            id=experiment_id,
            name=name,
            description=description,
            created_at=created_at,
            updated_at=updated_at,
            system_versions=system_versions
        )
        
        # Create experiment
        self.conn.execute(schema.INSERT["experiment"], 
                         [experiment.id, experiment.name, experiment.description, 
                          experiment.created_at, experiment.updated_at])
        
        # Link system versions if provided
        if experiment.system_versions:
            for sv in experiment.system_versions:
                self.conn.execute(schema.INSERT["experimental_system"],
                                [experiment.id, sv["system_id"], sv["system_version_id"]])
        
        return experiment.model_dump()
    
    def link_session_to_experiment(self, session_id: str, experiment_id: str):
        """Link a session to an experiment."""
        try:
            # Workaround for DuckDB UPDATE bug: use DELETE + INSERT instead of UPDATE
            # This avoids the "Duplicate key" error that occurs when updating experiment_id
            
            # First, get the current session data
            session_data = self.conn.execute("""
                SELECT session_id, created_at, num_timesteps, num_events, num_messages, metadata
                FROM session_traces WHERE session_id = ?
            """, [session_id]).fetchone()
            
            if not session_data:
                raise SessionNotFoundError(session_id)
            
            # Delete the existing session
            self.conn.execute(schema.DELETE["session_trace"], [session_id])
            
            # Insert the session with the new experiment_id
            self.conn.execute(
                "INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata, experiment_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [session_id, session_data[1], session_data[2], session_data[3], session_data[4], session_data[5], experiment_id]
            )
            
            self.conn.commit()
            logger.info(f"Linked session {session_id} to experiment {experiment_id}")
        except Exception as e:
            logger.error(f"Failed to link session to experiment: {e}")
            self.conn.rollback()
            raise
    
    def get_experiment_sessions(self, experiment_id: str) -> pd.DataFrame:
        """Get all sessions for an experiment."""
        return self.conn.execute(schema.SELECT["experiment_sessions"], [experiment_id]).df()
    
    def get_experiments_by_system_version(self, system_version_id: str) -> pd.DataFrame:
        """Get all experiments using a specific system version."""
        return self.conn.execute(schema.SELECT["experiments_by_system_version"], [system_version_id]).df()