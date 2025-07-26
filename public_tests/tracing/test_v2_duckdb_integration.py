"""
Test DuckDB integration for tracing_v2.
"""
import json
import tempfile
from pathlib import Path
from datetime import datetime
import pytest

from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import (
    SessionTrace, SessionTimeStep, CAISEvent, EnvironmentEvent, 
    RuntimeEvent, SessionEventMessage, SessionMetadum, TimeRecord
)


class TestDuckDBTraceManager:
    """Test DuckDB trace manager functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def manager(self, temp_db):
        """Create DuckDB manager instance."""
        return DuckDBTraceManager(temp_db)
    
    def test_init_schema(self, manager):
        """Test schema initialization."""
        # Check tables exist
        tables = manager.conn.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).df()
        
        expected_tables = {'session_traces', 'session_timesteps', 'events', 'messages'}
        assert set(tables['table_name']) >= expected_tables
    
    def test_insert_simple_trace(self, manager):
        """Test inserting a simple session trace."""
        # Create a simple trace
        trace = SessionTrace(session_id="test-session-001")
        
        # Add metadata
        metadata = SessionMetadum(
            metadata_type="test_metadata",
            data={"key": "value"}
        )
        trace.add_metadata(metadata)
        
        # Add a timestep
        timestep = SessionTimeStep(step_id="step-1")
        
        # Add CAIS event
        cais_event = CAISEvent(
            system_instance_id="test-agent",
            time_record=TimeRecord(event_time=datetime.now().isoformat(), message_time=0),
            model_name="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        timestep.add_event(cais_event)
        trace.add_timestep(timestep)
        trace.add_event(cais_event)
        
        # Insert trace
        manager.insert_session_trace(trace)
        
        # Verify insertion
        sessions = manager.conn.execute("SELECT * FROM session_traces").df()
        assert len(sessions) == 1
        assert sessions.iloc[0]['session_id'] == 'test-session-001'
        assert sessions.iloc[0]['num_timesteps'] == 1
        assert sessions.iloc[0]['num_events'] == 1
    
    def test_insert_complex_trace(self, manager):
        """Test inserting a complex trace with multiple event types."""
        trace = SessionTrace(session_id="complex-session-001")
        
        # Add multiple timesteps
        for i in range(3):
            timestep = SessionTimeStep(step_id=f"step-{i}")
            
            # Add different event types
            if i == 0:
                # CAIS event
                event = CAISEvent(
                    system_instance_id="llm-agent",
                    time_record=TimeRecord(event_time=datetime.now().isoformat(), message_time=i),
                    model_name="gpt-4",
                    prompt_tokens=100 + i * 10,
                    completion_tokens=50 + i * 5,
                    total_tokens=150 + i * 15,
                    metadata={"step": i}
                )
            elif i == 1:
                # Environment event
                event = EnvironmentEvent(
                    system_instance_id="env",
                    time_record=TimeRecord(event_time=datetime.now().isoformat(), message_time=i),
                    reward=0.5,
                    terminated=False,
                    metadata={"step": i}
                )
            else:
                # Runtime event
                event = RuntimeEvent(
                    system_instance_id="runtime",
                    time_record=TimeRecord(event_time=datetime.now().isoformat(), message_time=i),
                    actions=["move_right", "collect"],
                    metadata={"step": i}
                )
            
            timestep.add_event(event)
            trace.add_timestep(timestep)
            trace.add_event(event)
            
            # Add message
            message = SessionEventMessage(
                content={"text": f"Message {i}"},
                message_type="test_message",
                time_record=TimeRecord(event_time=datetime.now().isoformat(), message_time=i)
            )
            timestep.add_message(message)
            trace.add_message(message)
        
        # Insert trace
        manager.insert_session_trace(trace)
        
        # Verify data
        sessions = manager.conn.execute("SELECT * FROM session_traces").df()
        assert len(sessions) == 1
        assert sessions.iloc[0]['num_timesteps'] == 3
        assert sessions.iloc[0]['num_events'] == 3
        assert sessions.iloc[0]['num_messages'] == 3
        
        # Check events
        events = manager.conn.execute("SELECT * FROM events ORDER BY message_time").df()
        assert len(events) == 3
        assert events.iloc[0]['event_type'] == 'cais'
        assert events.iloc[1]['event_type'] == 'environment'
        assert events.iloc[2]['event_type'] == 'runtime'
    
    def test_query_traces(self, manager):
        """Test querying traces."""
        # Insert test data
        trace = SessionTrace(session_id="query-test-001")
        
        # Add CAIS events with different models
        for model in ["gpt-4", "claude-3", "gpt-4"]:
            event = CAISEvent(
                model_name=model,
                provider="openai" if "gpt" in model else "anthropic",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost=0.01
            )
            trace.add_event(event)
        
        manager.insert_session_trace(trace)
        
        # Test custom query
        df = manager.query_traces("""
            SELECT model_name, COUNT(*) as count 
            FROM events 
            WHERE event_type = 'cais' 
            GROUP BY model_name
        """)
        
        assert len(df) == 2  # gpt-4 and claude-3
        assert df[df['model_name'] == 'gpt-4']['count'].iloc[0] == 2
        assert df[df['model_name'] == 'claude-3']['count'].iloc[0] == 1
    
    def test_model_usage_stats(self, manager):
        """Test model usage statistics view."""
        # Insert multiple sessions with different models
        for i in range(5):
            trace = SessionTrace(session_id=f"stats-session-{i}")
            
            # Add events
            for j in range(3):
                event = CAISEvent(
                    model_name="gpt-4" if i % 2 == 0 else "claude-3",
                    provider="openai" if i % 2 == 0 else "anthropic",
                    prompt_tokens=100 + j * 10,
                    completion_tokens=50 + j * 5,
                    total_tokens=150 + j * 15,
                    cost=0.01 + j * 0.001,
                    latency_ms=100 + j * 50
                )
                trace.add_event(event)
            
            manager.insert_session_trace(trace)
        
        # Get model usage stats
        stats = manager.get_model_usage()
        
        assert len(stats) == 2  # Two models
        gpt4_stats = stats[stats['model_name'] == 'gpt-4'].iloc[0]
        assert gpt4_stats['call_count'] == 9  # 3 sessions * 3 calls
        assert gpt4_stats['provider'] == 'openai'
    
    def test_expensive_calls(self, manager):
        """Test finding expensive calls."""
        trace = SessionTrace(session_id="expensive-test")
        
        # Add events with varying costs
        costs = [0.001, 0.01, 0.1, 0.5, 0.02]
        for i, cost in enumerate(costs):
            event = CAISEvent(
                model_name="gpt-4",
                cost=cost,
                prompt_tokens=1000 * (i + 1),
                completion_tokens=500 * (i + 1)
            )
            trace.add_event(event)
        
        manager.insert_session_trace(trace)
        
        # Find expensive calls
        expensive = manager.get_expensive_calls(0.05)
        
        assert len(expensive) == 2  # 0.1 and 0.5
        assert expensive.iloc[0]['cost'] == 0.5  # Ordered by cost DESC
        assert expensive.iloc[1]['cost'] == 0.1
    
    def test_export_to_parquet(self, manager, temp_db):
        """Test exporting to Parquet format."""
        # Insert test data
        trace = SessionTrace(session_id="export-test")
        event = CAISEvent(model_name="gpt-4", prompt_tokens=100)
        trace.add_event(event)
        manager.insert_session_trace(trace)
        
        # Export to parquet
        with tempfile.TemporaryDirectory() as tmpdir:
            manager.export_to_parquet(tmpdir)
            
            # Check files exist
            export_path = Path(tmpdir)
            assert (export_path / "session_traces.parquet").exists()
            assert (export_path / "events.parquet").exists()
            assert (export_path / "messages.parquet").exists()
            assert (export_path / "session_timesteps.parquet").exists()
    
    def test_batch_upload(self, manager):
        """Test batch uploading traces."""
        traces = []
        for i in range(10):
            trace = SessionTrace(session_id=f"batch-{i}")
            event = CAISEvent(model_name=f"model-{i}", prompt_tokens=100 * i)
            trace.add_event(event)
            traces.append(trace)
        
        # Batch upload
        manager.batch_upload(traces)
        
        # Verify all uploaded
        sessions = manager.conn.execute("SELECT COUNT(*) as count FROM session_traces").df()
        assert sessions.iloc[0]['count'] == 10
        
        events = manager.conn.execute("SELECT COUNT(*) as count FROM events").df()
        assert events.iloc[0]['count'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])